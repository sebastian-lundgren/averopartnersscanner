import csv
import io
import json
from datetime import datetime

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse
from openpyxl import Workbook
from sqlalchemy.orm import Session, joinedload

from app import models
from app.database import get_db

router = APIRouter(prefix="/api/export", tags=["export"])

EXPORT_FIELDS = [
    "adresse_eller_kunde_id",
    "bildefilnavn",
    "foreslatt_status",
    "endelig_status",
    "annotasjon_label",
    "bbox_forslag_modell_json",
    "bbox_trening_json",
    "confidence",
    "manuell_overstyring",
    "kommentar",
    "feiltype",
    "siste_vurderingsdato",
    "modellversjon",
    "evidensbilde_sti",
]


def _rows(db: Session):
    preds = (
        db.query(models.Prediction)
        .options(
            joinedload(models.Prediction.image).joinedload(models.ImageAsset.address),
            joinedload(models.Prediction.model_version),
            joinedload(models.Prediction.review),
        )
        .order_by(models.Prediction.created_at.desc())
        .all()
    )
    pred_ids = [p.id for p in preds]
    te_map: dict[int, models.TrainingExample] = {}
    if pred_ids:
        for te in (
            db.query(models.TrainingExample)
            .filter(models.TrainingExample.source_prediction_id.in_(pred_ids))
            .order_by(models.TrainingExample.created_at.desc())
            .all()
        ):
            if te.source_prediction_id not in te_map:
                te_map[te.source_prediction_id] = te
    for p in preds:
        img = p.image
        addr = img.address if img else None
        addr_or_cust = ""
        if addr:
            addr_or_cust = addr.address_line or addr.customer_id or str(addr.id)
        rev = p.review
        te = te_map.get(p.id)
        tags = te.tags_json if te and isinstance(te.tags_json, dict) else {}
        ann = str(tags.get("annotation_label") or "") if tags else ""
        bbox_tr = None
        if tags:
            bm = tags.get("bboxes_norm")
            if isinstance(bm, list) and len(bm) > 0:
                bbox_tr = bm
            else:
                bbox_tr = tags.get("bbox_norm")
        bbox_tr_s = json.dumps(bbox_tr, ensure_ascii=True) if bbox_tr is not None else ""
        bbox_prop_s = json.dumps(p.bbox_json, ensure_ascii=True) if p.bbox_json else ""
        yield {
            "adresse_eller_kunde_id": addr_or_cust,
            "bildefilnavn": img.original_filename if img else "",
            "foreslatt_status": p.predicted_status,
            "endelig_status": rev.final_status if rev else "",
            "annotasjon_label": ann,
            "bbox_forslag_modell_json": bbox_prop_s,
            "bbox_trening_json": bbox_tr_s,
            "confidence": p.confidence,
            "manuell_overstyring": "ja" if rev and rev.was_override else "nei",
            "kommentar": (rev.comment if rev else "") or "",
            "feiltype": (rev.error_type if rev else "") or "",
            "siste_vurderingsdato": (
                rev.decided_at.isoformat() if rev and rev.decided_at else p.created_at.isoformat()
            ),
            "modellversjon": p.model_version.version_tag if p.model_version else "",
            "evidensbilde_sti": img.evidence_crop_path if img else "",
        }


@router.get("/csv")
def export_csv(db: Session = Depends(get_db)):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=EXPORT_FIELDS)
    w.writeheader()
    for row in _rows(db):
        w.writerow(row)
    buf.seek(0)
    filename = f"alarm_skilt_export_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/xlsx")
def export_xlsx(db: Session = Depends(get_db)):
    wb = Workbook()
    ws = wb.active
    ws.title = "Vurderinger"
    ws.append(EXPORT_FIELDS)
    for row in _rows(db):
        ws.append([row[k] for k in EXPORT_FIELDS])
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    filename = f"alarm_skilt_export_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.xlsx"
    return Response(
        content=bio.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
