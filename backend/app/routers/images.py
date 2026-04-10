import logging
import uuid
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from PIL import Image
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app import models, schemas
from app.config import settings
from app.database import SessionLocal, get_db
from app.services.active_learning import refresh_prediction_priority
from app.services.blob_storage import materialize_local_path, store_upload_bytes
from app.services.bbox_multi import first_bbox, yolo_trusted_primary_from_bbox_json
from app.services.evidence import save_evidence_crop
from app.services.settings_store import get_thresholds
from app.services.yolo_service import run_yolov8_on_image

router = APIRouter(prefix="/api/images", tags=["images"])
log = logging.getLogger(__name__)
_MAX_LIST_WINDOW = 1000


def _yolo_scan_model_version(db: Session) -> models.ModelVersion | None:
    """Samme merkelapp som scanner/ingest-yolo — konsistent bbox-kilde i review."""
    return db.query(models.ModelVersion).filter_by(version_tag="yolov8s-scan").first()


def _run_predictions_after_upload(image_ids: list[int]) -> None:
    """Etter opplasting: ML-prediksjon eller plassholder uten torch/HF når ml_inference_enabled=false."""
    db = SessionLocal()
    try:
        active_mv = db.query(models.ModelVersion).filter(models.ModelVersion.is_active.is_(True)).first()
        if not active_mv:
            return
        yolo_mv = _yolo_scan_model_version(db)
        thr = get_thresholds(db)
        strong = int(thr["threshold_strong_sign"])
        for iid in image_ids:
            try:
                img = db.get(models.ImageAsset, iid)
                if not img:
                    continue
                if db.query(models.Prediction).filter(models.Prediction.image_id == iid).first():
                    continue
                if not settings.ml_inference_enabled:
                    pred = models.Prediction(
                        image_id=img.id,
                        model_version_id=active_mv.id,
                        predicted_status=models.ReviewStatus.UKLART.value,
                        confidence=0,
                        bbox_json=None,
                        rationale=(
                            "Automatisk prediksjon er av (ML_INFERENCE_ENABLED=false). "
                            "Merking skjer manuelt i review."
                        ),
                        needs_review=True,
                        review_completed=False,
                    )
                    db.add(pred)
                    db.flush()
                    refresh_prediction_priority(db, pred)
                    db.commit()
                    continue

                pred_mv = yolo_mv or active_mv
                local_img, tmp_del = materialize_local_path(img.stored_path, suffix=".upload")
                try:
                    yo = run_yolov8_on_image(str(local_img), db_session=db)
                finally:
                    if tmp_del:
                        local_img.unlink(missing_ok=True)

                fb = first_bbox(yo.bbox_json)
                if fb and yolo_trusted_primary_from_bbox_json(yo.bbox_json):
                    ev_name = f"ev_{img.id}_{uuid.uuid4().hex[:8]}.jpg"
                    evidence_rel = save_evidence_crop(img.stored_path, ev_name, fb)
                    if evidence_rel:
                        img.evidence_crop_path = evidence_rel

                needs_review = yo.needs_review or (
                    yo.confidence < strong
                    or yo.predicted_status != models.ReviewStatus.SKILT_FUNNET.value
                )
                pred = models.Prediction(
                    image_id=img.id,
                    model_version_id=pred_mv.id,
                    predicted_status=yo.predicted_status,
                    confidence=yo.confidence,
                    bbox_json=yo.bbox_json,
                    rationale=yo.rationale,
                    needs_review=needs_review,
                    review_completed=False,
                )
                db.add(pred)
                db.flush()
                refresh_prediction_priority(db, pred)
                db.commit()
            except Exception:
                log.exception("Prediksjon etter opplasting feilet for image_id=%s", iid)
                db.rollback()
    finally:
        db.close()


def _active_model(db: Session) -> models.ModelVersion:
    m = db.query(models.ModelVersion).filter(models.ModelVersion.is_active.is_(True)).first()
    if not m:
        raise HTTPException(500, "Ingen aktiv modellversjon")
    return m


@router.post("/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    address_id: int | None = Form(None),
    customer_id: str | None = Form(None),
    address_line: str | None = Form(None),
    is_temporary_candidate: bool = Form(False),
    db: Session = Depends(get_db),
):
    """
    Enkelt- eller batch-opplasting. Knyt til adresse/kunde ved behov.
    """
    if not files:
        raise HTTPException(400, "Ingen filer")

    address = None
    if address_id:
        address = db.get(models.AddressRecord, address_id)
        if not address:
            raise HTTPException(404, "Adresse ikke funnet")
    elif customer_id or address_line:
        address = models.AddressRecord(customer_id=customer_id, address_line=address_line)
        db.add(address)
        db.flush()

    _active_model(db)
    created: list[dict] = []
    pending_prediction_ids: list[int] = []

    for uf in files:
        if not uf.content_type or not uf.content_type.startswith("image/"):
            continue
        raw = await uf.read()
        ct = uf.content_type or "image/jpeg"
        stored_path, orig_name = store_upload_bytes(raw, uf.filename or "image.jpg", ct)
        try:
            im = Image.open(BytesIO(raw))
            w, h = im.size
        except Exception:
            w, h = None, None

        img = models.ImageAsset(
            address_id=address.id if address else None,
            original_filename=orig_name,
            stored_path=stored_path,
            mime_type=ct,
            width=w,
            height=h,
            is_temporary_candidate=is_temporary_candidate,
        )
        db.add(img)
        db.flush()
        pending_prediction_ids.append(img.id)
        created.append(
            {
                "image_id": img.id,
                "prediction_id": None,
                "prediction_pending": True,
            }
        )

    db.commit()
    if pending_prediction_ids:
        _run_predictions_after_upload(pending_prediction_ids)
    return {"ok": True, "items": created, "address_id": address.id if address else None}


@router.get("/library", response_model=list[schemas.LibraryImageOut])
def library(
    skip: int = 0,
    limit: int = 100,
    home_status: str | None = Query(
        None,
        description="Filtrer etter lagret boligstatus (AddressRecord.final_human_status).",
    ),
    db: Session = Depends(get_db),
):
    skip = max(0, int(skip))
    limit = max(1, int(limit))
    if skip >= _MAX_LIST_WINDOW:
        return []
    limit = min(limit, _MAX_LIST_WINDOW - skip)

    hs = (home_status or "all").strip().lower()
    if hs not in ("all", "skilt_funnet", "trenger_manuell", "uklart"):
        raise HTTPException(
            400,
            "home_status må være all, skilt_funnet, trenger_manuell eller uklart",
        )
    lib_seq = (
        func.row_number()
        .over(
            partition_by=func.coalesce(
                models.ImageAsset.address_id,
                -models.ImageAsset.id,
            ),
            order_by=models.ImageAsset.uploaded_at.asc(),
        )
        .label("lib_seq")
    )
    q = (
        db.query(models.ImageAsset, models.AddressRecord, lib_seq)
        .outerjoin(
            models.AddressRecord,
            models.ImageAsset.address_id == models.AddressRecord.id,
        )
    )
    if hs == "skilt_funnet":
        q = q.filter(
            models.ImageAsset.address_id.isnot(None),
            models.AddressRecord.final_human_status == models.ReviewStatus.SKILT_FUNNET.value,
        )
    elif hs == "trenger_manuell":
        q = q.filter(
            models.ImageAsset.address_id.isnot(None),
            models.AddressRecord.final_human_status == models.ReviewStatus.TRENGER_MANUELL.value,
        )
    elif hs == "uklart":
        q = q.filter(
            models.ImageAsset.address_id.isnot(None),
            or_(
                models.AddressRecord.final_human_status == models.ReviewStatus.UKLART.value,
                models.AddressRecord.final_human_status.is_(None),
            ),
        )
    q = q.order_by(models.ImageAsset.uploaded_at.desc())
    rows = q.offset(skip).limit(limit).all()
    out: list[schemas.LibraryImageOut] = []
    for img, addr, seq in rows:
        base = schemas.ImageAssetOut.model_validate(img).model_dump()
        line = addr.address_line if addr else None
        st = addr.final_human_status if addr else None
        n = int(seq)
        display_name = f"{line or 'Uten adresse'} #{n}"
        base.update(
            {
                "address_line": line,
                "address_final_status": st,
                "sequence_within_address": n,
                "display_name": display_name,
            },
        )
        out.append(schemas.LibraryImageOut(**base))
    return out


@router.get("/{image_id}", response_model=schemas.ImageAssetOut)
def get_image(image_id: int, db: Session = Depends(get_db)):
    img = db.get(models.ImageAsset, image_id)
    if not img:
        raise HTTPException(404)
    return img


@router.get("/{image_id}/predictions", response_model=list[schemas.PredictionOut])
def image_predictions(image_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.Prediction)
        .filter(models.Prediction.image_id == image_id)
        .order_by(models.Prediction.created_at.desc())
        .all()
    )


@router.post("/{image_id}/send-to-review")
def send_image_to_review(image_id: int, db: Session = Depends(get_db)):
    img = db.get(models.ImageAsset, image_id)
    if not img:
        raise HTTPException(404, "Bilde ikke funnet")

    pred = (
        db.query(models.Prediction)
        .filter(models.Prediction.image_id == img.id)
        .order_by(models.Prediction.created_at.desc())
        .first()
    )
    if pred is None:
        mv = _active_model(db)
        pred = models.Prediction(
            image_id=img.id,
            model_version_id=mv.id,
            predicted_status=models.ReviewStatus.UKLART.value,
            confidence=0,
            bbox_json=None,
            rationale="Sendt tilbake fra bibliotek for ny merking",
            needs_review=True,
            review_completed=False,
        )
        db.add(pred)
        db.flush()
    else:
        rev = db.query(models.ReviewDecision).filter_by(prediction_id=pred.id).first()
        if rev:
            db.delete(rev)
        pred.review_completed = False
        pred.needs_review = True
        pred.claimed_by = None
        pred.claimed_at = None

    refresh_prediction_priority(db, pred)
    db.commit()
    return {"ok": True, "image_id": img.id, "prediction_id": pred.id}


@router.delete("/{image_id}")
def delete_library_image(image_id: int, db: Session = Depends(get_db)):
    img = db.get(models.ImageAsset, image_id)
    if not img:
        raise HTTPException(404, "Bilde ikke funnet")

    pred_rows = (
        db.query(models.Prediction.id)
        .filter(models.Prediction.image_id == img.id)
        .all()
    )
    pred_ids = [int(r[0]) for r in pred_rows]

    in_training_from_image = (
        db.query(models.TrainingExample.id)
        .filter(models.TrainingExample.image_id == img.id)
        .first()
    )
    in_training_from_prediction = (
        db.query(models.TrainingExample.id)
        .filter(models.TrainingExample.source_prediction_id.in_(pred_ids))
        .first()
        if pred_ids
        else None
    )
    in_library = (
        db.query(models.TrainingLibraryEntry.id)
        .filter(models.TrainingLibraryEntry.image_id == img.id)
        .first()
    )
    if in_training_from_image or in_training_from_prediction or in_library:
        raise HTTPException(
            409,
            "Bildet brukes i treningsdata/eksempelbibliotek og kan ikke slettes trygt.",
        )

    if pred_ids:
        (
            db.query(models.DetectionHit)
            .filter(models.DetectionHit.prediction_id.in_(pred_ids))
            .update({models.DetectionHit.prediction_id: None}, synchronize_session=False)
        )
        (
            db.query(models.ReviewDecision)
            .filter(models.ReviewDecision.prediction_id.in_(pred_ids))
            .delete(synchronize_session=False)
        )
        (
            db.query(models.Prediction)
            .filter(models.Prediction.id.in_(pred_ids))
            .delete(synchronize_session=False)
        )

    (
        db.query(models.DetectionHit)
        .filter(models.DetectionHit.image_id == img.id)
        .update({models.DetectionHit.image_id: None}, synchronize_session=False)
    )

    if img.address_id:
        addr = db.get(models.AddressRecord, img.address_id)
        if addr and addr.selected_image_id == img.id:
            addr.selected_image_id = None

    db.delete(img)
    db.commit()
    return {"ok": True, "deleted_image_id": image_id}
