"""
Street View scan-runner API: lagre forsøk, treff, og push bilder inn i review-kø (YOLO, ikke DINO).
"""

from __future__ import annotations

import json
import logging
import uuid
from io import BytesIO
from typing import Any
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from PIL import Image, ImageDraw
from sqlalchemy.orm import Session

from app import models, schemas
from app.config import settings
from app.database import get_db
from app.services.active_learning import refresh_prediction_priority
from app.services.blob_storage import materialize_local_path, store_upload_bytes
from app.services.bbox_multi import (
    canonicalize_bboxes,
    first_bbox,
    parse_bboxes_from_pred_json,
    yolo_trusted_primary_from_bbox_json,
)
from app.services.evidence import save_evidence_crop

router = APIRouter(prefix="/api/scanner", tags=["scanner"])
log = logging.getLogger(__name__)


def _annotate_scan_image_with_bboxes(
    source_path: str, bboxes: list[dict[str, float]], filename_hint: str
) -> str | None:
    if not bboxes:
        return None
    local_img, tmp_del = materialize_local_path(source_path, suffix=".scan-annotate")
    try:
        with Image.open(local_img).convert("RGB") as im:
            draw = ImageDraw.Draw(im)
            w, h = im.size
            for b in bboxes:
                x1 = max(0, min(w - 1, int(float(b.get("x", 0.0)) * w)))
                y1 = max(0, min(h - 1, int(float(b.get("y", 0.0)) * h)))
                x2 = max(0, min(w, int((float(b.get("x", 0.0)) + float(b.get("w", 0.0))) * w)))
                y2 = max(0, min(h, int((float(b.get("y", 0.0)) + float(b.get("h", 0.0))) * h)))
                if x2 <= x1 or y2 <= y1:
                    continue
                draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=4)

            buf = BytesIO()
            im.save(buf, format="JPEG", quality=92)
            return store_upload_bytes(
                buf.getvalue(),
                f"scan_annotated_{uuid.uuid4().hex[:12]}_{Path(filename_hint).name}",
                "image/jpeg",
            )[0]
    finally:
        if tmp_del:
            local_img.unlink(missing_ok=True)


def _check_scanner_token(x_scanner_token: str | None = Header(None)) -> None:
    expected = (settings.scanner_api_token or "").strip()
    if not expected:
        return
    if (x_scanner_token or "").strip() != expected:
        raise HTTPException(403, "Ugyldig eller manglende X-Scanner-Token")


def _yolo_model_version(db: Session) -> models.ModelVersion:
    tag = "yolov8s-scan"
    m = db.query(models.ModelVersion).filter_by(version_tag=tag).first()
    if not m:
        raise HTTPException(
            500,
            f"Mangler modellversjon {tag}. Kjør backend på nytt slik at seed oppretter den.",
        )
    return m


@router.post("/locations/bulk")
def bulk_locations(
    body: schemas.ScannerLocationBulk,
    db: Session = Depends(get_db),
    _: None = Depends(_check_scanner_token),
):
    created = []
    for loc in body.locations:
        row = models.TestLocation(
            address=loc.address,
            postcode=loc.postcode,
            latitude=loc.latitude,
            longitude=loc.longitude,
            status="pending",
        )
        db.add(row)
        db.flush()
        created.append({"id": row.id, "address": row.address})
    db.commit()
    return {"ok": True, "ids": created}


def _postcode_norm(s: str) -> str:
    return (s or "").strip().replace(" ", "")


@router.post("/runs/start", response_model=schemas.ScanRunStartOut)
def start_run(
    body: schemas.ScanRunStart,
    db: Session = Depends(get_db),
    _: None = Depends(_check_scanner_token),
):
    pc_body = _postcode_norm(body.postcode)
    if body.location_ids is not None:
        if not body.location_ids:
            raise HTTPException(400, "location_ids kan ikke være tom")
        if len(body.location_ids) > 500:
            raise HTTPException(400, "For mange location_ids")
        if len(set(body.location_ids)) != len(body.location_ids):
            raise HTTPException(400, "location_ids må være unike")
        if body.max_locations != len(body.location_ids):
            raise HTTPException(
                400,
                "max_locations må være lik antall location_ids når location_ids er satt",
            )
        rows = (
            db.query(models.TestLocation)
            .filter(models.TestLocation.id.in_(body.location_ids))
            .all()
        )
        by_id = {r.id: r for r in rows}
        locs = []
        for lid in body.location_ids:
            row = by_id.get(lid)
            if row is None:
                raise HTTPException(400, f"test_location id={lid} finnes ikke")
            if _postcode_norm(row.postcode) != pc_body:
                raise HTTPException(
                    400,
                    f"Postnummer på test_location id={lid} matcher ikke start_run.postcode",
                )
            locs.append(row)
    else:
        # Fallback uten eksplisitte id-er: nyeste radene for postkoden (bulk legger alltid til nye).
        q = (
            db.query(models.TestLocation)
            .filter(models.TestLocation.postcode == body.postcode)
            .order_by(models.TestLocation.id.desc())
            .limit(body.max_locations)
        )
        locs = list(reversed(q.all()))
    if not locs:
        raise HTTPException(400, "Ingen test_locations for denne postkoden")

    run = models.ScanRun(
        test_postcode=body.postcode,
        total_locations=len(locs),
        status="running",
    )
    db.add(run)
    db.flush()

    items = []
    for loc in locs:
        it = models.ScanRunItem(scan_run_id=run.id, location_id=loc.id)
        db.add(it)
        db.flush()
        items.append(
            {
                "scan_run_item_id": it.id,
                "location_id": loc.id,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "address": loc.address,
            }
        )
    db.commit()
    db.refresh(run)
    return schemas.ScanRunStartOut(scan_run_id=run.id, items=items)


@router.post("/runs/{run_id}/items/{item_id}/attempt")
def log_attempt(
    run_id: int,
    item_id: int,
    body: schemas.ScanAttemptIn,
    db: Session = Depends(get_db),
    _: None = Depends(_check_scanner_token),
):
    item = db.get(models.ScanRunItem, item_id)
    if not item or item.scan_run_id != run_id:
        raise HTTPException(404, "ScanRunItem ikke funnet")
    att = models.ScanAttempt(
        scan_run_item_id=item.id,
        attempt_index=body.attempt_index,
        screenshot_path=body.screenshot_path,
        camera_state=body.camera_state,
        prediction_status=body.prediction_status,
        confidence=body.confidence,
        bbox_json=body.bbox_json,
        rationale=body.rationale,
    )
    db.add(att)
    item.attempts_used = max(item.attempts_used, body.attempt_index + 1)
    db.commit()
    return {"ok": True, "attempt_id": att.id}


@router.post("/runs/{run_id}/items/{item_id}/complete")
def complete_item(
    run_id: int,
    item_id: int,
    body: schemas.ScanItemComplete,
    db: Session = Depends(get_db),
    _: None = Depends(_check_scanner_token),
):
    item = db.get(models.ScanRunItem, item_id)
    if not item or item.scan_run_id != run_id:
        raise HTTPException(404, "ScanRunItem ikke funnet")
    item.finished_at = datetime.utcnow()
    item.final_result = body.final_result
    item.best_confidence = body.best_confidence
    item.notes = body.notes

    run = db.get(models.ScanRun, run_id)
    if run:
        run.completed_locations = (run.completed_locations or 0) + 1
        if body.final_result == "detection_found":
            run.detections_found = (run.detections_found or 0) + 1
        if body.final_result == "failed":
            run.failed_locations = (run.failed_locations or 0) + 1
        if run.completed_locations >= run.total_locations:
            run.status = "completed"
            run.finished_at = datetime.utcnow()
    db.commit()
    return {"ok": True}


@router.post("/ingest-yolo")
async def ingest_yolo(
    file: UploadFile = File(...),
    scan_run_item_id: int | None = Form(None),
    location_id: int | None = Form(None),
    address_line: str | None = Form(None),
    postcode: str | None = Form(None),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    confidence: int = Form(...),
    bbox_json: str = Form(...),
    rationale: str | None = Form(None),
    predicted_status: str = Form("uklart"),
    db: Session = Depends(get_db),
    _: None = Depends(_check_scanner_token),
):
    """Lagre screenshot som ImageAsset + Prediction (YOLO-modellversjon) → review-kø."""
    try:
        raw = json.loads(bbox_json)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"bbox_json er ikke gyldig JSON: {e}") from e
    parsed = parse_bboxes_from_pred_json(raw)
    yolo_meta: dict[str, Any] = {}
    if isinstance(raw, dict):
        if "yolo_trusted_primary" in raw:
            yolo_meta["yolo_trusted_primary"] = bool(raw["yolo_trusted_primary"])
        gr = raw.get("yolo_primary_gate_reason")
        if isinstance(gr, str):
            yolo_meta["yolo_primary_gate_reason"] = gr[:500]
    box = canonicalize_bboxes(parsed, yolo_meta=yolo_meta if yolo_meta else None) if parsed else None
    bboxes_for_draw = parse_bboxes_from_pred_json(box) if box else []
    bbox_count = len(bboxes_for_draw)
    log.info(
        "DIAG ingest_yolo recv: item_id=%s location_id=%s predicted_status=%s confidence=%s bbox_count=%s bbox_json=%s",
        scan_run_item_id,
        location_id,
        predicted_status,
        confidence,
        bbox_count,
        bbox_json,
    )

    content = await file.read()
    safe = Path(file.filename or "scan.jpg").name
    stored_path, _ = store_upload_bytes(
        content, f"scan_{uuid.uuid4().hex[:12]}_{safe}", file.content_type or "image/jpeg"
    )

    local_img, tmp_del = materialize_local_path(stored_path, suffix=".scan")
    try:
        try:
            im = Image.open(local_img)
            w, h = im.size
        except Exception:
            w, h = None, None
    finally:
        if tmp_del:
            local_img.unlink(missing_ok=True)

    addr = None
    if address_line or postcode:
        note_parts: list[str] = []
        if latitude is not None:
            note_parts.append(f"scan lat={latitude} lon={longitude}")
        addr = models.AddressRecord(
            address_line=address_line,
            customer_id=postcode,
            notes=" | ".join(note_parts) if note_parts else None,
        )
        db.add(addr)
        db.flush()

    mv = _yolo_model_version(db)
    annotated_path = None
    img = models.ImageAsset(
        address_id=addr.id if addr else None,
        original_filename=safe,
        stored_path=stored_path,
        mime_type=file.content_type or "image/jpeg",
        width=w,
        height=h,
        is_temporary_candidate=True,
    )
    if box:
        annotated_path = _annotate_scan_image_with_bboxes(stored_path, bboxes_for_draw, safe)
        if annotated_path:
            img.stored_path = annotated_path
    db.add(img)
    db.flush()

    ev_rel = None
    fb = first_bbox(box)
    if fb and yolo_trusted_primary_from_bbox_json(box):
        ev_name = f"ev_scan_{img.id}_{uuid.uuid4().hex[:6]}.jpg"
        ev_rel = save_evidence_crop(stored_path, ev_name, fb)
        if ev_rel:
            img.evidence_crop_path = ev_rel

    pred = models.Prediction(
        image_id=img.id,
        model_version_id=mv.id,
        predicted_status=predicted_status,
        confidence=max(0, min(100, int(confidence))),
        bbox_json=box,
        rationale=rationale or "YOLOv8s (scan-runner)",
        needs_review=True,
        review_completed=False,
    )
    db.add(pred)
    db.flush()
    refresh_prediction_priority(db, pred)

    hit = models.DetectionHit(
        scan_run_item_id=scan_run_item_id,
        location_id=location_id,
        image_id=img.id,
        prediction_id=pred.id,
        screenshot_path=stored_path,
        confidence=float(confidence) / 100.0 if confidence <= 100 else float(confidence),
        bbox_json=box,
        rationale=rationale,
        review_status="pending",
    )
    db.add(hit)
    db.commit()

    return {
        "ok": True,
        "image_id": img.id,
        "prediction_id": pred.id,
        "detection_hit_id": hit.id,
        "original_stored_path": stored_path,
        "annotated_stored_path": annotated_path,
        "used_stored_path": img.stored_path,
        "bbox_count": bbox_count,
    }
