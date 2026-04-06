"""Web-MVP: start Google Street View-scan som bakgrunnsjobb (subprocess runner)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session

from app import models, schemas
from app.config import settings
from app.database import get_db
from app.models import StreetViewScanJobStatus
from app.services.gsv_scan_job_runner import create_gsv_scan_job, start_gsv_scan_job_thread

router = APIRouter(prefix="/api/streetview-scan-jobs", tags=["streetview-scan-jobs"])


def _parse_locations_plan(raw: str | None) -> schemas.ScanJobLocationsPlan | None:
    if not raw or not str(raw).strip():
        return None
    try:
        return schemas.ScanJobLocationsPlan.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        return None


def _build_result_summary(
    db: Session, job: models.StreetViewScanJob
) -> schemas.StreetViewScanJobResultSummary | None:
    if job.status != StreetViewScanJobStatus.DONE or not job.scan_run_id:
        return None
    run = db.get(models.ScanRun, job.scan_run_id)
    if not run:
        return None
    item_ids = [
        row[0] for row in db.query(models.ScanRunItem.id).filter_by(scan_run_id=run.id).all()
    ]
    image_ids: list[int] = []
    if item_ids:
        hits = db.query(models.DetectionHit).filter(models.DetectionHit.scan_run_item_id.in_(item_ids)).all()
        image_ids = sorted({h.image_id for h in hits if h.image_id is not None})
    pred_pending = 0
    if image_ids:
        pred_pending = (
            db.query(models.Prediction)
            .filter(
                models.Prediction.image_id.in_(image_ids),
                models.Prediction.review_completed.is_(False),
            )
            .count()
        )
    items = (
        db.query(models.ScanRunItem)
        .filter_by(scan_run_id=run.id)
        .order_by(models.ScanRunItem.id.asc())
        .all()
    )
    address_outcomes: list[schemas.ScanJobAddressOutcome] = []
    for ord_i, it in enumerate(items, start=1):
        tl = db.get(models.TestLocation, it.location_id)
        addr = tl.address if tl else ""
        img_saved = (
            db.query(models.DetectionHit)
            .filter(
                models.DetectionHit.scan_run_item_id == it.id,
                models.DetectionHit.image_id.isnot(None),
            )
            .count()
        )
        notes = it.notes
        if notes and len(notes) > 240:
            notes = notes[:237] + "..."
        address_outcomes.append(
            schemas.ScanJobAddressOutcome(
                order=ord_i,
                location_id=it.location_id,
                address=addr,
                final_result=it.final_result,
                notes=notes,
                images_saved=img_saved,
            )
        )
    return schemas.StreetViewScanJobResultSummary(
        scan_run_id=run.id,
        run_status=run.status,
        total_locations=run.total_locations,
        completed_locations=run.completed_locations or 0,
        locations_with_detection=run.detections_found or 0,
        images_saved=len(image_ids),
        image_ids=image_ids,
        predictions_pending_review=pred_pending,
        address_outcomes=address_outcomes,
    )


def scan_job_to_out(db: Session, job: models.StreetViewScanJob) -> schemas.StreetViewScanJobOut:
    base = schemas.StreetViewScanJobOut.model_validate(job)
    plan = _parse_locations_plan(getattr(job, "locations_plan_json", None))
    return base.model_copy(
        update={
            "result_summary": _build_result_summary(db, job),
            "locations_plan": plan,
        }
    )


@router.get("", response_model=list[schemas.StreetViewScanJobOut])
def list_jobs(limit: int = 30, db: Session = Depends(get_db)):
    q = db.query(models.StreetViewScanJob).order_by(models.StreetViewScanJob.created_at.desc())
    rows = q.limit(min(max(limit, 1), 100)).all()
    return [scan_job_to_out(db, j) for j in rows]


@router.get("/{job_id}", response_model=schemas.StreetViewScanJobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(models.StreetViewScanJob, job_id)
    if not job:
        raise HTTPException(404, "Jobb ikke funnet")
    return scan_job_to_out(db, job)


@router.post("/start", response_model=schemas.StreetViewScanJobOut)
def start_job(
    db: Session = Depends(get_db),
    body: schemas.StreetViewScanJobStartBody | None = Body(default=None),
):
    raw = body.model_dump(exclude_unset=True) if body else {}
    pc = (raw.get("postcode") or settings.gsv_scan_default_postcode or "").strip()
    if not pc:
        raise HTTPException(400, "Mangler postnummer (satt GSV_SCAN_DEFAULT_POSTCODE eller send postcode i body)")
    use_dyn = bool(raw.get("use_dynamic_locations", True))
    job = create_gsv_scan_job(
        db,
        postcode=pc,
        max_locations=raw.get("max_locations"),
        max_attempts=raw.get("max_attempts"),
        max_images_per_address=raw.get("max_images_per_address"),
        locations_json_path=raw.get("locations_json_path"),
        use_dynamic_locations=use_dyn,
    )
    db.commit()
    db.refresh(job)
    start_gsv_scan_job_thread(job.id)
    return scan_job_to_out(db, job)
