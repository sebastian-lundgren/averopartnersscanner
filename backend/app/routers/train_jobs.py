from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session

from app import models, schemas
from app.config import settings
from app.database import get_db
from app.services.train_pipeline import (
    count_new_annotations_since_checkpoint,
    create_train_job,
    has_active_train_job,
    start_train_job_thread,
)

router = APIRouter(prefix="/api/train-jobs", tags=["train-jobs"])


@router.get("/auto-trigger-status")
def auto_trigger_status(db: Session = Depends(get_db)):
    return {
        "new_annotations_since_checkpoint": count_new_annotations_since_checkpoint(db),
        "trigger_threshold": settings.yolo_train_trigger_min_new_annotations,
        "auto_enabled": settings.yolo_train_auto_enabled,
        "train_job_busy": has_active_train_job(db),
    }


@router.get("", response_model=list[schemas.TrainJobOut])
def list_jobs(limit: int = 50, db: Session = Depends(get_db)):
    return (
        db.query(models.TrainJob)
        .order_by(models.TrainJob.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )


@router.get("/{job_id}", response_model=schemas.TrainJobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(models.TrainJob, job_id)
    if not job:
        raise HTTPException(404, "Jobb ikke funnet")
    return job


@router.post("/start", response_model=schemas.TrainJobOut)
def start_job(
    raw: dict | None = Body(None),
    db: Session = Depends(get_db),
):
    if has_active_train_job(db):
        raise HTTPException(409, "En treningsjobb er allerede i kø eller kjører")
    body = schemas.TrainJobStartBody.model_validate(raw or {})
    override: dict = {}
    if body.base_model is not None:
        override["base_model"] = body.base_model
    if body.epochs is not None:
        override["epochs"] = body.epochs
    if body.imgsz is not None:
        override["imgsz"] = body.imgsz
    if body.batch is not None:
        override["batch"] = body.batch
    if body.device is not None:
        override["device"] = body.device
    job = create_train_job(db, trigger="manual", config_override=override or None)
    db.commit()
    db.refresh(job)
    start_train_job_thread(job.id)
    return job
