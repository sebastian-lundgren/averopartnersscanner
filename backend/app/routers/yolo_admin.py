"""Eksport av YOLO-datasett, tildeling av split, og start av trening (subprocess)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import models, schemas
from app.config import settings
from app.database import get_db
from app.services.train_pipeline import create_train_job, has_active_train_job, start_train_job_thread
from app.services.yolo_export_files import write_yolo_dataset

router = APIRouter(prefix="/api/yolo", tags=["yolo"])


@router.get("/dataset/summary")
def dataset_summary(db: Session = Depends(get_db)):
    n = db.query(models.YoloDatasetEntry).count()
    by_split: dict[str, int] = {}
    for row in db.query(models.YoloDatasetEntry).all():
        by_split[row.split] = by_split.get(row.split, 0) + 1
    return {"total_entries": n, "by_split": by_split}


@router.post("/dataset/assign")
def dataset_assign(body: schemas.YoloDatasetAssign, db: Session = Depends(get_db)):
    te = db.get(models.TrainingExample, body.training_example_id)
    if not te:
        raise HTTPException(404, "TrainingExample ikke funnet")
    row = db.query(models.YoloDatasetEntry).filter_by(training_example_id=te.id).first()
    if row:
        row.split = body.split
    else:
        row = models.YoloDatasetEntry(training_example_id=te.id, split=body.split)
        db.add(row)
    db.commit()
    return {"ok": True, "id": row.id, "split": row.split}


@router.post("/dataset/export-disk")
def export_disk(clear: bool = False, db: Session = Depends(get_db)):
    root = Path(settings.yolo_dataset_export_dir)
    counts = write_yolo_dataset(db, root, clear_first=clear)
    return {"export_dir": str(root.resolve()), "counts": counts}


@router.post("/train")
def start_train(body: schemas.YoloTrainRequest, db: Session = Depends(get_db)):
    """Samme som POST /api/train-jobs/start med valgfri overstyring (bakoverkompatibel)."""
    if has_active_train_job(db):
        raise HTTPException(409, "En treningsjobb er allerede i kø eller kjører")
    override = {
        "base_model": body.base_model,
        "epochs": body.epochs,
        "imgsz": body.imgsz,
        "batch": body.batch,
    }
    if body.device is not None:
        override["device"] = body.device
    job = create_train_job(db, trigger="manual", config_override=override)
    db.commit()
    db.refresh(job)
    start_train_job_thread(job.id)
    return {"ok": True, "job_id": job.id, "message": "Treningsjobb startet (se /api/train-jobs)."}
