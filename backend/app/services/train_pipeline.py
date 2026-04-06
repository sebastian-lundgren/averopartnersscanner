"""YOLO treningsjobb: eksport → trening → validering → ev. aktivering."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.database import SessionLocal
from app.models import TrainJobStatus
from app.services import settings_store
from app.services.yolo_export_files import write_yolo_dataset

_cancel_lock = threading.Lock()
_cancel_events: dict[int, threading.Event] = {}

METRIC_KEY_ALIASES = {
    "map50_95": "mAP50-95",
    "map50": "mAP50",
    "precision": "precision",
    "recall": "recall",
}


def count_new_annotations_since_checkpoint(db: Session) -> int:
    cp = settings_store.get_train_checkpoint_te_id(db)
    return (
        db.query(func.count(models.TrainingExample.id))
        .filter(models.TrainingExample.id > cp)
        .scalar()
        or 0
    )


def has_active_train_job(db: Session) -> bool:
    return (
        db.query(models.TrainJob)
        .filter(
            models.TrainJob.status.in_([TrainJobStatus.QUEUED, TrainJobStatus.RUNNING]),
        )
        .first()
        is not None
    )


def default_train_config() -> dict:
    return {
        "base_model": settings.yolo_train_base_model,
        "epochs": settings.yolo_train_epochs,
        "imgsz": settings.yolo_train_imgsz,
        "batch": settings.yolo_train_batch,
        "device": settings.yolo_train_device or None,
    }


def create_train_job(db: Session, *, trigger: str, config_override: dict | None = None) -> models.TrainJob:
    cfg = {**default_train_config(), **(config_override or {})}
    job = models.TrainJob(
        status=TrainJobStatus.QUEUED,
        trigger=trigger,
        config_json=cfg,
        new_annotations_snapshot=count_new_annotations_since_checkpoint(db),
    )
    db.add(job)
    db.flush()
    return job


def extract_val_metrics(val_stats) -> dict[str, float]:
    out: dict[str, float] = {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0}
    if val_stats is None:
        return out
    box = getattr(val_stats, "box", None)
    if box is not None:
        out["precision"] = float(getattr(box, "mp", 0) or 0)
        out["recall"] = float(getattr(box, "mr", 0) or 0)
        out["mAP50"] = float(getattr(box, "map50", 0) or 0)
        out["mAP50-95"] = float(getattr(box, "map", 0) or 0)
        return out
    rd = getattr(val_stats, "results_dict", None) or {}
    if isinstance(rd, dict):
        out["mAP50"] = float(rd.get("metrics/mAP50(B)", rd.get("mAP50", 0)) or 0)
        out["mAP50-95"] = float(
            rd.get("metrics/mAP50-95(B)", rd.get("mAP50-95", rd.get("map", 0))) or 0
        )
    return out


def _metric_for_comparison(new_metrics: dict, baseline: dict | None) -> tuple[float, float]:
    alias = METRIC_KEY_ALIASES.get(settings.yolo_activation_metric, "mAP50-95")
    new_v = float(new_metrics.get(alias, 0) or 0)
    old_v = float(baseline.get(alias, 0) or 0) if baseline else 0.0
    return new_v, old_v


def should_activate_new_model(db: Session, new_metrics: dict) -> bool:
    baseline = settings_store.get_yolo_baseline_metrics(db)
    new_v, old_v = _metric_for_comparison(new_metrics, baseline)
    if baseline is None:
        return True
    return new_v >= old_v + float(settings.yolo_activation_min_delta)


def _train_cancel_requested(job_id: int) -> bool:
    with _cancel_lock:
        ev = _cancel_events.get(job_id)
        return bool(ev and ev.is_set())


def signal_cancel_train_job(job_id: int) -> bool:
    with _cancel_lock:
        ev = _cancel_events.get(job_id)
        if ev is None:
            return False
        ev.set()
        return True


def start_train_job_thread(job_id: int) -> None:
    with _cancel_lock:
        _cancel_events[job_id] = threading.Event()
    t = threading.Thread(target=run_train_job_sync, args=(job_id,), daemon=True)
    t.start()


def _finalize_cancelled(db: Session, job_id: int) -> None:
    job = db.get(models.TrainJob, job_id)
    if job and job.status == TrainJobStatus.RUNNING:
        job.status = TrainJobStatus.CANCELLED
        job.finished_at = datetime.utcnow()
        job.error_message = "Avbrutt av bruker"
        db.commit()


def run_train_job_sync(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(models.TrainJob, job_id)
        if not job:
            return
        job.status = TrainJobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.error_message = None
        db.commit()

        if _train_cancel_requested(job_id):
            _finalize_cancelled(db, job_id)
            return

        root = Path(settings.yolo_dataset_export_dir)
        root.mkdir(parents=True, exist_ok=True)
        counts = write_yolo_dataset(db, root, clear_first=True)
        job.export_counts_json = dict(counts)
        db.commit()

        if _train_cancel_requested(job_id):
            _finalize_cancelled(db, job_id)
            return

        train_n = int(counts.get("train", 0))
        val_n = int(counts.get("val", 0))
        if train_n < settings.yolo_train_min_train_images or val_n < settings.yolo_train_min_val_images:
            raise ValueError(
                f"For lite data: train={train_n} (min {settings.yolo_train_min_train_images}), "
                f"val={val_n} (min {settings.yolo_train_min_val_images})"
            )

        yaml = root / "dataset.yaml"
        if not yaml.is_file():
            raise ValueError("dataset.yaml ble ikke opprettet")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(f"ultralytics mangler: {e}") from e

        cfg = job.config_json or default_train_config()
        base = str(cfg.get("base_model") or settings.yolo_train_base_model)
        project = str(Path(settings.yolo_train_output_dir).resolve())
        Path(project).mkdir(parents=True, exist_ok=True)
        run_name = f"job_{job.id}_{uuid.uuid4().hex[:8]}"

        model = YOLO(base)

        def on_train_epoch_end(trainer):
            if _train_cancel_requested(job_id):
                trainer.stop = True

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        train_kw: dict = {
            "data": str(yaml.resolve()),
            "epochs": int(cfg.get("epochs", settings.yolo_train_epochs)),
            "imgsz": int(cfg.get("imgsz", settings.yolo_train_imgsz)),
            "batch": int(cfg.get("batch", settings.yolo_train_batch)),
            "project": project,
            "name": run_name,
            "exist_ok": True,
        }
        dev = cfg.get("device") or settings.yolo_train_device
        if dev:
            train_kw["device"] = dev

        model.train(**train_kw)

        if _train_cancel_requested(job_id):
            _finalize_cancelled(db, job_id)
            return

        best = Path(project) / run_name / "weights" / "best.pt"
        if not best.is_file():
            if _train_cancel_requested(job_id):
                _finalize_cancelled(db, job_id)
                return
            raise RuntimeError(f"Fant ikke best.pt under {best}")

        val_model = YOLO(str(best))
        val_stats = val_model.val(data=str(yaml.resolve()), verbose=False)
        metrics = extract_val_metrics(val_stats)

        train_img_total = train_n + val_n
        tag = f"yolo-{job.id}-{uuid.uuid4().hex[:6]}"
        mv = models.ModelVersion(
            version_tag=tag,
            description=f"YOLO train job #{job.id} ({job.trigger})",
            is_active=False,
            deployed_at=datetime.utcnow(),
            metrics_json=metrics,
            weights_path=str(best.resolve()),
            training_config_json=cfg,
            train_image_count=train_img_total,
        )
        db.add(mv)
        db.flush()

        activated = should_activate_new_model(db, metrics)
        if activated:
            settings_store.set_yolo_inference_weights_path(db, str(best.resolve()))
            settings_store.set_yolo_baseline_metrics(db, metrics)
            max_te = db.query(func.max(models.TrainingExample.id)).scalar()
            if max_te is not None:
                settings_store.set_train_checkpoint_te_id(db, int(max_te))

        job.status = TrainJobStatus.FINISHED
        job.finished_at = datetime.utcnow()
        job.metrics_json = metrics
        job.candidate_model_version_id = mv.id
        job.activated_new_model = activated
        db.commit()
    except Exception as e:
        try:
            job = db.get(models.TrainJob, job_id)
            if job and job.status not in (
                TrainJobStatus.CANCELLED,
                TrainJobStatus.FINISHED,
            ):
                job.status = TrainJobStatus.FAILED
                job.finished_at = datetime.utcnow()
                job.error_message = str(e)[:8000]
                db.commit()
        except Exception:
            db.rollback()
    finally:
        with _cancel_lock:
            _cancel_events.pop(job_id, None)
        db.close()


def maybe_auto_enqueue_after_annotation() -> None:
    if not settings.yolo_train_auto_enabled:
        return
    db = SessionLocal()
    try:
        n = count_new_annotations_since_checkpoint(db)
        if n < settings.yolo_train_trigger_min_new_annotations:
            return
        if has_active_train_job(db):
            return
        job = create_train_job(db, trigger="auto")
        db.commit()
        start_train_job_thread(job.id)
    finally:
        db.close()
