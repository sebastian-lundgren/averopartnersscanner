import json

from sqlalchemy.orm import Session

from app import models
from app.config import settings as app_settings


DEFAULT_THRESHOLDS = {
    "threshold_strong_sign": app_settings.threshold_strong_sign,
    "threshold_unclear_high": app_settings.threshold_unclear_high,
    "threshold_unclear_low": app_settings.threshold_unclear_low,
    "max_best_view_attempts": app_settings.max_best_view_attempts,
    "quality_threshold": app_settings.quality_threshold,
}


def get_thresholds(db: Session) -> dict:
    row = db.get(models.AppSetting, "thresholds")
    if row:
        try:
            merged = {**DEFAULT_THRESHOLDS, **json.loads(row.value_json)}
            return merged
        except json.JSONDecodeError:
            pass
    return dict(DEFAULT_THRESHOLDS)


def set_thresholds(db: Session, data: dict) -> dict:
    current = get_thresholds(db)
    current.update({k: v for k, v in data.items() if k in DEFAULT_THRESHOLDS})
    row = db.get(models.AppSetting, "thresholds")
    if row:
        row.value_json = json.dumps(current)
    else:
        db.add(models.AppSetting(key="thresholds", value_json=json.dumps(current)))
    db.commit()
    return current


KEY_YOLO_INFERENCE_WEIGHTS = "yolo_inference_weights_path"
KEY_YOLO_BASELINE_METRICS = "yolo_baseline_metrics_json"
KEY_TRAIN_CHECKPOINT_TE_ID = "train_checkpoint_training_example_id"


def ensure_defaults(db: Session) -> None:
    if db.get(models.AppSetting, "thresholds") is None:
        db.add(
            models.AppSetting(key="thresholds", value_json=json.dumps(DEFAULT_THRESHOLDS))
        )
        db.commit()


def _get_json_setting(db: Session, key: str):
    row = db.get(models.AppSetting, key)
    if not row:
        return None
    try:
        return json.loads(row.value_json)
    except json.JSONDecodeError:
        return None


def _set_json_setting(db: Session, key: str, value) -> None:
    row = db.get(models.AppSetting, key)
    payload = json.dumps(value)
    if row:
        row.value_json = payload
    else:
        db.add(models.AppSetting(key=key, value_json=payload))


def get_yolo_inference_weights_path(db: Session) -> str | None:
    v = _get_json_setting(db, KEY_YOLO_INFERENCE_WEIGHTS)
    return v if isinstance(v, str) and v else None


def set_yolo_inference_weights_path(db: Session, path: str) -> None:
    _set_json_setting(db, KEY_YOLO_INFERENCE_WEIGHTS, path)


def get_yolo_baseline_metrics(db: Session) -> dict | None:
    v = _get_json_setting(db, KEY_YOLO_BASELINE_METRICS)
    return v if isinstance(v, dict) else None


def set_yolo_baseline_metrics(db: Session, metrics: dict) -> None:
    _set_json_setting(db, KEY_YOLO_BASELINE_METRICS, metrics)


def get_train_checkpoint_te_id(db: Session) -> int:
    v = _get_json_setting(db, KEY_TRAIN_CHECKPOINT_TE_ID)
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return 0


def set_train_checkpoint_te_id(db: Session, te_id: int) -> None:
    _set_json_setting(db, KEY_TRAIN_CHECKPOINT_TE_ID, te_id)
