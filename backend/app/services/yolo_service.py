"""
YOLOv8s inferens for API og verktøy. Ultralytics er valgfritt; manglende modellfil håndteres eksplisitt.
Én klasse: alarm_sign (id 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.models import ReviewStatus


@dataclass
class YoloInferenceOutput:
    predicted_status: str
    confidence: int  # 0–100
    bbox_json: dict[str, float] | None  # x,y,w,h normalisert 0–1 (topleft)
    rationale: str
    needs_review: bool
    raw_detections: list[dict[str, Any]]


def _bbox_xywh_norm_from_xyxy(x1, y1, x2, y2, iw: int, ih: int) -> dict[str, float]:
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    return {
        "x": max(0.0, min(1.0, float(x1) / iw)),
        "y": max(0.0, min(1.0, float(y1) / ih)),
        "w": max(0.0, min(1.0, w / iw)),
        "h": max(0.0, min(1.0, h / ih)),
    }


def run_yolov8_on_image(
    image_path: str | Path,
    *,
    model_path: Path | None = None,
    conf_strong: float | None = None,
    conf_weak: float | None = None,
    db_session=None,
) -> YoloInferenceOutput:
    """
    Kjør YOLO på bilde. Krever ultralytics og gyldig .pt ved angitt sti.
    """
    path = Path(image_path)
    mp = model_path
    if mp is None and db_session is not None:
        from app.services import settings_store

        custom = settings_store.get_yolo_inference_weights_path(db_session)
        if custom:
            mp = Path(custom)
    if mp is None:
        mp = Path(settings.yolo_model_path)
    hi = conf_strong if conf_strong is not None else settings.yolo_confidence_strong
    lo = conf_weak if conf_weak is not None else settings.yolo_confidence_weak

    if not mp.is_file():
        return YoloInferenceOutput(
            predicted_status=ReviewStatus.TRENGER_MANUELL.value,
            confidence=0,
            bbox_json=None,
            rationale=f"YOLO-modellfil mangler: {mp.resolve()} — legg inn f.eks. yolov8s.pt eller trenet vekt.",
            needs_review=True,
            raw_detections=[],
        )

    try:
        from ultralytics import YOLO
    except ImportError as e:
        return YoloInferenceOutput(
            predicted_status=ReviewStatus.TRENGER_MANUELL.value,
            confidence=0,
            bbox_json=None,
            rationale=f"Ultralytics ikke installert ({e!s}). pip install ultralytics",
            needs_review=True,
            raw_detections=[],
        )

    try:
        model = YOLO(str(mp))
        results = model.predict(str(path), conf=min(lo, 0.25), verbose=False)
    except Exception as e:
        return YoloInferenceOutput(
            predicted_status=ReviewStatus.TRENGER_MANUELL.value,
            confidence=0,
            bbox_json=None,
            rationale=f"YOLO-inferens feilet: {e!s}",
            needs_review=True,
            raw_detections=[],
        )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return YoloInferenceOutput(
            predicted_status=ReviewStatus.UKLART.value,
            confidence=0,
            bbox_json=None,
            rationale="YOLOv8: ingen alarm_sign-kandidat over minimumsterskel.",
            needs_review=True,
            raw_detections=[],
        )

    r0 = results[0]
    ih, iw = r0.orig_shape[:2]
    boxes = r0.boxes
    best_i = int(boxes.conf.argmax().item()) if len(boxes) > 1 else 0
    conf_f = float(boxes.conf[best_i].item())
    xyxy = boxes.xyxy[best_i].tolist()
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    bbox = _bbox_xywh_norm_from_xyxy(x1, y1, x2, y2, iw, ih)
    conf_pct = int(round(max(0, min(100, conf_f * 100))))

    raw: list[dict[str, Any]] = []
    for i in range(len(boxes)):
        b = boxes.xyxy[i].tolist()
        raw.append(
            {
                "conf": float(boxes.conf[i].item()),
                "xyxy": b,
                "cls": int(boxes.cls[i].item()) if boxes.cls is not None else 0,
            }
        )

    if conf_f >= hi:
        return YoloInferenceOutput(
            predicted_status=ReviewStatus.UKLART.value,
            confidence=conf_pct,
            bbox_json=bbox,
            rationale=f"YOLOv8s: alarm_sign kandidat conf={conf_f:.2f} (sterk) — krever manuell bekreftelse i review.",
            needs_review=True,
            raw_detections=raw,
        )
    if conf_f >= lo:
        return YoloInferenceOutput(
            predicted_status=ReviewStatus.UKLART.value,
            confidence=conf_pct,
            bbox_json=bbox,
            rationale=f"YOLOv8s: svak alarm_sign kandidat conf={conf_f:.2f} — review anbefalt.",
            needs_review=True,
            raw_detections=raw,
        )

    return YoloInferenceOutput(
        predicted_status=ReviewStatus.UKLART.value,
        confidence=conf_pct,
        bbox_json=bbox,
        rationale=f"YOLOv8s: svært lav conf={conf_f:.2f} — terskel ikke nådd for sikker lagring.",
        needs_review=True,
        raw_detections=raw,
    )


def bbox_to_yolo_line(class_id: int, bbox: dict[str, float]) -> str:
    """YOLO txt: class xc yc w h (0–1). bbox er x,y,w,h topleft norm."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    xc = x + w / 2.0
    yc = y + h / 2.0
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
