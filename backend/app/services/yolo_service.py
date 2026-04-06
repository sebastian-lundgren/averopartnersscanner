"""
YOLOv8s inferens for API og verktøy. Ultralytics er valgfritt; manglende modellfil håndteres eksplisitt.
Én klasse: alarm_sign (id 0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.models import ReviewStatus
from app.services.bbox_multi import canonicalize_bboxes
from app.services.yolo_bbox_rank import reorder_scored_for_storage

log = logging.getLogger(__name__)


@dataclass
class YoloInferenceOutput:
    predicted_status: str
    confidence: int  # 0–100 (beste kandidat)
    # Kanonisk multi: {"boxes": [{x,y,w,h}, ...], "v": 2} eller None
    bbox_json: dict[str, Any] | None
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
        log.info("YOLO inferens %s: 0 kandidater (under predict-conf)", path.name)
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
    raw: list[dict[str, Any]] = []
    scored: list[tuple[float, dict[str, float]]] = []
    for i in range(len(boxes)):
        cf = float(boxes.conf[i].item())
        b = boxes.xyxy[i].tolist()
        raw.append(
            {
                "conf": cf,
                "xyxy": b,
                "cls": int(boxes.cls[i].item()) if boxes.cls is not None else 0,
            }
        )
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        scored.append((cf, _bbox_xywh_norm_from_xyxy(x1, y1, x2, y2, iw, ih)))
    bboxes_norm, conf_f, trust = reorder_scored_for_storage(
        scored,
        context="api",
        image_label=path.name,
        trust_min_conf=settings.yolo_primary_trust_min_conf,
        trust_min_composite=settings.yolo_primary_trust_min_composite,
    )
    yolo_meta = {
        "yolo_trusted_primary": trust["trusted_primary"],
        "yolo_primary_gate_reason": str(trust["primary_gate_reason"])[:500],
    }
    canonical = canonicalize_bboxes(bboxes_norm, yolo_meta=yolo_meta)
    conf_pct = int(round(max(0, min(100, conf_f * 100))))
    n = len(bboxes_norm)
    log.info(
        "YOLO inferens %s: %s kandidat(er) etter rangering (primær rå conf=%.3f trusted_primary=%s), lagrer multi-bbox=%s",
        path.name,
        n,
        conf_f,
        trust["trusted_primary"],
        canonical is not None,
    )

    trust_note = ""
    if not trust["trusted_primary"]:
        trust_note = (
            f" Usikre modellforslag — ingen pålitelig auto-primær ({trust['primary_gate_reason']}). "
            "Alle bokser er kun forslag; velg manuelt i review."
        )

    if conf_f >= hi:
        return YoloInferenceOutput(
            predicted_status=ReviewStatus.UKLART.value,
            confidence=conf_pct,
            bbox_json=canonical,
            rationale=(
                f"YOLOv8s: {n} alarm_sign-kandidat(er), beste conf={conf_f:.2f} (sterk) — manuell QA i review."
                + trust_note
            ),
            needs_review=True,
            raw_detections=raw,
        )
    if conf_f >= lo:
        return YoloInferenceOutput(
            predicted_status=ReviewStatus.UKLART.value,
            confidence=conf_pct,
            bbox_json=canonical,
            rationale=f"YOLOv8s: {n} kandidat(er), beste conf={conf_f:.2f} (svak) — review anbefalt." + trust_note,
            needs_review=True,
            raw_detections=raw,
        )

    return YoloInferenceOutput(
        predicted_status=ReviewStatus.UKLART.value,
        confidence=conf_pct,
        bbox_json=canonical,
        rationale=f"YOLOv8s: {n} kandidat(er), beste conf={conf_f:.2f} — under svak terskel, fortsatt bbox-forslag."
        + trust_note,
        needs_review=True,
        raw_detections=raw,
    )


def bbox_to_yolo_line(class_id: int, bbox: dict[str, float]) -> str:
    """YOLO txt: class xc yc w h (0–1). bbox er x,y,w,h topleft norm."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    xc = x + w / 2.0
    yc = y + h / 2.0
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
