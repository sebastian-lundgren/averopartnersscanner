"""YOLOv8s — samme kontrakt som backend (ingen DINO)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class DetectorResult:
    has_detection: bool
    confidence: float  # 0–1
    confidence_pct: int
    bbox_xyxy_pixels: tuple[float, float, float, float] | None
    bbox_norm_xywh: dict[str, float] | None
    rationale: str
    raw: list[dict[str, Any]]


def _xyxy_to_norm_xywh(x1, y1, x2, y2, iw: int, ih: int) -> dict[str, float]:
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    return {
        "x": max(0.0, min(1.0, float(x1) / iw)),
        "y": max(0.0, min(1.0, float(y1) / ih)),
        "w": max(0.0, min(1.0, w / iw)),
        "h": max(0.0, min(1.0, h / ih)),
    }


def run_yolo(image_path: Path, model_path: Path, conf_floor: float = 0.25) -> DetectorResult:
    if not model_path.is_file():
        log.error("YOLO-modellfil mangler: %s", model_path)
        return DetectorResult(
            False,
            0.0,
            0,
            None,
            None,
            f"Mangler modellfil: {model_path}",
            [],
        )
    try:
        from ultralytics import YOLO
    except ImportError as e:
        return DetectorResult(False, 0.0, 0, None, None, f"Ultralytics: {e!s}", [])

    model = YOLO(str(model_path))
    results = model.predict(str(image_path), conf=conf_floor, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return DetectorResult(
            False,
            0.0,
            0,
            None,
            None,
            "Ingen deteksjoner",
            [],
        )

    r0 = results[0]
    ih, iw = r0.orig_shape[:2]
    boxes = r0.boxes
    bi = int(boxes.conf.argmax().item()) if len(boxes) > 1 else 0
    cf = float(boxes.conf[bi].item())
    xyxy = boxes.xyxy[bi].tolist()
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    norm = _xyxy_to_norm_xywh(x1, y1, x2, y2, iw, ih)
    raw = []
    for i in range(len(boxes)):
        raw.append(
            {
                "conf": float(boxes.conf[i].item()),
                "xyxy": boxes.xyxy[i].tolist(),
            }
        )
    return DetectorResult(
        True,
        cf,
        int(round(cf * 100)),
        (x1, y1, x2, y2),
        norm,
        f"YOLOv8s best conf={cf:.3f}",
        raw,
    )
