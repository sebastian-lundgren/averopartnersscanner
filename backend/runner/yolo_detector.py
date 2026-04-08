"""YOLOv8s — samme kontrakt som backend (ingen DINO)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner import config
from runner.timing import step_timer
from runner.yolo_bbox_rank import reorder_scored_for_storage

log = logging.getLogger(__name__)

# Én lastet modell per sti — unngår å relaste vekter for hvert capture (stor tidsbesparelse).
_yolo_models: dict[str, object] = {}


@dataclass
class DetectorResult:
    has_detection: bool
    confidence: float  # 0–1
    confidence_pct: int
    bbox_xyxy_pixels: tuple[float, float, float, float] | None
    bbox_norm_xywh: dict[str, float] | None
    rationale: str
    raw: list[dict[str, Any]]
    all_bboxes_norm: list[dict[str, float]]
    yolo_trusted_primary: bool = True
    yolo_primary_gate_reason: str = ""


def _xyxy_to_norm_xywh(x1, y1, x2, y2, iw: int, ih: int) -> dict[str, float]:
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    return {
        "x": max(0.0, min(1.0, float(x1) / iw)),
        "y": max(0.0, min(1.0, float(y1) / ih)),
        "w": max(0.0, min(1.0, w / iw)),
        "h": max(0.0, min(1.0, h / ih)),
    }


def run_yolo(image_path: Path, model_path: Path, conf_floor: float | None = None) -> DetectorResult:
    floor = config.YOLO_CONF_FLOOR if conf_floor is None else conf_floor

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
            [],
        )
    try:
        from ultralytics import YOLO
    except ImportError as e:
        return DetectorResult(False, 0.0, 0, None, None, f"Ultralytics: {e!s}", [], [])

    key = str(model_path.resolve())
    model = _yolo_models.get(key)
    if model is None:
        model = YOLO(key)
        _yolo_models[key] = model

    with step_timer(log, "yolo_predict", image=image_path.name):
        results = model.predict(str(image_path), conf=floor, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        log.info("YOLO runner: 0 deteksjoner (under conf=%s)", floor)
        return DetectorResult(
            False,
            0.0,
            0,
            None,
            None,
            "Ingen deteksjoner",
            [],
            [],
        )

    r0 = results[0]
    ih, iw = r0.orig_shape[:2]
    boxes = r0.boxes
    raw = []
    scored: list[tuple[float, dict[str, float]]] = []
    for i in range(len(boxes)):
        cf_i = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        norm_i = _xyxy_to_norm_xywh(x1, y1, x2, y2, iw, ih)
        raw.append({"conf": cf_i, "xyxy": xyxy})
        scored.append((cf_i, norm_i))
    all_norm, cf, trust = reorder_scored_for_storage(
        scored,
        context="runner",
        image_label=image_path.name,
        trust_min_conf=config.YOLO_PRIMARY_TRUST_MIN_CONF,
        trust_min_composite=config.YOLO_PRIMARY_TRUST_MIN_COMPOSITE,
    )
    b0 = all_norm[0]
    x1 = b0["x"] * iw
    y1 = b0["y"] * ih
    x2 = (b0["x"] + b0["w"]) * iw
    y2 = (b0["y"] + b0["h"]) * ih
    norm = dict(b0)
    log.info(
        "YOLO runner: %s deteksjon(er), primær rå conf=%.3f trusted_primary=%s",
        len(all_norm),
        cf,
        trust["trusted_primary"],
    )
    return DetectorResult(
        True,
        cf,
        int(round(cf * 100)),
        (x1, y1, x2, y2),
        norm,
        f"YOLOv8s n={len(all_norm)} best conf={cf:.3f} | {trust['primary_gate_reason']}",
        raw,
        all_norm,
        yolo_trusted_primary=bool(trust["trusted_primary"]),
        yolo_primary_gate_reason=str(trust["primary_gate_reason"])[:500],
    )
