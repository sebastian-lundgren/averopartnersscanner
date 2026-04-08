"""Terskler, dedup (IoU), stopp når treff er godt nok."""

from __future__ import annotations

from dataclasses import dataclass

from runner import config
from runner.yolo_detector import DetectorResult


def iou_xywh(a: dict[str, float], b: dict[str, float]) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Decision:
    save_hit: bool
    stop_attempts: bool
    tier: str  # strong | weak | none
    reason: str


def evaluate(det: DetectorResult, previous_best: dict[str, float] | None) -> Decision:
    if not det.has_detection or det.bbox_norm_xywh is None:
        return Decision(False, False, "none", "Ingen deteksjon")

    if det.confidence >= config.CONF_STRONG:
        if previous_best and iou_xywh(det.bbox_norm_xywh, previous_best) >= config.DEDUP_IOU:
            return Decision(False, True, "strong", "Dedup: lik bbox som tidligere beste")
        return Decision(True, True, "strong", "Over høy terskel")

    if det.confidence >= config.CONF_WEAK:
        if previous_best and iou_xywh(det.bbox_norm_xywh, previous_best) >= config.DEDUP_IOU:
            return Decision(False, False, "weak", "Dedup svak kandidat")
        return Decision(True, False, "weak", "Svak kandidat — lagre til review")

    return Decision(False, False, "none", "Under lav terskel")
