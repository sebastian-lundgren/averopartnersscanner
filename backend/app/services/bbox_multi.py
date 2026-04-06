"""Normalisert bbox (0–1): én eller flere per prediksjon — felles parsing/lagring."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def is_valid_box(b: dict) -> bool:
    try:
        return (
            all(k in b for k in ("x", "y", "w", "h"))
            and float(b["w"]) > 1e-8
            and float(b["h"]) > 1e-8
        )
    except (TypeError, KeyError, ValueError):
        return False


def normalize_box(b: dict) -> dict[str, float]:
    return {
        "x": float(b["x"]),
        "y": float(b["y"]),
        "w": float(b["w"]),
        "h": float(b["h"]),
    }


def parse_bboxes_from_pred_json(raw: Any) -> list[dict[str, float]]:
    """Leser lagret prediction.bbox_json: legacy enkelt-dict, liste, eller {boxes, v}."""
    if raw is None:
        return []
    if isinstance(raw, list):
        out = [normalize_box(b) for b in raw if isinstance(b, dict) and is_valid_box(b)]
        return out
    if isinstance(raw, dict):
        inner = raw.get("boxes")
        if isinstance(inner, list):
            out = [normalize_box(b) for b in inner if isinstance(b, dict) and is_valid_box(b)]
            return out
        if is_valid_box(raw):
            return [normalize_box(raw)]
    return []


def canonicalize_bboxes(
    boxes: list[dict[str, float]],
    *,
    yolo_meta: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    valid = [normalize_box(b) for b in boxes if is_valid_box(b)]
    if not valid:
        return None
    out: dict[str, Any] = {"boxes": valid, "v": 2}
    if yolo_meta:
        if "yolo_trusted_primary" in yolo_meta:
            out["yolo_trusted_primary"] = bool(yolo_meta["yolo_trusted_primary"])
        r = yolo_meta.get("yolo_primary_gate_reason")
        if isinstance(r, str) and r.strip():
            out["yolo_primary_gate_reason"] = r.strip()[:500]
    return out


def yolo_trusted_primary_from_bbox_json(raw: Any) -> bool:
    """
    Om første bbox er ment som pålitelig auto-primær (evidens/review).
    Manglende nøkkel: True (eldre JSON uten flagg).
    """
    if not isinstance(raw, dict):
        return True
    if "yolo_trusted_primary" not in raw:
        return True
    return bool(raw["yolo_trusted_primary"])


def first_bbox(raw: Any) -> dict[str, float] | None:
    xs = parse_bboxes_from_pred_json(raw)
    return xs[0] if xs else None
