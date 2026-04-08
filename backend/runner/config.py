"""Miljø for Street View scan-runner (Playwright + YOLOv8s)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repo-rot = forelder av runner/-mappen (uavhengig av cwd ved `python -m runner` fra prosjektrot).
_RUNNER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RUNNER_DIR.parent
_DEFAULT_YOLO = _REPO_ROOT / "data" / "models" / "yolov8s.pt"


def _resolve_yolo_model_path(raw: str | None) -> Path:
    """Unngå feil sti når cwd er repo (GSV-jobb) mens .env har ../backend/... ment for cwd=runner/."""
    s = (raw or "").strip()
    if not s:
        return _DEFAULT_YOLO.resolve()
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    # Legacy: ../backend/... i runner/.env → tolk relativt til runner/
    if p.parts and p.parts[0] == "..":
        return (_RUNNER_DIR / p).resolve()
    return (_REPO_ROOT / p).resolve()


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name, str(default)).lower().strip()
    return v in ("1", "true", "yes", "on")


_default_port = os.environ.get("PORT", "8000").strip() or "8000"
API_BASE = os.environ.get("SCANNER_API_BASE", f"http://127.0.0.1:{_default_port}").rstrip("/")
SCANNER_TOKEN = os.environ.get("SCANNER_API_TOKEN", "").strip()
YOLO_MODEL_PATH = _resolve_yolo_model_path(os.environ.get("YOLO_MODEL_PATH"))
CONF_STRONG = float(os.environ.get("YOLO_CONF_STRONG", "0.65"))
# Må matche (eller være lavere enn) YOLO inferens-gulv — ellers «finnes» bokser for runner men aldri save_hit.
YOLO_CONF_FLOOR = float(os.environ.get("YOLO_CONF_FLOOR", "0.25"))
_conf_weak_raw = os.environ.get("YOLO_CONF_WEAK")
CONF_WEAK = float(_conf_weak_raw) if _conf_weak_raw is not None else YOLO_CONF_FLOOR
YOLO_PRIMARY_TRUST_MIN_CONF = float(os.environ.get("YOLO_PRIMARY_TRUST_MIN_CONF", "0.45"))
YOLO_PRIMARY_TRUST_MIN_COMPOSITE = float(os.environ.get("YOLO_PRIMARY_TRUST_MIN_COMPOSITE", "0.30"))
MAX_ATTEMPTS = int(os.environ.get("SCAN_MAX_ATTEMPTS", "4"))
VIEW_WAIT_MS = int(os.environ.get("STREETVIEW_WAIT_MS", "4500"))
VIEW_WAIT_QUICK_MS = int(os.environ.get("STREETVIEW_WAIT_QUICK_MS", "2750"))
_vwa = os.environ.get("STREETVIEW_WAIT_AFTER_FIRST_ADDR_MS")
VIEW_WAIT_AFTER_FIRST_ADDR_MS = int(_vwa) if _vwa else int(VIEW_WAIT_MS * 0.86)
STREETVIEW_POST_READY_MS_FIRST = int(os.environ.get("STREETVIEW_POST_READY_MS_FIRST", "1600"))
STREETVIEW_POST_READY_MS_NEXT = int(os.environ.get("STREETVIEW_POST_READY_MS_NEXT", "1180"))
_spr = os.environ.get("STREETVIEW_POST_READY_MS_FIRST_SUBSEQUENT_ADDR")
STREETVIEW_POST_READY_MS_FIRST_SUBSEQUENT_ADDR = (
    int(_spr) if _spr else max(900, int(STREETVIEW_POST_READY_MS_FIRST * 0.78))
)
PAGE_TIMEOUT_MS = int(os.environ.get("PLAYWRIGHT_TIMEOUT_MS", "60000"))
HEADLESS = _b("PLAYWRIGHT_HEADLESS", False)
DEDUP_IOU = float(os.environ.get("DEDUP_IOU", "0.85"))
STABILIZE_MS = int(os.environ.get("CAPTURE_STABILIZE_MS", "800"))
FOCUS_PRE_CLICK_MS = int(os.environ.get("FOCUS_PRE_CLICK_MS", "120"))
CAPTURE_PRE_STABILIZE_MS = int(os.environ.get("CAPTURE_PRE_STABILIZE_MS", "700"))
CAPTURE_PRE_STABILIZE_MS_NEXT = int(os.environ.get("CAPTURE_PRE_STABILIZE_MS_NEXT", "640"))
# Kommaseparerte avstander (m) vinkelrett på hovedpano→hus; prøves i rekkefølge til ny panoid/pos.
_ft = os.environ.get("FRONT_NEIGHBOR_STEP_TRIES_M", "7,9.5,12")
FRONT_NEIGHBOR_STEP_TRIES_M: tuple[float, ...] = tuple(
    float(x.strip()) for x in _ft.split(",") if x.strip()
)
if not FRONT_NEIGHBOR_STEP_TRIES_M:
    FRONT_NEIGHBOR_STEP_TRIES_M = (7.0, 9.5, 12.0)
# Street View-capture: høyere oppløsning + DPR gir skarpere canvas (WebGL) for YOLO/review
CAPTURE_VIEWPORT_WIDTH = int(os.environ.get("CAPTURE_VIEWPORT_WIDTH", "1920"))
CAPTURE_VIEWPORT_HEIGHT = int(os.environ.get("CAPTURE_VIEWPORT_HEIGHT", "1080"))
CAPTURE_DEVICE_SCALE_FACTOR = float(os.environ.get("CAPTURE_DEVICE_SCALE_FACTOR", "2"))
CAPTURE_JPEG_QUALITY = int(os.environ.get("CAPTURE_JPEG_QUALITY", "94"))
