"""Miljø for Street View scan-runner (Playwright + YOLOv8s)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name, str(default)).lower().strip()
    return v in ("1", "true", "yes", "on")


API_BASE = os.environ.get("SCANNER_API_BASE", "http://127.0.0.1:8000").rstrip("/")
SCANNER_TOKEN = os.environ.get("SCANNER_API_TOKEN", "").strip()
YOLO_MODEL_PATH = Path(os.environ.get("YOLO_MODEL_PATH", "../backend/data/models/yolov8s.pt")).resolve()
CONF_STRONG = float(os.environ.get("YOLO_CONF_STRONG", "0.65"))
CONF_WEAK = float(os.environ.get("YOLO_CONF_WEAK", "0.35"))
MAX_ATTEMPTS = int(os.environ.get("SCAN_MAX_ATTEMPTS", "4"))
VIEW_WAIT_MS = int(os.environ.get("STREETVIEW_WAIT_MS", "4500"))
PAGE_TIMEOUT_MS = int(os.environ.get("PLAYWRIGHT_TIMEOUT_MS", "60000"))
HEADLESS = _b("PLAYWRIGHT_HEADLESS", False)
DEDUP_IOU = float(os.environ.get("DEDUP_IOU", "0.85"))
STABILIZE_MS = int(os.environ.get("CAPTURE_STABILIZE_MS", "800"))
