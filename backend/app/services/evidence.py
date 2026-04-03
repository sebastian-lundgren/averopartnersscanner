"""Lagre evidensutsnitt fra bounding box (lokal disk eller R2)."""

from pathlib import Path

import cv2

from app.config import settings
from app.services.blob_storage import materialize_local_path, put_bytes, r2_enabled


def save_evidence_crop(
    source_stored_path: str,
    evidence_filename: str,
    bbox: dict | None,
    padding: float = 0.05,
) -> str | None:
    if not bbox:
        return None
    src, del_src = materialize_local_path(source_stored_path, suffix=".src")
    try:
        img = cv2.imread(str(src))
        if img is None:
            return None
        h, w = img.shape[:2]
        x = max(0, int((bbox["x"] - padding) * w))
        y = max(0, int((bbox["y"] - padding) * h))
        x2 = min(w, int((bbox["x"] + bbox["w"] + padding) * w))
        y2 = min(h, int((bbox["y"] + bbox["h"] + padding) * h))
        if x2 <= x or y2 <= y:
            return None
        crop = img[y:y2, x:x2]
        ok, enc = cv2.imencode(".jpg", crop)
        if not ok:
            return None
        data = enc.tobytes()
        if r2_enabled():
            key = f"evidence/{evidence_filename}"
            return put_bytes(key, data, "image/jpeg")
        out_dir = Path(settings.evidence_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / evidence_filename
        out_path.write_bytes(data)
        return str(out_path.resolve())
    finally:
        if del_src:
            src.unlink(missing_ok=True)
