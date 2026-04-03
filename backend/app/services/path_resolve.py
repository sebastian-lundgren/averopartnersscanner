"""Løs bildestier for delt lagring: absolutt sti i DB eller relativt under UPLOAD_DIR."""

from __future__ import annotations

from pathlib import Path

from app.config import settings


def resolve_stored_path(stored_path: str) -> Path:
    if stored_path.startswith("r2:"):
        raise ValueError("r2:-referanser må løses via blob_storage.materialize_local_path")
    p = Path(stored_path)
    if p.is_absolute():
        return p.resolve()
    return (Path(settings.upload_dir) / p).resolve()


def resolve_evidence_path(stored_path: str | None) -> Path | None:
    if not stored_path:
        return None
    if stored_path.startswith("r2:"):
        raise ValueError("r2:-referanser må strømmes via blob_storage / files-router")
    p = Path(stored_path)
    if p.is_absolute():
        return p.resolve()
    return (Path(settings.evidence_dir) / p).resolve()
