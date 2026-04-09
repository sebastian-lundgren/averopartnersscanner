from pathlib import Path
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.database import get_db
from app.services.blob_storage import is_r2_ref, stream_r2_object


def _iter_r2_body(body):
    while True:
        chunk = body.read(65536)
        if not chunk:
            break
        yield chunk
from app.services.path_resolve import resolve_evidence_path, resolve_stored_path

router = APIRouter(prefix="/api/files", tags=["files"])
log = logging.getLogger(__name__)


def _safe_image(path: str | None, *, image_id: int, kind: str) -> Path:
    if not path:
        log.warning(
            "FILES_RESOLVE_MISSING image_id=%s kind=%s stored_path=%r resolved_path=%r fallback_path=%r",
            image_id,
            kind,
            path,
            None,
            None,
        )
        raise HTTPException(404)
    try:
        p = resolve_stored_path(path)
    except Exception:
        log.exception(
            "FILES_RESOLVE_ERROR image_id=%s kind=%s stored_path=%r resolved_path=%r fallback_path=%r",
            image_id,
            kind,
            path,
            None,
            None,
        )
        raise HTTPException(404)
    if p.is_file():
        return p
    # Legacy/deploy-safe fallback: use filename under current UPLOAD_DIR.
    fallback = (Path(settings.upload_dir) / Path(path).name).resolve()
    if fallback.is_file():
        return fallback
    log.warning(
        "FILES_RESOLVE_NOT_FOUND image_id=%s kind=%s stored_path=%r resolved_path=%r fallback_path=%r",
        image_id,
        kind,
        path,
        str(p),
        str(fallback),
    )
    raise HTTPException(404)


def _safe_evidence(path: str | None) -> Path:
    p = resolve_evidence_path(path)
    if not p or not p.is_file():
        raise HTTPException(404)
    return p


@router.get("/image/{image_id}/original")
def serve_original(image_id: int, db: Session = Depends(get_db)):
    img = db.get(models.ImageAsset, image_id)
    if not img:
        raise HTTPException(404)
    if is_r2_ref(img.stored_path):
        try:
            body = stream_r2_object(img.stored_path)
        except Exception:
            log.exception(
                "FILES_R2_ORIGINAL_ERROR image_id=%s kind=%s stored_path=%r resolved_path=%r fallback_path=%r",
                image_id,
                "original",
                img.stored_path,
                None,
                None,
            )
            raise HTTPException(404)
        return StreamingResponse(
            _iter_r2_body(body),
            media_type=img.mime_type,
            headers={
                "Content-Disposition": f'inline; filename="{img.original_filename}"',
            },
        )
    p = _safe_image(img.stored_path, image_id=image_id, kind="original")
    return FileResponse(
        p,
        filename=img.original_filename,
        media_type=img.mime_type,
        content_disposition_type="inline",
    )


@router.get("/image/{image_id}/evidence")
def serve_evidence(image_id: int, db: Session = Depends(get_db)):
    img = db.get(models.ImageAsset, image_id)
    if not img or not img.evidence_crop_path:
        raise HTTPException(404)
    if is_r2_ref(img.evidence_crop_path):
        try:
            body = stream_r2_object(img.evidence_crop_path)
        except Exception:
            log.exception(
                "FILES_R2_EVIDENCE_ERROR image_id=%s kind=%s stored_path=%r resolved_path=%r fallback_path=%r",
                image_id,
                "evidence",
                img.evidence_crop_path,
                None,
                None,
            )
            raise HTTPException(404)
        return StreamingResponse(_iter_r2_body(body), media_type="image/jpeg")
    p = _safe_evidence(img.evidence_crop_path)
    return FileResponse(p, media_type="image/jpeg")
