from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app import models
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


def _safe_image(path: str | None) -> Path:
    if not path:
        raise HTTPException(404)
    p = resolve_stored_path(path)
    if not p.is_file():
        raise HTTPException(404)
    return p


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
        body = stream_r2_object(img.stored_path)
        return StreamingResponse(
            _iter_r2_body(body),
            media_type=img.mime_type,
            headers={
                "Content-Disposition": f'inline; filename="{img.original_filename}"',
            },
        )
    p = _safe_image(img.stored_path)
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
        body = stream_r2_object(img.evidence_crop_path)
        return StreamingResponse(_iter_r2_body(body), media_type="image/jpeg")
    p = _safe_evidence(img.evidence_crop_path)
    return FileResponse(p, media_type="image/jpeg")
