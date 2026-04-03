from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db
from app.services.best_view import run_best_view_for_address

router = APIRouter(prefix="/api/addresses", tags=["addresses"])


class AddressCreate(BaseModel):
    customer_id: str | None = None
    address_line: str | None = None
    notes: str | None = None


@router.post("", response_model=schemas.AddressOut)
def create_address(body: AddressCreate, db: Session = Depends(get_db)):
    a = models.AddressRecord(
        customer_id=body.customer_id,
        address_line=body.address_line,
        notes=body.notes,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return _enrich_address(db, a)


def _enrich_address(db: Session, a: models.AddressRecord) -> schemas.AddressOut:
    cnt = db.query(func.count(models.ImageAsset.id)).filter(models.ImageAsset.address_id == a.id).scalar()
    sub = (
        db.query(func.max(models.Prediction.confidence))
        .join(models.ImageAsset, models.ImageAsset.id == models.Prediction.image_id)
        .filter(models.ImageAsset.address_id == a.id)
        .scalar()
    )
    last_at = (
        db.query(func.max(models.Prediction.created_at))
        .join(models.ImageAsset, models.ImageAsset.id == models.Prediction.image_id)
        .filter(models.ImageAsset.address_id == a.id)
        .scalar()
    )
    return schemas.AddressOut(
        id=a.id,
        customer_id=a.customer_id,
        address_line=a.address_line,
        notes=a.notes,
        attempt_count=a.attempt_count,
        best_quality_score=a.best_quality_score,
        selected_image_id=a.selected_image_id,
        final_human_status=a.final_human_status,
        selection_metadata_json=a.selection_metadata_json,
        image_count=int(cnt or 0),
        highest_confidence=int(sub) if sub is not None else None,
        last_prediction_at=last_at,
    )


@router.get("", response_model=list[schemas.AddressOut])
def list_addresses(db: Session = Depends(get_db)):
    rows = db.query(models.AddressRecord).order_by(models.AddressRecord.updated_at.desc()).all()
    return [_enrich_address(db, a) for a in rows]


@router.get("/{address_id}", response_model=schemas.AddressOut)
def get_address(address_id: int, db: Session = Depends(get_db)):
    a = db.get(models.AddressRecord, address_id)
    if not a:
        raise HTTPException(404)
    return _enrich_address(db, a)


@router.get("/{address_id}/images", response_model=list[schemas.ImageAssetOut])
def address_images(address_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.ImageAsset)
        .filter(models.ImageAsset.address_id == address_id)
        .order_by(models.ImageAsset.uploaded_at.desc())
        .all()
    )


@router.patch("/{address_id}/final-status")
def set_final_human_status(address_id: int, body: dict, db: Session = Depends(get_db)):
    a = db.get(models.AddressRecord, address_id)
    if not a:
        raise HTTPException(404)
    st = body.get("final_human_status")
    if st not in (
        models.ReviewStatus.SKILT_FUNNET.value,
        models.ReviewStatus.UKLART.value,
        models.ReviewStatus.TRENGER_MANUELL.value,
    ):
        raise HTTPException(400, "Ugyldig status")
    a.final_human_status = st
    a.updated_at = datetime.utcnow()
    db.commit()
    return {"ok": True}


@router.post("/{address_id}/best-view")
def best_view(address_id: int, db: Session = Depends(get_db)):
    return run_best_view_for_address(db, address_id)
