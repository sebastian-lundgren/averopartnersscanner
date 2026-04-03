from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/api/model-versions", tags=["model-versions"])


class ModelVersionCreate(BaseModel):
    version_tag: str = Field(..., min_length=1, max_length=64)
    description: str | None = None
    set_active: bool = False
    weights_path: str | None = None


@router.post("", response_model=schemas.ModelVersionOut)
def create_version(body: ModelVersionCreate, db: Session = Depends(get_db)):
    if db.query(models.ModelVersion).filter_by(version_tag=body.version_tag).first():
        raise HTTPException(409, "version_tag finnes allerede")
    if body.set_active:
        for m in db.query(models.ModelVersion).all():
            m.is_active = False
    desc = (body.description or "").strip() or None
    row = models.ModelVersion(
        version_tag=body.version_tag,
        description=desc,
        is_active=body.set_active,
        deployed_at=datetime.utcnow(),
        weights_path=body.weights_path,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("", response_model=list[schemas.ModelVersionOut])
def list_versions(db: Session = Depends(get_db)):
    return db.query(models.ModelVersion).order_by(models.ModelVersion.deployed_at.desc()).all()


@router.get("/compare-summary")
def compare_summary_get(db: Session = Depends(get_db)):
    return _compare_summary_body(db)


@router.post("/compare-summary")
def compare_summary_post(db: Session = Depends(get_db)):
    return _compare_summary_body(db)


def _compare_summary_body(db: Session):
    versions = db.query(models.ModelVersion).all()
    out = []
    for v in versions:
        preds = db.query(models.Prediction).filter(models.Prediction.model_version_id == v.id).all()
        overrides = 0
        by_err: dict[str, int] = {}
        for p in preds:
            if p.review and p.review.was_override:
                overrides += 1
                et = p.review.error_type or "ukjent"
                by_err[et] = by_err.get(et, 0) + 1
        out.append(
            {
                "version_tag": v.version_tag,
                "total_predictions": len(preds),
                "overrides": overrides,
                "override_rate": (overrides / len(preds)) if preds else 0,
                "error_types": by_err,
            }
        )
    return {"versions": out}
