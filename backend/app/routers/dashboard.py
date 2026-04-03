from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=schemas.DashboardOut)
def dashboard_stats(db: Session = Depends(get_db)):
    total_images = int(db.query(func.count(models.ImageAsset.id)).scalar() or 0)
    total_predictions = int(db.query(func.count(models.Prediction.id)).scalar() or 0)

    def count_status(status: str) -> int:
        return (
            int(
                db.query(func.count(models.ReviewDecision.id))
                .filter(models.ReviewDecision.final_status == status)
                .scalar()
                or 0
            )
        )

    sk = count_status(models.ReviewStatus.SKILT_FUNNET.value)
    uk = count_status(models.ReviewStatus.UKLART.value)
    tr = count_status(models.ReviewStatus.TRENGER_MANUELL.value)
    overrides = int(
        db.query(func.count(models.ReviewDecision.id))
        .filter(models.ReviewDecision.was_override.is_(True))
        .scalar()
        or 0
    )
    pending = int(
        db.query(func.count(models.Prediction.id))
        .filter(models.Prediction.review_completed.is_(False))
        .scalar()
        or 0
    )

    since = datetime.utcnow() - timedelta(days=7)
    reviewed_recent = (
        db.query(models.ReviewDecision)
        .filter(models.ReviewDecision.decided_at >= since)
        .all()
    )
    err_recent = sum(1 for r in reviewed_recent if r.was_override)
    error_rate = (err_recent / len(reviewed_recent)) if reviewed_recent else None

    by_model_version = []
    for v in db.query(models.ModelVersion).all():
        preds = db.query(models.Prediction).filter(models.Prediction.model_version_id == v.id).all()
        ov = sum(1 for p in preds if p.review and p.review.was_override)
        by_model_version.append(
            {
                "version": v.version_tag,
                "predictions": len(preds),
                "overrides": ov,
            }
        )

    return schemas.DashboardOut(
        total_images=total_images,
        total_predictions=total_predictions,
        count_skilt_funnet=sk,
        count_uklart=uk,
        count_trenger_manuell=tr,
        overrides_count=overrides,
        pending_review=pending,
        error_rate_last_7d=error_rate,
        by_model_version=by_model_version,
    )
