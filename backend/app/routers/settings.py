from fastapi import APIRouter, Depends

from app import schemas
from app.database import get_db
from app.services.settings_store import get_thresholds, set_thresholds, ensure_defaults
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/thresholds")
def get_thr(db: Session = Depends(get_db)):
    ensure_defaults(db)
    return get_thresholds(db)


@router.put("/thresholds")
def put_thr(body: schemas.ThresholdsUpdate, db: Session = Depends(get_db)):
    ensure_defaults(db)
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    return set_thresholds(db, data)
