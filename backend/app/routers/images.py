import threading
import uuid
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import SessionLocal, get_db
from app.services.active_learning import refresh_prediction_priority
from app.services.blob_storage import materialize_local_path, store_upload_bytes
from app.services.evidence import save_evidence_crop
from app.services.prediction import run_heuristic_predict
from app.services.settings_store import get_thresholds

router = APIRouter(prefix="/api/images", tags=["images"])


def _run_predictions_after_upload(image_ids: list[int]) -> None:
    """Grounding DINO / heuristikk — kjøres etter opplasting for å unngå timeout på POST /upload."""
    db = SessionLocal()
    try:
        model = db.query(models.ModelVersion).filter(models.ModelVersion.is_active.is_(True)).first()
        if not model:
            return
        thr = get_thresholds(db)
        strong = int(thr["threshold_strong_sign"])
        for iid in image_ids:
            try:
                img = db.get(models.ImageAsset, iid)
                if not img:
                    continue
                if db.query(models.Prediction).filter(models.Prediction.image_id == iid).first():
                    continue
                local_img, tmp_del = materialize_local_path(img.stored_path, suffix=".upload")
                try:
                    pred_result = run_heuristic_predict(str(local_img))
                finally:
                    if tmp_del:
                        local_img.unlink(missing_ok=True)

                if pred_result.bbox_norm:
                    ev_name = f"ev_{img.id}_{uuid.uuid4().hex[:8]}.jpg"
                    evidence_rel = save_evidence_crop(img.stored_path, ev_name, pred_result.bbox_norm)
                    if evidence_rel:
                        img.evidence_crop_path = evidence_rel

                needs_review = (
                    pred_result.confidence < strong
                    or pred_result.status != models.ReviewStatus.SKILT_FUNNET
                )
                pred = models.Prediction(
                    image_id=img.id,
                    model_version_id=model.id,
                    predicted_status=pred_result.status.value,
                    confidence=pred_result.confidence,
                    bbox_json=pred_result.bbox_norm,
                    rationale=pred_result.rationale,
                    needs_review=needs_review,
                    review_completed=False,
                )
                db.add(pred)
                db.flush()
                refresh_prediction_priority(db, pred)
                db.commit()
            except Exception:
                db.rollback()
    finally:
        db.close()


def _start_prediction_worker(image_ids: list[int]) -> None:
    ids = list(image_ids)
    threading.Thread(target=_run_predictions_after_upload, args=(ids,), daemon=True).start()


def _active_model(db: Session) -> models.ModelVersion:
    m = db.query(models.ModelVersion).filter(models.ModelVersion.is_active.is_(True)).first()
    if not m:
        raise HTTPException(500, "Ingen aktiv modellversjon")
    return m


@router.post("/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    address_id: int | None = Form(None),
    customer_id: str | None = Form(None),
    address_line: str | None = Form(None),
    is_temporary_candidate: bool = Form(False),
    db: Session = Depends(get_db),
):
    """
    Enkelt- eller batch-opplasting. Knyt til adresse/kunde ved behov.
    """
    if not files:
        raise HTTPException(400, "Ingen filer")

    address = None
    if address_id:
        address = db.get(models.AddressRecord, address_id)
        if not address:
            raise HTTPException(404, "Adresse ikke funnet")
    elif customer_id or address_line:
        address = models.AddressRecord(customer_id=customer_id, address_line=address_line)
        db.add(address)
        db.flush()

    _active_model(db)
    created: list[dict] = []
    pending_prediction_ids: list[int] = []

    for uf in files:
        if not uf.content_type or not uf.content_type.startswith("image/"):
            continue
        raw = await uf.read()
        ct = uf.content_type or "image/jpeg"
        stored_path, orig_name = store_upload_bytes(raw, uf.filename or "image.jpg", ct)
        try:
            im = Image.open(BytesIO(raw))
            w, h = im.size
        except Exception:
            w, h = None, None

        img = models.ImageAsset(
            address_id=address.id if address else None,
            original_filename=orig_name,
            stored_path=stored_path,
            mime_type=ct,
            width=w,
            height=h,
            is_temporary_candidate=is_temporary_candidate,
        )
        db.add(img)
        db.flush()
        pending_prediction_ids.append(img.id)
        created.append(
            {
                "image_id": img.id,
                "prediction_id": None,
                "prediction_pending": True,
            }
        )

    db.commit()
    if pending_prediction_ids:
        _start_prediction_worker(pending_prediction_ids)
    return {"ok": True, "items": created, "address_id": address.id if address else None}


@router.get("/library", response_model=list[schemas.ImageAssetOut])
def library(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    q = db.query(models.ImageAsset).order_by(models.ImageAsset.uploaded_at.desc())
    return q.offset(skip).limit(limit).all()


@router.get("/{image_id}", response_model=schemas.ImageAssetOut)
def get_image(image_id: int, db: Session = Depends(get_db)):
    img = db.get(models.ImageAsset, image_id)
    if not img:
        raise HTTPException(404)
    return img


@router.get("/{image_id}/predictions", response_model=list[schemas.PredictionOut])
def image_predictions(image_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.Prediction)
        .filter(models.Prediction.image_id == image_id)
        .order_by(models.Prediction.created_at.desc())
        .all()
    )
