from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app import models, schemas
from app.config import settings
from app.database import get_db
from app.services.bbox_multi import (
    is_valid_box,
    normalize_box,
    parse_bboxes_from_pred_json,
)

router = APIRouter(prefix="/api/reviews", tags=["reviews"])


def _training_bbox_list(body: schemas.ReviewSubmit, pred: models.Prediction) -> list[dict[str, float]]:
    if body.annotation_bboxes_json is not None:
        return [
            normalize_box(b)
            for b in body.annotation_bboxes_json
            if isinstance(b, dict) and is_valid_box(b)
        ]
    if body.annotation_bbox_json is not None and is_valid_box(body.annotation_bbox_json):
        return [normalize_box(body.annotation_bbox_json)]
    return parse_bboxes_from_pred_json(pred.bbox_json)

_ANNOTATION_TO_STATUS: dict[str, str] = {
    "alarm_sign": models.ReviewStatus.SKILT_FUNNET.value,
    "not_alarm_sign": models.ReviewStatus.TRENGER_MANUELL.value,
    "unclear": models.ReviewStatus.UKLART.value,
}


def _annotation_label_from_final(final: str) -> str:
    return {
        models.ReviewStatus.SKILT_FUNNET.value: "alarm_sign",
        models.ReviewStatus.TRENGER_MANUELL.value: "not_alarm_sign",
        models.ReviewStatus.UKLART.value: "unclear",
    }.get(final, "unclear")


def _claim_cutoff():
    return datetime.utcnow() - timedelta(seconds=settings.review_claim_ttl_seconds)


def _visible_claim_clause(annotator_id: str | None):
    cutoff = _claim_cutoff()
    clause = or_(
        models.Prediction.claimed_by.is_(None),
        models.Prediction.claimed_at < cutoff,
    )
    if annotator_id:
        clause = or_(clause, models.Prediction.claimed_by == annotator_id)
    return clause


@router.get("/queue-stats")
def queue_stats(db: Session = Depends(get_db)):
    cutoff = _claim_cutoff()
    pending = (
        db.query(models.Prediction)
        .filter(models.Prediction.review_completed.is_(False))
        .count()
    )
    unclaimed_or_expired = (
        db.query(models.Prediction)
        .filter(
            models.Prediction.review_completed.is_(False),
            or_(models.Prediction.claimed_by.is_(None), models.Prediction.claimed_at < cutoff),
        )
        .count()
    )
    claimed_active = (
        db.query(models.Prediction)
        .filter(
            models.Prediction.review_completed.is_(False),
            models.Prediction.claimed_by.isnot(None),
            models.Prediction.claimed_at >= cutoff,
        )
        .count()
    )
    completed = db.query(models.Prediction).filter(models.Prediction.review_completed.is_(True)).count()
    return {
        "pending_review": pending,
        "free_or_expired_claim": unclaimed_or_expired,
        "claimed_active": claimed_active,
        "completed_total": completed,
    }


@router.get("/queue", response_model=list[schemas.QueueItemOut])
def review_queue(
    limit: int = 50,
    annotator_id: str | None = None,
    image_ids: str | None = None,
    db: Session = Depends(get_db),
):
    id_filter: list[int] | None = None
    if image_ids and image_ids.strip():
        id_filter = [int(x.strip()) for x in image_ids.split(",") if x.strip().isdigit()]
        if not id_filter:
            id_filter = None
    q = (
        db.query(models.Prediction)
        .options(
            joinedload(models.Prediction.image),
            joinedload(models.Prediction.model_version),
            joinedload(models.Prediction.review),
        )
        .filter(
            models.Prediction.review_completed.is_(False),
            _visible_claim_clause(annotator_id),
        )
    )
    if id_filter is not None:
        q = q.filter(models.Prediction.image_id.in_(id_filter))
    preds = q.order_by(models.Prediction.priority_score.desc(), models.Prediction.created_at.asc()).limit(limit).all()
    out: list[schemas.QueueItemOut] = []
    for p in preds:
        out.append(
            schemas.QueueItemOut(
                prediction=schemas.PredictionOut.model_validate(p),
                image=schemas.ImageAssetOut.model_validate(p.image),
                review=schemas.ReviewDecisionOut.model_validate(p.review) if p.review else None,
            )
        )
    return out


@router.post("/claim-next", response_model=schemas.QueueItemOut)
def claim_next(body: schemas.AnnotatorBody, db: Session = Depends(get_db)):
    cutoff = _claim_cutoff()
    pred = (
        db.query(models.Prediction)
        .options(
            joinedload(models.Prediction.image),
            joinedload(models.Prediction.model_version),
            joinedload(models.Prediction.review),
        )
        .filter(
            models.Prediction.review_completed.is_(False),
            or_(
                models.Prediction.claimed_by.is_(None),
                models.Prediction.claimed_at < cutoff,
            ),
        )
        .order_by(models.Prediction.priority_score.desc(), models.Prediction.created_at.asc())
        .first()
    )
    if not pred:
        raise HTTPException(404, "Ingen ledige elementer i køen")
    pred.claimed_by = body.annotator_id
    pred.claimed_at = datetime.utcnow()
    db.commit()
    db.refresh(pred)
    return schemas.QueueItemOut(
        prediction=schemas.PredictionOut.model_validate(pred),
        image=schemas.ImageAssetOut.model_validate(pred.image),
        review=schemas.ReviewDecisionOut.model_validate(pred.review) if pred.review else None,
    )


@router.post("/{prediction_id}/release")
def release_claim(prediction_id: int, body: schemas.AnnotatorBody, db: Session = Depends(get_db)):
    pred = db.get(models.Prediction, prediction_id)
    if not pred or pred.review_completed:
        raise HTTPException(404)
    if pred.claimed_by != body.annotator_id:
        raise HTTPException(403, "Du har ikke claim på dette elementet")
    pred.claimed_by = None
    pred.claimed_at = None
    db.commit()
    return {"ok": True}


@router.post("/{prediction_id}/submit", response_model=schemas.ReviewDecisionOut)
def submit_review(
    prediction_id: int,
    body: schemas.ReviewSubmit,
    db: Session = Depends(get_db),
    x_annotator_id: str | None = Header(None, alias="X-Annotator-Id"),
):
    pred = db.get(models.Prediction, prediction_id)
    if not pred:
        raise HTTPException(404, "Prediksjon ikke funnet")
    if pred.review_completed:
        raise HTTPException(400, "Allerede ferdigstilt")

    if body.approve_without_change:
        final = pred.predicted_status
        ann_label = _annotation_label_from_final(final)
    elif body.annotation_label:
        final = _ANNOTATION_TO_STATUS[body.annotation_label]
        ann_label = body.annotation_label
    else:
        final = body.final_status
        ann_label = _annotation_label_from_final(final)

    was_override = final != pred.predicted_status

    if pred.review:
        rev = pred.review
        rev.final_status = final
        rev.was_override = was_override
        rev.comment = body.comment
        rev.error_type = body.error_type if was_override else None
    else:
        rev = models.ReviewDecision(
            prediction_id=pred.id,
            final_status=final,
            was_override=was_override,
            comment=body.comment,
            error_type=body.error_type if was_override else None,
        )
        db.add(rev)

    pred.review_completed = True
    pred.needs_review = False
    pred.claimed_by = None
    pred.claimed_at = None

    img = pred.image
    mv = pred.model_version

    b_list = _training_bbox_list(body, pred)
    bbox_training_legacy = b_list[0] if b_list else None
    annotator = (body.annotator_id or x_annotator_id or "").strip() or None
    te = models.TrainingExample(
        image_id=img.id,
        source_prediction_id=pred.id,
        human_status=final,
        original_model_guess=pred.predicted_status,
        model_version_tag=mv.version_tag if mv else None,
        confidence_at_time=pred.confidence,
        error_type=body.error_type if was_override else None,
        comment=body.comment,
        evidence_crop_path=img.evidence_crop_path,
        annotated_by=annotator,
        tags_json={
            "annotation_label": ann_label,
            "bboxes_norm": b_list,
            "bbox_norm": bbox_training_legacy,
            "format": "yolo_xywh_norm_topleft_multi",
        },
    )
    db.add(te)
    db.flush()

    if body.yolo_dataset_split:
        split = body.yolo_dataset_split
        ex = db.query(models.YoloDatasetEntry).filter_by(training_example_id=te.id).first()
        if ex:
            ex.split = split
        else:
            db.add(models.YoloDatasetEntry(training_example_id=te.id, split=split))

    if img.address_id:
        addr = db.get(models.AddressRecord, img.address_id)
        if addr:
            addr.final_human_status = final

    db.commit()
    db.refresh(rev)
    if settings.ml_inference_enabled:
        from app.services.train_pipeline import maybe_auto_enqueue_after_annotation

        maybe_auto_enqueue_after_annotation()
    return rev


@router.get("/{prediction_id}")
def get_review_context(prediction_id: int, db: Session = Depends(get_db)):
    pred = (
        db.query(models.Prediction)
        .options(
            joinedload(models.Prediction.image),
            joinedload(models.Prediction.model_version),
            joinedload(models.Prediction.review),
        )
        .filter(models.Prediction.id == prediction_id)
        .first()
    )
    if not pred:
        raise HTTPException(404)
    return {
        "prediction": schemas.PredictionOut.model_validate(pred),
        "image": schemas.ImageAssetOut.model_validate(pred.image),
        "review": schemas.ReviewDecisionOut.model_validate(pred.review) if pred.review else None,
    }
