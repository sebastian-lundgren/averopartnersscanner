from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/annotations-overview")
def annotations_overview(skip: int = 0, limit: int = 500, db: Session = Depends(get_db)):
    """
    Merkede eksempler for annoterings-/læringsside: modell vs manuell bbox, label, split, notat.
    """
    rows = (
        db.query(models.TrainingExample)
        .order_by(models.TrainingExample.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    out: list[dict] = []
    err_counts: dict[str, int] = {}
    for te in rows:
        img = db.get(models.ImageAsset, te.image_id)
        pred = db.get(models.Prediction, te.source_prediction_id) if te.source_prediction_id else None
        yde = (
            db.query(models.YoloDatasetEntry)
            .filter(models.YoloDatasetEntry.training_example_id == te.id)
            .first()
        )
        tags = te.tags_json if isinstance(te.tags_json, dict) else {}
        et = te.error_type or ""
        if et:
            err_counts[et] = err_counts.get(et, 0) + 1
        out.append(
            {
                "id": te.id,
                "image_id": te.image_id,
                "filename": img.original_filename if img else "",
                "original_model_status": te.original_model_guess,
                "model_predicted_status": pred.predicted_status if pred else None,
                "model_bbox": pred.bbox_json if pred else None,
                "manual_bbox": tags.get("bbox_norm"),
                "manual_bboxes": tags.get("bboxes_norm") or (
                    [tags["bbox_norm"]] if tags.get("bbox_norm") else None
                ),
                "training_label": tags.get("annotation_label"),
                "final_status": te.human_status,
                "error_type": te.error_type,
                "comment": te.comment,
                "dataset_split": yde.split if yde else None,
                "created_at": te.created_at.isoformat(),
                "annotated_by": te.annotated_by,
            }
        )
    return {
        "rows": out,
        "error_type_summary": sorted(err_counts.items(), key=lambda x: -x[1])[:20],
        "total": len(out),
    }


@router.get("/examples")
def list_examples(skip: int = 0, limit: int = 200, db: Session = Depends(get_db)):
    q = (
        db.query(models.TrainingExample)
        .order_by(models.TrainingExample.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "image_id": r.image_id,
            "human_status": r.human_status,
            "original_model_guess": r.original_model_guess,
            "model_version_tag": r.model_version_tag,
            "confidence_at_time": r.confidence_at_time,
            "error_type": r.error_type,
            "comment": r.comment,
            "created_at": r.created_at.isoformat(),
            "annotated_by": r.annotated_by,
        }
        for r in q
    ]


@router.post("/library/{image_id}")
def upsert_library(image_id: int, body: schemas.TrainingLibraryUpsert, db: Session = Depends(get_db)):
    img = db.get(models.ImageAsset, image_id)
    if not img:
        raise HTTPException(404)
    row = db.query(models.TrainingLibraryEntry).filter_by(image_id=image_id).first()
    if row:
        row.category = body.category
        row.tags_json = body.tags
        row.notes = body.notes
    else:
        row = models.TrainingLibraryEntry(
            image_id=image_id,
            category=body.category,
            tags_json=body.tags,
            notes=body.notes,
        )
        db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/library")
def list_library(db: Session = Depends(get_db)):
    rows = db.query(models.TrainingLibraryEntry).all()
    return [
        {
            "id": r.id,
            "image_id": r.image_id,
            "category": r.category,
            "tags": r.tags_json,
            "notes": r.notes,
            "updated_at": r.updated_at.isoformat(),
        }
        for r in rows
    ]
