"""Demo-data og initial modellversjon."""

from pathlib import Path

from PIL import Image, ImageDraw

from app import models
from app.config import settings
from app.database import SessionLocal, init_db
from app.services.active_learning import refresh_prediction_priority
from app.services.evidence import save_evidence_crop
from app.services.settings_store import ensure_defaults, get_thresholds


def _ensure_model_versions(db):
    """Sørg for Grounding DINO-rad (aktiv ved innsetting); ellers bruk aktiv modell i DB."""
    tag = "grounding-dino-base-hf"
    desc = "Grounding DINO via Hugging Face (IDEA-Research/grounding-dino-base)"

    grounding = db.query(models.ModelVersion).filter_by(version_tag=tag).first()
    if grounding is None:
        for m in db.query(models.ModelVersion).all():
            m.is_active = False
        grounding = models.ModelVersion(version_tag=tag, description=desc, is_active=True)
        db.add(grounding)
        if db.query(models.ModelVersion).filter_by(version_tag="heuristic-v0-baseline").first() is None:
            db.add(
                models.ModelVersion(
                    version_tag="heuristic-v0-baseline",
                    description="Historisk heuristikk-baseline (inaktiv)",
                    is_active=False,
                )
            )
        db.commit()
        db.refresh(grounding)

    if db.query(models.ModelVersion).filter_by(version_tag="yolov8s-scan").first() is None:
        db.add(
            models.ModelVersion(
                version_tag="yolov8s-scan",
                description="YOLOv8s — Street View scan-runner / manuell annotering (inaktiv vs DINO)",
                is_active=False,
            )
        )
        db.commit()

    m = db.query(models.ModelVersion).filter_by(is_active=True).first()
    return m or grounding


def _synthetic_images(upload_dir: Path):
    """To enkle testbilder (faktiske PNG-filer) for lokal demo."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    # «Skarpt» med rektangel (kan trigge kant-heuristikk)
    im1 = Image.new("RGB", (640, 480), (180, 175, 160))
    d = ImageDraw.Draw(im1)
    d.rectangle([200, 40, 420, 120], outline=(20, 20, 20), width=3)
    p1 = upload_dir / "demo_synthetic_facade.png"
    im1.save(p1)
    paths.append(str(p1))
    # Uskarpt / lav kontrast
    im2 = Image.new("RGB", (640, 480), (140, 140, 140))
    p2 = upload_dir / "demo_synthetic_unclear.png"
    im2.save(p2)
    paths.append(str(p2))
    return paths


def seed_if_empty():
    init_db()
    db = SessionLocal()
    try:
        ensure_defaults(db)
        model = _ensure_model_versions(db)

        addr = db.query(models.AddressRecord).filter_by(customer_id="DEMO-KUNDE-001").first()
        if not addr:
            addr = models.AddressRecord(
                customer_id="DEMO-KUNDE-001",
                address_line="Demo vei 1, 0001 Oslo",
                notes="Eksempel kun for autorisert testmiljø",
            )
            db.add(addr)
            db.commit()
            db.refresh(addr)

        if db.query(models.ImageAsset).first():
            return

        from app.services.prediction import run_heuristic_predict

        upload_dir = Path(settings.upload_dir)
        evidence_dir = Path(settings.evidence_dir)
        thr = get_thresholds(db)
        strong = int(thr["threshold_strong_sign"])

        for i, path in enumerate(_synthetic_images(upload_dir)):
            p = Path(path)
            im = Image.open(p)
            w, h = im.size
            img = models.ImageAsset(
                address_id=addr.id,
                original_filename=p.name,
                stored_path=str(p.resolve()),
                mime_type="image/png",
                width=w,
                height=h,
                is_temporary_candidate=False,
                is_primary_for_address=(i == 0),
            )
            db.add(img)
            db.flush()

            pr = run_heuristic_predict(img.stored_path)
            import uuid

            ev_path = None
            if pr.bbox_norm:
                ev_name = f"ev_seed_{img.id}_{uuid.uuid4().hex[:6]}.jpg"
                ev_path = save_evidence_crop(img.stored_path, ev_name, pr.bbox_norm)
                if ev_path:
                    img.evidence_crop_path = ev_path

            pred = models.Prediction(
                image_id=img.id,
                model_version_id=model.id,
                predicted_status=pr.status.value,
                confidence=pr.confidence,
                bbox_json=pr.bbox_norm,
                rationale=pr.rationale,
                needs_review=pr.confidence < strong
                or pr.status != models.ReviewStatus.SKILT_FUNNET,
                review_completed=False,
            )
            db.add(pred)
            db.flush()
            refresh_prediction_priority(db, pred)

        db.commit()
    finally:
        db.close()
