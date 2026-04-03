"""Best view selection per adresse — maks forsøk, unknown-first."""

from datetime import datetime

from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.services.blob_storage import materialize_local_path
from app.services.quality import assess_image_quality
from app.services.settings_store import get_thresholds

import cv2


def run_best_view_for_address(db: Session, address_id: int) -> dict:
    thr = get_thresholds(db)
    max_attempts = int(thr.get("max_best_view_attempts", settings.max_best_view_attempts))
    q_thr = float(thr.get("quality_threshold", settings.quality_threshold))

    addr = db.get(models.AddressRecord, address_id)
    if not addr:
        return {"error": "address_not_found"}

    candidates = (
        db.query(models.ImageAsset)
        .filter(
            models.ImageAsset.address_id == address_id,
            models.ImageAsset.is_temporary_candidate.is_(True),
        )
        .order_by(models.ImageAsset.uploaded_at.asc())
        .all()
    )
    if not candidates:
        return {"message": "no_candidates", "address_id": address_id}

    if addr.attempt_count >= max_attempts:
        return {"error": "max_attempts_reached", "attempt_count": addr.attempt_count}

    scored: list[tuple[models.ImageAsset, float, list[str]]] = []
    discard_notes: list[dict] = []

    for other in db.query(models.ImageAsset).filter(models.ImageAsset.address_id == address_id).all():
        if other.id != addr.selected_image_id:
            other.is_primary_for_address = False

    for img in candidates:
        local_p, tmp_del = materialize_local_path(img.stored_path, suffix=".bv")
        try:
            bgr = cv2.imread(str(local_p))
        finally:
            if tmp_del:
                local_p.unlink(missing_ok=True)
        if bgr is None:
            img.discard_reason = "fil_uleselig"
            discard_notes.append({"image_id": img.id, "reason": img.discard_reason})
            continue
        q = assess_image_quality(bgr)
        img.quality_score = q.combined
        if q.combined < q_thr:
            img.discard_reason = (
                "under_kvalitetsterskel: " + ", ".join(q.flags) if q.flags else "under_kvalitetsterskel"
            )
            discard_notes.append(
                {"image_id": img.id, "score": q.combined, "reason": img.discard_reason}
            )
            continue
        scored.append((img, q.combined, q.flags))

    addr.attempt_count = min(max_attempts, addr.attempt_count + 1)

    if not scored:
        addr.selection_metadata_json = {
            "last_run": datetime.utcnow().isoformat(),
            "discarded": discard_notes,
            "result": "no_candidate_met_threshold",
        }
        db.commit()
        return {
            "address_id": address_id,
            "selected_image_id": None,
            "status_suggestion": models.ReviewStatus.UKLART.value,
            "message": "Ingen kandidat over kvalitetsterskel — sett Uklart eller Trenger manuell vurdering.",
            "discarded": discard_notes,
        }

    winner, best_score, _ = max(scored, key=lambda t: t[1])
    addr.best_quality_score = best_score
    addr.selected_image_id = winner.id
    winner.is_primary_for_address = True
    winner.is_temporary_candidate = False

    for img, sc, _ in scored:
        if img.id != winner.id:
            img.is_primary_for_address = False
            img.discard_reason = f"ikke_best_valg (score {sc:.3f} vs {best_score:.3f})"
            discard_notes.append({"image_id": img.id, "reason": img.discard_reason})

    # Fjern alle kandidater fra aktiv utvelgelseskø (unngå gjentatte kjøringer på samme sett).
    for img in candidates:
        if img.id != winner.id:
            img.is_temporary_candidate = False

    addr.selection_metadata_json = {
        "last_run": datetime.utcnow().isoformat(),
        "selected_image_id": winner.id,
        "best_quality_score": best_score,
        "discarded": discard_notes,
    }
    db.commit()
    return {
        "address_id": address_id,
        "selected_image_id": winner.id,
        "best_quality_score": best_score,
        "discarded": discard_notes,
    }
