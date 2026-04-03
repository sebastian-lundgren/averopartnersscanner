"""Enkel prioritering for review-kø."""

from sqlalchemy.orm import Session

from app import models


def compute_priority(pred: models.Prediction, db: Session) -> float:
    """
    Høyere = viktigere å se først.
    Lav confidence, ikke fullført review, evt. likhet med tidligere feil (forenklet: feiltype-felt brukes etter review).
    """
    base = (100 - pred.confidence) / 100.0 * 50.0
    if pred.predicted_status == models.ReviewStatus.UKLART.value:
        base += 15
    if pred.predicted_status == models.ReviewStatus.TRENGER_MANUELL.value:
        base += 10
    # Boost hvis bildet er knyttet til adresse med flere forsøk
    img = pred.image
    if img and img.address_id:
        addr = db.get(models.AddressRecord, img.address_id)
        if addr and addr.attempt_count and addr.attempt_count > 1:
            base += 8
    return round(base, 2)


def refresh_prediction_priority(db: Session, pred: models.Prediction) -> None:
    pred.priority_score = compute_priority(pred, db)
