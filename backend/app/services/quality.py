"""Kvalitetsscore for bilder: skarphet, lys, synlighet — unknown-first."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityBreakdown:
    sharpness: float  # 0–1
    exposure: float  # 0–1 (pen midt, straff ved utbrent/mørk)
    visibility: float  # 0–1 (kontrast i fasaderegion)
    distance_proxy: float  # 0–1 (høyere = mer detalj / nærmere antatt)
    combined: float
    flags: list[str]


def _norm01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    t = (x - lo) / (hi - lo)
    return float(np.clip(t, 0.0, 1.0))


def assess_image_quality(bgr: np.ndarray) -> QualityBreakdown:
    """Returnerer 0–1 scores. Lav skarphet eller dårlig eksponering senker totalen."""
    flags: list[str] = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())
    sharpness = _norm01(lap_var, 30.0, 800.0)
    if sharpness < 0.25:
        flags.append("uskarpt")

    mean_brightness = float(np.mean(gray)) / 255.0
    exposure = 1.0 - abs(mean_brightness - 0.45) * 2.0
    exposure = float(np.clip(exposure, 0.0, 1.0))
    if mean_brightness > 0.92:
        flags.append("sterk_refleks_eller_utbrent")
        exposure *= 0.5
    if mean_brightness < 0.12:
        flags.append("for_morkt")
        exposure *= 0.6

    # Øvre del av bildet (typisk fasade/skilt)
    top = gray[: max(1, h // 2), :]
    contrast = float(np.std(top)) / 80.0
    visibility = float(np.clip(contrast, 0.0, 1.0))
    if visibility < 0.2:
        flags.append("lav_kontrast_omrade")

    # Enkel "detalj" = gradientmagnitude i midten
    mid = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    gx = cv2.Sobel(mid, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(mid, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    distance_proxy = _norm01(float(np.mean(mag)), 5.0, 80.0)
    if distance_proxy < 0.2:
        flags.append("for_langt_unna_eller_lav_detalj")

    combined = (
        0.35 * sharpness + 0.25 * exposure + 0.2 * visibility + 0.2 * distance_proxy
    )
    combined = float(np.clip(combined, 0.0, 1.0))

    return QualityBreakdown(
        sharpness=sharpness,
        exposure=exposure,
        visibility=visibility,
        distance_proxy=distance_proxy,
        combined=combined,
        flags=flags,
    )
