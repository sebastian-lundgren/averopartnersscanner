"""
Heuristisk omrangering av YOLO alarm_sign-kandidater: prioriter veggmonterte skilt (fasade,
vertikal sone, ikke ved bildets nedre kant); straff bakke/forgrunn (potter, pynt, busker), store
flate objekter og typisk «bil i bunnen».

YOLO kan fortsatt gi høy conf på feil objekt — dette er kun post-prosessering ved flere/én kandidat.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _heuristic_multiplier(b: dict[str, float]) -> tuple[float, str]:
    """Returner (multiplikator, kort begrunnelsestreng). Kun normert xywh (0–1)."""
    _, y, w, h = b["x"], b["y"], b["w"], b["h"]
    area = max(0.0, w * h)
    yc = y + h / 2.0
    y_top = y
    y_bottom = min(1.0, y + h)
    margin_bunn = 1.0 - y_bottom
    ar = w / max(h, 1e-6)
    tags: list[str] = []
    m = 1.0

    # Store irrelevante flater (bil, stor veggflate feilaktig detektert)
    if area >= 0.12:
        m *= 0.32
        tags.append("veldig_stor_flate")
    elif area >= 0.07:
        m *= 0.55
        tags.append("stor_flate")
    elif area >= 0.04:
        m *= 0.78
        tags.append("middels_stor")

    # Avstand til nederste bildekant: potter/pynt/busk sitter ofte med bunn nær ramme
    if y_bottom >= 0.93:
        m *= 0.26
        tags.append("bunnkant_naer_ramme")
    elif y_bottom >= 0.88:
        m *= 0.45
        tags.append("lav_bunnkant")
    elif y_bottom >= 0.82:
        m *= 0.66
        tags.append("nedre_sone_bunn")

    # Toppen av bbox starter lavt → hele objektet i «bakke-/inngangssone», sjelden veggskilt høyt
    if y_top >= 0.58:
        m *= 0.55
        tags.append("starter_lavt_forgrunn")
    elif y_top >= 0.48:
        m *= 0.78
        tags.append("starter_midt_nede")

    # Vertikal fasade-sone (typisk veggmontert skilt i Street View)
    if 0.16 <= yc <= 0.58:
        m *= 1.16
        tags.append("fasade_vertikal_sone")
    elif 0.58 < yc <= 0.68:
        m *= 1.03
        tags.append("midt_nedre_fasade")
    elif yc > 0.74:
        m *= 0.52
        tags.append("lavt_senter_bakke")

    if yc >= 0.70 and area >= 0.03:
        m *= 0.40
        tags.append("lav_i_bilde_og_stor")

    # Rektangulært skilt (portrett/kvadrat); straff ekstrem bredde (banner/horisontalt støy)
    if 0.32 <= ar <= 1.28:
        m *= 1.09
        tags.append("skiltlignende_aspekt")
    if ar > 2.6:
        m *= 0.78
        tags.append("veldig_bred")
    if ar < 0.22:
        m *= 0.88
        tags.append("ekstremt_smal_hoy")

    # Plausibel skiltflate (ikke mikroskopisk, ikke digert)
    if 0.0012 <= area <= 0.072:
        m *= 1.05
        tags.append("typisk_skiltstorrelse")

    reason = "+".join(tags) if tags else "neutral"
    return m, reason


def _geom_debug(box: dict[str, float]) -> tuple[float, float, float, float]:
    y = box["y"]
    h = box["h"]
    w = box["w"]
    y_bottom = min(1.0, y + h)
    margin_bunn = 1.0 - y_bottom
    ar = w / max(h, 1e-6)
    return y, y_bottom, margin_bunn, ar


def rank_alarm_sign_candidates(
    scored: list[tuple[float, dict[str, float]]],
    *,
    context: str,
    image_label: str = "",
) -> list[tuple[float, dict[str, float], float, str]]:
    """
    scored: (yolo_conf, bbox_norm_xywh) — returneres sortert etter prioritet.

    Sortering: høyest composite (conf * heuristikk), deretter høyere rå conf, deretter mindre areal.
    """
    if not scored:
        return []

    enriched: list[tuple[float, dict[str, float], float, str, float]] = []
    for cf, box in scored:
        mult, h_reason = _heuristic_multiplier(box)
        comp = max(0.0, min(1.0, cf * mult))
        area = box["w"] * box["h"]
        enriched.append((cf, box, comp, h_reason, area))

    enriched.sort(
        key=lambda t: (-t[2], -t[0], t[4]),
    )

    out: list[tuple[float, dict[str, float], float, str]] = []
    lines: list[str] = []
    for i, (cf, box, comp, h_reason, area) in enumerate(enriched):
        y_top, y_bottom, margin_bunn, ar = _geom_debug(box)
        yc = box["y"] + box["h"] / 2.0
        lines.append(
            f"  kandidat[{i}] conf_yolo={cf:.4f} mult≈{comp/max(cf,1e-9):.3f} score={comp:.4f} "
            f"area={area:.4f} y_top={y_top:.3f} y_center={yc:.3f} y_bottom={y_bottom:.3f} "
            f"margin_bunn={margin_bunn:.3f} ar={ar:.3f} tags={h_reason} "
            f"xywh=({box['x']:.3f},{box['y']:.3f},{box['w']:.3f},{box['h']:.3f})"
        )
        out.append((cf, box, comp, h_reason))

    best_cf, _, best_comp, best_h = out[0]
    second = ""
    if len(out) > 1:
        cf2, _, comp2, h2 = out[1]
        second = (
            f"\n  vs_2_plass: conf_yolo={cf2:.4f} score={comp2:.4f} tags={h2} "
            f"(vinner hvis score høyere, ved lik score høyere conf, deretter mindre areal)"
        )
    log.info(
        "YOLO bbox-rangering [%s] %s: %s kandidat(er)\n%s%s\n"
        "  VALGT: kandidat[0] (kun rangering) — conf_yolo=%.4f score=%.4f tags=%s | "
        "regel: høyest composite (conf×heuristikk); uavgjort → høyere rå conf → mindre areal | "
        "pålitelig auto-primær avgjøres av egen tillitssjekk (logg «YOLO primær-tillit»)",
        context,
        image_label,
        len(out),
        "\n".join(lines),
        second,
        best_cf,
        best_comp,
        best_h,
    )
    return out


def reorder_scored_for_storage(
    scored: list[tuple[float, dict[str, float]]],
    *,
    context: str,
    image_label: str = "",
    trust_min_conf: float = 0.45,
    trust_min_composite: float = 0.30,
) -> tuple[list[dict[str, float]], float, dict[str, Any]]:
    """
    Returner (bboxes_norm beste først, rå conf primær, tillits-info).

    trusted_primary krever både min rå conf og min composite (conf×heuristikk) på rangert nr.1 —
    ikke «beste blant svake» som pålitelig auto-valg.
    """
    ranked = rank_alarm_sign_candidates(scored, context=context, image_label=image_label)
    if not ranked:
        info: dict[str, Any] = {
            "trusted_primary": False,
            "primary_gate_reason": "ingen_kandidater",
            "primary_composite": 0.0,
            "primary_mult": 0.0,
        }
        log.info(
            "YOLO primær-tillit [%s] %s: trusted=False | %s",
            context,
            image_label,
            info["primary_gate_reason"],
        )
        return [], 0.0, info

    bboxes = [b for _, b, _, _ in ranked]
    primary_conf = float(ranked[0][0])
    primary_comp = float(ranked[0][2])
    mult = primary_comp / max(primary_conf, 1e-9)
    ok_parts: list[str] = []
    fail_parts: list[str] = []
    if primary_conf >= trust_min_conf:
        ok_parts.append(f"conf>={trust_min_conf}")
    else:
        fail_parts.append(f"conf={primary_conf:.3f}<{trust_min_conf}")
    if primary_comp >= trust_min_composite:
        ok_parts.append(f"composite>={trust_min_composite}")
    else:
        fail_parts.append(f"composite={primary_comp:.3f}<{trust_min_composite}")
    trusted = len(fail_parts) == 0
    gate = "auto_primær_ok: " + ", ".join(ok_parts) if trusted else "ikke_auto_primær: " + "; ".join(fail_parts)

    log.info(
        "YOLO primær-tillit [%s] %s: trusted=%s conf=%.4f composite=%.4f mult≈%.3f | %s",
        context,
        image_label,
        trusted,
        primary_conf,
        primary_comp,
        mult,
        gate,
    )

    return bboxes, primary_conf, {
        "trusted_primary": trusted,
        "primary_gate_reason": gate,
        "primary_composite": primary_comp,
        "primary_mult": mult,
    }
