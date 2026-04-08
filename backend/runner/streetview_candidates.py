"""
Venstre/høyre: egne Street View-panoramaer (nabopunkter vinkelrett på hovedpano→hus),
heading mot hus fra hvert punkt — ikke rotasjon fra samme pano.
"""

from __future__ import annotations

import logging
import math

from runner import config
from runner.navigator import StreetViewAttempt

log = logging.getLogger(__name__)

R_EARTH_M = 6_371_000.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH_M * math.asin(min(1.0, math.sqrt(a)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _destination(lat: float, lon: float, bearing: float, dist_m: float) -> tuple[float, float]:
    br = math.radians(bearing)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    ang = dist_m / R_EARTH_M
    lat2 = math.asin(math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _angle_diff_deg(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _house_front_sector_ok(
    h_lat: float,
    h_lon: float,
    p0_lat: float,
    p0_lon: float,
    q_lat: float,
    q_lon: float,
    *,
    max_deg: float = 82.0,
) -> bool:
    b_ref = _bearing_deg(h_lat, h_lon, p0_lat, p0_lon)
    b_q = _bearing_deg(h_lat, h_lon, q_lat, q_lon)
    return _angle_diff_deg(b_ref, b_q) <= max_deg


def _fov_for_distance_m(d: float) -> float:
    d = max(d, 5.5)
    return float(max(45.0, min(70.0, 500.0 / (d + 4.0) + 28.0)))


def _neighbor_from_coords(
    house_lat: float,
    house_lon: float,
    p0_lat: float,
    p0_lon: float,
    qlat: float,
    qlon: float,
    *,
    view_id: str,
    desc: str,
) -> tuple[StreetViewAttempt | None, str | None]:
    if _haversine_m(qlat, qlon, p0_lat, p0_lon) < 3.5:
        return None, "for kort fra hovedpano (sannsynlig samme pano)"
    if (round(qlat * 1e5), round(qlon * 1e5)) == (
        round(p0_lat * 1e5),
        round(p0_lon * 1e5),
    ):
        return None, "samme rutenett som hovedpano"
    if not _house_front_sector_ok(house_lat, house_lon, p0_lat, p0_lon, qlat, qlon):
        return None, "utenfor trygg front-sektor fra hus"
    dist_h = _haversine_m(qlat, qlon, house_lat, house_lon)
    if dist_h < 3.5:
        return None, f"for nær hus ({dist_h:.1f} m)"
    if dist_h > 58.0:
        return None, f"for langt fra hus ({dist_h:.1f} m)"
    h_back = _bearing_deg(qlat, qlon, house_lat, house_lon)
    fov = _fov_for_distance_m(dist_h)
    pitch = 2.0 if dist_h < 14.0 else 0.0
    reason = f"{view_id}: {desc}; heading_etter=mot hus {h_back:.0f}°"
    return (
        StreetViewAttempt(
            view_id=view_id,
            camera_lat=qlat,
            camera_lon=qlon,
            heading_deg=None,
            pitch_deg=pitch,
            fov_deg=fov,
            plan_reason=reason,
        ),
        None,
    )


def _neighbor_step_m_attempt(
    house_lat: float,
    house_lon: float,
    p0_lat: float,
    p0_lon: float,
    *,
    side: str,
    view_id: str,
    step_m: float,
) -> tuple[StreetViewAttempt | None, str | None]:
    step_m = max(5.5, min(16.0, step_m))
    b_ph = _bearing_deg(p0_lat, p0_lon, house_lat, house_lon)
    br = (b_ph - 90.0) % 360.0 if side == "left" else (b_ph + 90.0) % 360.0
    qlat, qlon = _destination(p0_lat, p0_lon, br, step_m)
    lr = "venstre" if side == "left" else "høyre"
    return _neighbor_from_coords(
        house_lat,
        house_lon,
        p0_lat,
        p0_lon,
        qlat,
        qlon,
        view_id=view_id,
        desc=f"~{step_m:.1f} m {lr} for hovedpano (egen nabopano)",
    )


def _build_side_chain(
    house_lat: float,
    house_lon: float,
    anchor_lat: float,
    anchor_lon: float,
    *,
    side: str,
    view_id: str,
    step_list: tuple[float, ...],
    anchor_grid: tuple[int, int],
    shared_seen: set[tuple[int, int]],
) -> tuple[list[StreetViewAttempt], str]:
    chain: list[StreetViewAttempt] = []
    for step_m in step_list:
        a, skip = _neighbor_step_m_attempt(
            house_lat,
            house_lon,
            anchor_lat,
            anchor_lon,
            side=side,
            view_id=view_id,
            step_m=step_m,
        )
        if a is None:
            log.info("SV_NEIGHBOR %s steg %.1f m: %s", view_id, step_m, skip or "ukjent")
            continue
        key = (round(a.camera_lat * 1e5), round(a.camera_lon * 1e5))
        if key == anchor_grid or key in shared_seen:
            continue
        if key in {(round(x.camera_lat * 1e5), round(x.camera_lon * 1e5)) for x in chain}:
            continue
        shared_seen.add(key)
        chain.append(a)
        log.info(
            "SV_NEIGHBOR %s steg %.1f m -> kandidat cam=%.6f,%.6f",
            view_id,
            step_m,
            a.camera_lat,
            a.camera_lon,
        )
    if not chain:
        return [], "hoppet: ingen gyldige nabopunkter i front-sektor"
    return chain, f"plan: {len(chain)} geometrisk(e) nabotrinn"


def build_lateral_front_attempts(
    house_lat: float,
    house_lon: float,
    anchor_lat: float,
    anchor_lon: float,
    *,
    want_left: bool,
    want_right: bool,
) -> tuple[dict[str, str], list[StreetViewAttempt], list[StreetViewAttempt]]:
    status: dict[str, str] = {}
    steps = config.FRONT_NEIGHBOR_STEP_TRIES_M
    anchor_grid = (round(anchor_lat * 1e5), round(anchor_lon * 1e5))
    shared_seen: set[tuple[int, int]] = set()

    log.info(
        "SV_NEIGHBOR_PLAN hovedpano=%.6f,%.6f hus_maal=%.6f,%.6f step_tries_m=%s",
        anchor_lat,
        anchor_lon,
        house_lat,
        house_lon,
        steps,
    )

    if want_left:
        left_chain, st = _build_side_chain(
            house_lat,
            house_lon,
            anchor_lat,
            anchor_lon,
            side="left",
            view_id="front_left",
            step_list=steps,
            anchor_grid=anchor_grid,
            shared_seen=shared_seen,
        )
        status["front_left"] = st
        log.info("SV_NEIGHBOR venstre %s (kjede=%s punkter)", st, len(left_chain))
    else:
        status["front_left"] = "hoppet: ikke med i plan (maks forsøk)"
        left_chain = []
        log.info("SV_NEIGHBOR venstre %s", status["front_left"])

    if want_right:
        right_chain, st = _build_side_chain(
            house_lat,
            house_lon,
            anchor_lat,
            anchor_lon,
            side="right",
            view_id="front_right",
            step_list=steps,
            anchor_grid=anchor_grid,
            shared_seen=shared_seen,
        )
        status["front_right"] = st
        log.info("SV_NEIGHBOR høyre %s (kjede=%s punkter)", st, len(right_chain))
    else:
        status["front_right"] = "hoppet: ikke med i plan (maks forsøk)"
        right_chain = []
        log.info("SV_NEIGHBOR høyre %s", status["front_right"])

    return status, left_chain, right_chain
