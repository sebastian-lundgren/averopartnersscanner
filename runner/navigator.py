"""Google Street View i Playwright — viewpoint = kameraposisjon, heading/pitch/fov mot scene."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout

from runner import config
from runner.timing import step_timer

log = logging.getLogger(__name__)


def bearing_deg_camera_toward_target(
    cam_lat: float, cam_lon: float, target_lat: float, target_lon: float
) -> float:
    """Absolutt azimut (grader, 0–360, klokka fra nord) fra kamerapunkt mot mål."""
    phi1, phi2 = math.radians(cam_lat), math.radians(target_lat)
    dl = math.radians(target_lon - cam_lon)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


@dataclass(frozen=True)
class StreetViewAttempt:
    """Ett forsøk: hvor kamera står, hvordan det peker, og kort begrunnelse for planen."""

    view_id: str
    camera_lat: float
    camera_lon: float
    heading_deg: float | None  # None = ikke sett i URL (sjelden)
    pitch_deg: float
    fov_deg: float
    plan_reason: str = ""


def open_streetview_view(
    page: Page,
    view: StreetViewAttempt,
    *,
    quick_settle: bool = False,
    subsequent_address: bool = False,
    target_lat: float | None = None,
    target_lon: float | None = None,
    skip_navigation: bool = False,
) -> float | None:
    """
    Naviger til Street View ved kameraposisjon. Heading i URL settes ut fra målhus når target_* er gitt.
    skip_navigation=True: ikke page.goto (allerede på Google-hovedpano etter Maps-søk).
    Returnerer brukt heading (grader) eller None.
    """
    if skip_navigation:
        wait_ms = (
            config.VIEW_WAIT_QUICK_MS
            if quick_settle
            else (
                config.VIEW_WAIT_AFTER_FIRST_ADDR_MS
                if subsequent_address
                else config.VIEW_WAIT_MS
            )
        )
        log.info(
            "SV skip goto (Google Maps hovedpano) id=%s cam=%.6f,%.6f heading_plan=%s",
            view.view_id,
            view.camera_lat,
            view.camera_lon,
            f"{view.heading_deg:.2f}°" if view.heading_deg is not None else "ukjent",
        )
        with step_timer(log, "sv_skip_goto_settle_ms", view_id=view.view_id, wait_ms=wait_ms):
            page.wait_for_timeout(wait_ms)
        if view.heading_deg is not None:
            return view.heading_deg
        if target_lat is not None and target_lon is not None:
            return bearing_deg_camera_toward_target(
                view.camera_lat, view.camera_lon, target_lat, target_lon
            )
        return None

    is_heading_pivot = "_pivot_" in view.view_id
    heading_use: float | None
    if is_heading_pivot:
        heading_use = view.heading_deg
        if heading_use is None and target_lat is not None and target_lon is not None:
            heading_use = bearing_deg_camera_toward_target(
                view.camera_lat, view.camera_lon, target_lat, target_lon
            )
    elif target_lat is not None and target_lon is not None:
        heading_use = bearing_deg_camera_toward_target(
            view.camera_lat, view.camera_lon, target_lat, target_lon
        )
    else:
        heading_use = view.heading_deg

    q = (
        f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={view.camera_lat},{view.camera_lon}"
        f"&pitch={view.pitch_deg}&fov={view.fov_deg}"
    )
    if heading_use is not None:
        q += f"&heading={heading_use}"
    log.info(
        "SV goto id=%s cam=%.6f,%.6f heading_url=%s pitch=%s fov=%s lateral_neighbor=%s | %s",
        view.view_id,
        view.camera_lat,
        view.camera_lon,
        f"{heading_use:.2f}°" if heading_use is not None else "none",
        view.pitch_deg,
        view.fov_deg,
        view.view_id in ("front_left", "front_right"),
        view.plan_reason[:120] + ("…" if len(view.plan_reason) > 120 else ""),
    )
    with step_timer(log, "sv_page_goto", view_id=view.view_id):
        page.goto(q, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT_MS)
    if quick_settle:
        wait_ms = config.VIEW_WAIT_QUICK_MS
    elif subsequent_address:
        wait_ms = config.VIEW_WAIT_AFTER_FIRST_ADDR_MS
    else:
        wait_ms = config.VIEW_WAIT_MS
    with step_timer(log, "sv_post_goto_settle_ms", view_id=view.view_id, wait_ms=wait_ms):
        page.wait_for_timeout(wait_ms)

    return heading_use


def focus_street_view_for_capture(page: Page, *, allow_scene_click: bool = True) -> None:
    """
    Sørg for fokus før skjermdump. allow_scene_click=False: ikke klikk body (unngår SV-navigasjon i canvas).
    """
    log.info(
        "SV_CAPTURE view_milestone=focus_interaction_start allow_scene_click=%s",
        allow_scene_click,
    )
    page.wait_for_timeout(config.FOCUS_PRE_CLICK_MS)
    if allow_scene_click:
        page.locator("body").click(timeout=5000)
    else:
        log.info("SV_CAPTURE view_milestone=focus_skip_body_click (front_right / samme pano)")
    page.wait_for_timeout(config.STABILIZE_MS)
    log.info(
        "SV_CAPTURE view_milestone=focus_interaction_done post_stabilize_ms=%s",
        config.STABILIZE_MS,
    )


def settle_streetview_after_canvas_ready(
    page: Page, *, first_attempt: bool, subsequent_address: bool = False
) -> None:
    """Ekstra vent etter canvas er synlig — WebGL fyller ofte skarpe tiles først etterpå."""
    if first_attempt:
        ms = (
            config.STREETVIEW_POST_READY_MS_FIRST_SUBSEQUENT_ADDR
            if subsequent_address
            else config.STREETVIEW_POST_READY_MS_FIRST
        )
    else:
        ms = config.STREETVIEW_POST_READY_MS_NEXT
    ms = max(0, int(ms))
    log.info(
        "SV_CAPTURE view_milestone=sharp_tile_settle_start ms=%s first_attempt=%s",
        ms,
        first_attempt,
    )
    if ms:
        page.wait_for_timeout(ms)
    log.info(
        "SV_CAPTURE view_milestone=sharp_tile_settle_done first_attempt=%s",
        first_attempt,
    )


def wait_view_ready(page: Page, retries: int = 3, fast: bool = False) -> bool:
    """Vent til Street View-canvas er synlig. fast=True: kortere timeout/poll (forsøk 2+)."""
    with step_timer(log, "wait_view_ready", fast=fast):
        rmax = min(retries, 3) if fast else max(4, retries)
        if fast:
            canvas_timeout = 3200
            between_ms = 850
        else:
            canvas_timeout = 7000
            between_ms = 2200
        for i in range(rmax):
            try:
                canvas = page.locator("canvas").first
                if canvas.count() and canvas.is_visible(timeout=canvas_timeout):
                    log.info(
                        "SV_CAPTURE view_milestone=canvas_visible_accepted fast=%s retry_i=%s",
                        fast,
                        i,
                    )
                    return True
            except PlaywrightTimeout:
                pass
            log.warning("Vent på canvas, retry %s/%s", i + 1, rmax)
            page.wait_for_timeout(between_ms)
        ok = page.locator("canvas").count() > 0
        if ok:
            log.info(
                "SV_CAPTURE view_milestone=canvas_present_after_poll fast=%s (svak indikator)",
                fast,
            )
        return ok
