"""
Åpne Street View slik en bruker ville: Google Maps-søk på adresse, deretter første Street View-forslag.
Ingen eget geometrisk frontpunkt og ingen pin-/kompass-POV-logikk her.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout

from runner import config
from runner.navigator import StreetViewAttempt, open_streetview_view, wait_view_ready
from runner.timing import step_timer

log = logging.getLogger(__name__)

# Tekst/aria som ofte brukes i Maps (engelsk + norsk; flere språk kan legges til).
_STREET_VIEW_NAME_RES = (
    re.compile(r"street\s*view", re.I),
    re.compile(r"gatevisning", re.I),
    re.compile(r"gatuvy", re.I),
    re.compile(r"gadevisning", re.I),
)


@dataclass(frozen=True)
class MapsPanoSnapshot:
    """Pano-posisjon og retning lest fra Maps-URL etter at Street View er åpnet."""

    lat: float
    lon: float
    heading_deg: float | None
    page_url: str
    panoid: str | None = None


def _extract_panoid_from_maps_url(url: str) -> str | None:
    m = re.search(r"!3m5!1s([^!]+)", url)
    return m.group(1) if m else None


def parse_maps_pano_from_url(url: str) -> MapsPanoSnapshot | None:
    """
    Les kameraposisjon (og ev. heading) fra typisk Google Maps Street View-URL.
    """
    # Street View: @lat,lon,3a,75y,12.5h,90t
    m = re.search(
        r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(?:\d+(?:\.\d+)?)a,(?:\d+(?:\.\d+)?)y,(-?\d+(?:\.\d+)?)h",
        url,
    )
    pid = _extract_panoid_from_maps_url(url)
    if m:
        return MapsPanoSnapshot(
            lat=float(m.group(1)),
            lon=float(m.group(2)),
            heading_deg=float(m.group(3)),
            page_url=url,
            panoid=pid,
        )
    m3 = re.search(r"!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)", url)
    if m3:
        return MapsPanoSnapshot(
            lat=float(m3.group(1)),
            lon=float(m3.group(2)),
            heading_deg=None,
            page_url=url,
            panoid=pid,
        )
    m4 = re.search(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),", url)
    if m4:
        return MapsPanoSnapshot(
            lat=float(m4.group(1)),
            lon=float(m4.group(2)),
            heading_deg=None,
            page_url=url,
            panoid=pid,
        )
    return None


def _dismiss_common_overlays(page: Page) -> None:
    for label in (
        "Accept all",
        "I agree",
        "Godta alle",
        "Akzeptieren",
        "Alle akzeptieren",
    ):
        try:
            page.get_by_role("button", name=re.compile(re.escape(label), re.I)).click(timeout=1800)
            page.wait_for_timeout(400)
            log.info("MAPS_CONSENT dismissed via button matching %r", label)
            return
        except PlaywrightTimeout:
            continue
        except Exception:
            continue


def _try_click_street_view_entry(page: Page) -> bool:
    for pat in _STREET_VIEW_NAME_RES:
        try:
            loc = page.get_by_role("button", name=pat)
            if loc.count() > 0:
                loc.first.click(timeout=5000)
                log.info("MAPS_SV_CLICK role=button name=%s", pat.pattern)
                return True
        except (PlaywrightTimeout, Exception):
            pass
    try:
        loc = page.locator('[aria-label*="Street View" i]').first
        if loc.count():
            loc.click(timeout=5000)
            log.info("MAPS_SV_CLICK aria-label*=Street View")
            return True
    except (PlaywrightTimeout, Exception):
        pass
    try:
        loc = page.locator('a[href*="/maps/place/"]').filter(
            has=page.locator("text=/street\\s*view/i"),
        )
        if loc.count():
            loc.first.click(timeout=5000)
            log.info("MAPS_SV_CLICK place link with Street View text")
            return True
    except (PlaywrightTimeout, Exception):
        pass
    try:
        for a in page.locator('a[href*="/@"]').all()[:25]:
            href = a.get_attribute("href") or ""
            if "3a" in href and ("data=" in href or "map_action" in href or "pano" in href):
                a.click(timeout=5000)
                log.info("MAPS_SV_CLICK direct @ link with 3a")
                return True
    except (PlaywrightTimeout, Exception):
        pass
    try:
        loc = page.locator('button[jsaction*="streetview" i], button[jsaction*="streetView" i]').first
        if loc.count():
            loc.click(timeout=5000)
            log.info("MAPS_SV_CLICK button jsaction*=streetview")
            return True
    except (PlaywrightTimeout, Exception):
        pass
    return False


def _try_click_first_thumbnail(page: Page) -> bool:
    candidates = (
        '[aria-label*="bilder" i] button img',
        '[aria-label*="photos" i] button img',
        '[aria-label*="bilder" i] a img',
        '[aria-label*="photos" i] a img',
        '[data-section-id*="photos" i] button img',
        '[data-section-id*="photos" i] a img',
    )
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=2500):
                loc.click(timeout=5000)
                return True
        except Exception:
            continue
    try:
        loc = page.locator('button[jsaction="pane.wfvdle7.heroHeaderImage"]').first
        if loc.count() and loc.is_visible(timeout=2500):
            loc.click(timeout=5000)
            log.info("THUMBNAIL_FORCED_CLICK_SELECTOR %s", 'button[jsaction="pane.wfvdle7.heroHeaderImage"]')
            return True
    except Exception:
        pass
    try:
        loc = page.locator('button:has(img[src*="streetviewpixels-pa.googleapis.com/v1/thumbnail"])').first
        if loc.count() and loc.is_visible(timeout=2500):
            loc.click(timeout=5000)
            log.info(
                "THUMBNAIL_FORCED_CLICK_SELECTOR %s",
                'button:has(img[src*="streetviewpixels-pa.googleapis.com/v1/thumbnail"])',
            )
            return True
    except Exception:
        pass
    try:
        img = page.locator('img[src*="streetviewpixels-pa.googleapis.com/v1/thumbnail"]').first
        if img.count() and img.is_visible(timeout=2500):
            btn = img.locator("xpath=ancestor::button[1]")
            if btn.count() and btn.first.is_visible(timeout=2500):
                btn.first.click(timeout=5000)
                log.info(
                    "THUMBNAIL_FORCED_CLICK_SELECTOR %s",
                    'img[src*="streetviewpixels-pa.googleapis.com/v1/thumbnail"] -> ancestor::button[1]',
                )
                return True
    except Exception:
        pass
    return False


def _looks_like_street_view_url(url: str) -> bool:
    u = (url or "").lower()
    return (
        "map_action=pano" in u
        or ",3a," in u
        or ("!3m5!1s" in u and "/@" in u)
    )


def open_default_streetview_from_address(
    page: Page,
    address: str,
    *,
    subsequent_address: bool = False,
) -> tuple[bool, MapsPanoSnapshot | None, str]:
    """
    Søk på adresse i Google Maps og åpne Street View slik brukeren typisk gjør.
    Returnerer (ok, snapshot, feiltekst ved avbrudd).
    """
    log.info('MAPS_ADDR_SEARCH query=%r', address)
    enc = urllib.parse.quote(address, safe="")
    search_url = f"https://www.google.com/maps/search/{enc}"
    with step_timer(log, "maps_search_goto", url=search_url[:120]):
        page.goto(search_url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT_MS)
    page.wait_for_timeout(1200)
    _dismiss_common_overlays(page)
    base_wait = (
        config.VIEW_WAIT_AFTER_FIRST_ADDR_MS if subsequent_address else config.VIEW_WAIT_MS
    )
    page.wait_for_timeout(min(3200, max(800, int(base_wait * 0.35))))

    clicked = _try_click_street_view_entry(page)
    if not clicked:
        log.info("THUMBNAIL_FORCED_TRY address=%r", address)
        log.info("THUMB_DEBUG_URL_BEFORE_RETURN %s", page.url)
        try:
            page.goto(search_url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT_MS)
            log.info("THUMB_DEBUG_URL_AFTER_RETURN %s", page.url)
            _dismiss_common_overlays(page)
            for sel in (
                'button[jsaction="pane.wfvdle7.heroHeaderImage"]',
                'img[src*="streetviewpixels-pa.googleapis.com/v1/thumbnail"]',
                'img[src*="panoid="]',
            ):
                try:
                    loc = page.locator(sel)
                    count = loc.count()
                    visible = bool(count > 0 and loc.first.is_visible(timeout=1200))
                    log.info(
                        "THUMB_DEBUG_SELECTOR %s count=%s visible=%s",
                        sel,
                        count,
                        str(visible).lower(),
                    )
                except Exception:
                    log.info("THUMB_DEBUG_SELECTOR %s count=0 visible=false", sel)
            place_card_ok = False
            for sel in ('h1', '[role="main"] h1', '[data-section-id]', '[aria-label*="Results" i]'):
                try:
                    loc = page.locator(sel).first
                    if loc.count() and loc.is_visible(timeout=2200):
                        place_card_ok = True
                        break
                except Exception:
                    continue
            if not place_card_ok:
                raise RuntimeError("place card-state ikke synlig etter return to search")
            thumb_area_ok = False
            for sel in (
                '[aria-label*="bilder" i] button img',
                '[aria-label*="photos" i] button img',
                '[aria-label*="bilder" i] a img',
                '[aria-label*="photos" i] a img',
                '[data-section-id*="photos" i] button img',
                '[data-section-id*="photos" i] a img',
            ):
                try:
                    loc = page.locator(sel).first
                    if loc.count() and loc.is_visible(timeout=2200):
                        thumb_area_ok = True
                        break
                except Exception:
                    continue
            if not thumb_area_ok:
                raise RuntimeError("thumbnail/bilder-område ikke synlig i place card-state")
            thumb_ok = _try_click_first_thumbnail(page)
            if thumb_ok:
                page.wait_for_timeout(2200)
                canvas_ok = page.locator("canvas").count() > 0
                img_ok = page.locator("img").count() > 0
                if canvas_ok or img_ok:
                    log.info("THUMBNAIL_FORCED_CAPTURED address=%r", address)
                    return True, None, "thumbnail_forced"
        except Exception:
            pass
        try:
            safe_addr = re.sub(r"[^a-zA-Z0-9_-]+", "_", address).strip("_")[:80] or "unknown"
            dbg_path = f"/tmp/thumb_debug_{safe_addr}.png"
            page.screenshot(path=dbg_path, full_page=True)
            log.info("THUMB_DEBUG_SCREENSHOT %s", dbg_path)
        except Exception:
            pass
        log.warning("THUMBNAIL_FORCED_FAILED address=%r", address)
        reason = "fant ingen klikkbar Street View-kontroll etter adressesøk"
        log.warning("MAPS_SV_MAIN_FAIL %s", reason)
        return False, None, reason

    page.wait_for_timeout(900)
    deadline = 18.0
    step_s = 0.45
    elapsed = 0.0
    snapshot: MapsPanoSnapshot | None = None
    while elapsed < deadline:
        href = page.url
        snapshot = parse_maps_pano_from_url(href)
        canvas_ok = page.locator("canvas").count() > 0
        sv_url = _looks_like_street_view_url(href)
        if snapshot and sv_url:
            log.info(
                "MAPS_SV_MAIN_OPEN url_head=%s lat=%.6f lon=%.6f heading=%s canvas=%s",
                href[:100],
                snapshot.lat,
                snapshot.lon,
                f"{snapshot.heading_deg:.1f}°" if snapshot.heading_deg is not None else "ukjent",
                canvas_ok,
            )
            return True, snapshot, ""
        if sv_url and canvas_ok:
            log.info(
                "MAPS_SV_MAIN_OPEN url_head=%s uten_snapshot_parse (canvas+SV-URL) — godtar hovedview",
                href[:100],
            )
            return True, None, ""
        page.wait_for_timeout(int(step_s * 1000))
        elapsed += step_s

    reason = "Street View åpnet ikke innen timeout (URL/canvas matcher ikke forventet SV)"
    log.warning("MAPS_SV_MAIN_FAIL %s last_url=%s", reason, page.url[:160])
    return False, None, reason


def refresh_pano_snapshot(page: Page) -> MapsPanoSnapshot | None:
    """Oppdater snapshot fra nåværende URL (etter at Street View har satt seg)."""
    return parse_maps_pano_from_url(page.url)


def open_first_distinct_neighbor_pano(
    page: Page,
    attempts: list[StreetViewAttempt],
    *,
    tgt_lat: float,
    tgt_lon: float,
    quick_settle: bool,
) -> tuple[StreetViewAttempt, float] | None:
    """
    Prøv geometriske naboviewpoints i samme rekkefølge som før. Første steg der
    open_streetview_view lykkes og canvas er klar, returneres. URL-parse er valgfri
    (main_loop bruker plan-koordinater som fallback). Ingen «egen nabopano»-filtrering.
    """
    for view in attempts:
        h = open_streetview_view(
            page,
            view,
            quick_settle=quick_settle,
            subsequent_address=False,
            target_lat=tgt_lat,
            target_lon=tgt_lon,
        )
        if h is None:
            continue
        if not wait_view_ready(page, fast=quick_settle):
            continue
        snap = parse_maps_pano_from_url(page.url)
        if snap is None:
            log.info(
                "SV_NEIGHBOR_OPEN view_id=%s ok uten_URL_parse (parse_maps_pano_from_url=None) "
                "— nabosteg godtatt etter canvas/h | plan_cam=%.6f,%.6f | url_head=%s",
                view.view_id,
                view.camera_lat,
                view.camera_lon,
                (page.url or "")[:120],
            )
            return (view, h)
        pid = (snap.panoid or "").strip()
        log.info(
            "SV_NEIGHBOR_OPEN view_id=%s ok første_vellykkede_steg panoid=%s pano_lat=%.6f pano_lon=%.6f",
            view.view_id,
            pid or "ukjent",
            snap.lat,
            snap.lon,
        )
        return (view, h)
    return None
