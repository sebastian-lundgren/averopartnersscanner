"""Maps place-card hero-thumbnail capture."""

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


@dataclass(frozen=True)
class MapsHeroThumbnail:
    src_url: str
    fetch_url: str
    panoid: str | None
    yaw: float | None
    pitch: float | None
    thumbfov: float | None
    w: int | None
    h: int | None


@dataclass(frozen=True)
class MapsPanoSnapshot:
    lat: float
    lon: float
    heading_deg: float | None
    page_url: str
    panoid: str | None = None


def _extract_panoid_from_maps_url(url: str) -> str | None:
    m = re.search(r"!3m5!1s([^!]+)", url)
    return m.group(1) if m else None


def parse_maps_pano_from_url(url: str) -> MapsPanoSnapshot | None:
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
    for label in ("Accept all", "I agree", "Godta alle", "Akzeptieren", "Alle akzeptieren"):
        try:
            page.get_by_role("button", name=re.compile(re.escape(label), re.I)).click(timeout=1800)
            page.wait_for_timeout(400)
            return
        except PlaywrightTimeout:
            continue
        except Exception:
            continue


def _read_visible_hero_src(page: Page) -> str:
    sel_primary = 'button[jsaction="pane.wfvdle7.heroHeaderImage"] img[src*="streetviewpixels-pa.googleapis.com/v1/thumbnail"]'
    sel_fallback = 'img[src*="streetviewpixels-pa.googleapis.com/v1/thumbnail"][src*="panoid="]'
    for sel in (sel_primary, sel_fallback):
        try:
            img = page.locator(sel).first
            if img.count() and img.is_visible(timeout=2200):
                src = (img.get_attribute("src") or "").strip()
                if src:
                    return src
        except Exception:
            continue
    return ""


def _try_click_street_view_entry(page: Page) -> bool:
    pats = (
        re.compile(r"street\s*view", re.I),
        re.compile(r"gatevisning", re.I),
        re.compile(r"gatuvy", re.I),
        re.compile(r"gadevisning", re.I),
    )
    for pat in pats:
        try:
            loc = page.get_by_role("button", name=pat)
            if loc.count() > 0:
                loc.first.click(timeout=5000)
                return True
        except Exception:
            pass
    try:
        loc = page.locator('[aria-label*="Street View" i]').first
        if loc.count():
            loc.click(timeout=5000)
            return True
    except Exception:
        pass
    try:
        loc = page.locator('button[jsaction*="streetview" i], button[jsaction*="streetView" i]').first
        if loc.count():
            loc.click(timeout=5000)
            return True
    except Exception:
        pass
    return False


def _looks_like_street_view_url(url: str) -> bool:
    u = (url or "").lower()
    return ("map_action=pano" in u) or (",3a," in u) or ("!3m5!1s" in u and "/@" in u)


def _parse_hero_thumbnail(src_url: str) -> MapsHeroThumbnail:
    raw = (src_url or "").replace("&amp;", "&").strip()
    parsed = urllib.parse.urlparse(raw)
    q = urllib.parse.parse_qs(parsed.query or "")

    def _f(name: str) -> float | None:
        try:
            return float((q.get(name) or [None])[0])
        except Exception:
            return None

    def _i(name: str) -> int | None:
        try:
            return int(float((q.get(name) or [None])[0]))
        except Exception:
            return None

    panoid = (q.get("panoid") or [None])[0]
    yaw = _f("yaw")
    pitch = _f("pitch")
    thumbfov = _f("thumbfov")
    w = _i("w")
    h = _i("h")

    q2 = dict(q)
    if w:
        q2["w"] = [str(min(max(w, 1200), 1600))]
    if h:
        q2["h"] = [str(min(max(h, 900), 1200))]
    fetch_q = urllib.parse.urlencode([(k, v) for k, vals in q2.items() for v in vals], doseq=True)
    fetch_url = urllib.parse.urlunparse(parsed._replace(query=fetch_q))

    return MapsHeroThumbnail(
        src_url=raw,
        fetch_url=fetch_url,
        panoid=panoid,
        yaw=yaw,
        pitch=pitch,
        thumbfov=thumbfov,
        w=w,
        h=h,
    )


def open_placecard_hero_thumbnail(page: Page, address: str, *, subsequent_address: bool = False) -> tuple[bool, MapsHeroThumbnail | None, str]:
    log.info('MAPS_ADDR_SEARCH query=%r', address)
    enc = urllib.parse.quote(address, safe="")
    search_url = f"https://www.google.com/maps/search/{enc}"
    with step_timer(log, "maps_search_goto", url=search_url[:120]):
        page.goto(search_url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT_MS)
    page.wait_for_timeout(1200)
    _dismiss_common_overlays(page)
    base_wait = config.VIEW_WAIT_AFTER_FIRST_ADDR_MS if subsequent_address else config.VIEW_WAIT_MS
    page.wait_for_timeout(min(3200, max(800, int(base_wait * 0.35))))

    place_card_ok = False
    src = ""
    for _ in range(20):  # ~8s total med 400ms intervall
        place_card_ok = False
        for sel in ("h1", '[role="main"] h1'):
            try:
                loc = page.locator(sel).first
                if loc.count() and loc.is_visible(timeout=300):
                    place_card_ok = True
                    break
            except Exception:
                continue
        if place_card_ok:
            src = _read_visible_hero_src(page)
            if src:
                hero = _parse_hero_thumbnail(src)
                return True, hero, ""
        page.wait_for_timeout(400)
    if not place_card_ok:
        return False, None, "place card-state ikke synlig etter adressesøk"
    if not src:
        return False, None, "hero-thumbnail mangler i aktivt place card"

    hero = _parse_hero_thumbnail(src)
    return True, hero, ""


def open_default_streetview_from_address(
    page: Page,
    address: str,
    *,
    subsequent_address: bool = False,
) -> tuple[bool, MapsPanoSnapshot | None, str]:
    enc = urllib.parse.quote(address, safe="")
    search_url = f"https://www.google.com/maps/search/{enc}"
    with step_timer(log, "maps_search_goto", url=search_url[:120]):
        page.goto(search_url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT_MS)
    page.wait_for_timeout(1200)
    _dismiss_common_overlays(page)
    base_wait = config.VIEW_WAIT_AFTER_FIRST_ADDR_MS if subsequent_address else config.VIEW_WAIT_MS
    page.wait_for_timeout(min(3200, max(800, int(base_wait * 0.35))))

    clicked = _try_click_street_view_entry(page)
    if not clicked:
        return False, None, "fant ingen klikkbar Street View-kontroll etter adressesøk"

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
            return True, snapshot, ""
        if sv_url and canvas_ok:
            return True, None, ""
        page.wait_for_timeout(int(step_s * 1000))
        elapsed += step_s
    return False, None, "Street View åpnet ikke innen timeout (URL/canvas matcher ikke forventet SV)"


def refresh_pano_snapshot(page: Page) -> MapsPanoSnapshot | None:
    return parse_maps_pano_from_url(page.url)


def open_first_distinct_neighbor_pano(
    page: Page,
    attempts: list[StreetViewAttempt],
    *,
    tgt_lat: float,
    tgt_lon: float,
    quick_settle: bool,
) -> tuple[StreetViewAttempt, float] | None:
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
        return (view, h)
    return None
