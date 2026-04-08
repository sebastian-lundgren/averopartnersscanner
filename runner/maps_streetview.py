"""Maps place-card hero-thumbnail capture."""

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout

from runner import config
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
    for sel in ("h1", '[role="main"] h1'):
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible(timeout=2200):
                place_card_ok = True
                break
        except Exception:
            continue
    if not place_card_ok:
        return False, None, "place card-state ikke synlig etter adressesøk"

    sel_primary = 'button[jsaction="pane.wfvdle7.heroHeaderImage"] img[src*="streetviewpixels-pa.googleapis.com/v1/thumbnail"]'
    sel_fallback = 'img[src*="streetviewpixels-pa.googleapis.com/v1/thumbnail"][src*="panoid="]'
    src = ""
    for sel in (sel_primary, sel_fallback):
        try:
            img = page.locator(sel).first
            if img.count() and img.is_visible(timeout=2200):
                src = (img.get_attribute("src") or "").strip()
                if src:
                    break
        except Exception:
            continue
    if not src:
        return False, None, "hero-thumbnail mangler i aktivt place card"

    hero = _parse_hero_thumbnail(src)
    return True, hero, ""
