"""
Isolert prototype: ett museklikk på heuristisk «veipil»-posisjon (nedre senter av største SV-canvas).
Ingen tastatur. Ikke en del av hovedløkken.
Kjør: python -m runner.prototype_road_arrow_click 'https://www.google.com/maps/...'
eller: STREETVIEW_TEST_URL=... python -m runner.prototype_road_arrow_click
"""

from __future__ import annotations

import logging
import os
import sys

from playwright.sync_api import sync_playwright

from runner import config
from runner.maps_streetview import _dismiss_common_overlays, parse_maps_pano_from_url
from runner.navigator import wait_view_ready

log = logging.getLogger("prototype_road_arrow")


def _largest_visible_canvas_box(page):
    locs = page.locator("canvas")
    n = locs.count()
    best = None
    best_area = 0.0
    for i in range(n):
        L = locs.nth(i)
        try:
            if not L.is_visible(timeout=1200):
                continue
            box = L.bounding_box()
            if not box or box["width"] < 220 or box["height"] < 140:
                continue
            area = box["width"] * box["height"]
            if area > best_area:
                best_area = area
                best = box
        except Exception:
            continue
    return best


def _heading_str(s):
    if s is None or s.heading_deg is None:
        return "ukjent"
    return f"{s.heading_deg:.2f}"


def _pano_changed(before, after) -> str:
    if before is None or after is None:
        return "ukjent"
    pid_b = (before.panoid or "").strip()
    pid_a = (after.panoid or "").strip()
    if pid_b and pid_a and pid_b != pid_a:
        return "ja"
    if abs(before.lat - after.lat) > 1e-5 or abs(before.lon - after.lon) > 1e-5:
        return "ja"
    return "nei"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    url = (sys.argv[1] if len(sys.argv) > 1 else "").strip() or os.environ.get(
        "STREETVIEW_TEST_URL", ""
    ).strip()
    if not url:
        log.error("Mangler URL: argv eller STREETVIEW_TEST_URL")
        return 2

    # Heuristikk: pilen ligger typisk nær bunn midt på pano-canvas (ikke minimap).
    frac_x, frac_y = 0.5, 0.82

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=config.HEADLESS)
        page = browser.new_page(
            viewport={"width": 1280, "height": 720},
            device_scale_factor=1,
        )
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT_MS)
            page.wait_for_timeout(800)
            _dismiss_common_overlays(page)
            if not wait_view_ready(page, retries=4, fast=False):
                log.warning("Canvas ikke bekreftet synlig — fortsetter likevel")

            box = _largest_visible_canvas_box(page)
            if not box:
                log.error("SV_ARROW_PROTO ingen passende canvas")
                return 1

            cx = box["x"] + box["width"] * frac_x
            cy = box["y"] + box["height"] * frac_y
            before = parse_maps_pano_from_url(page.url)
            log.info(
                "SV_ARROW_PROTO før_klikk klikk_frac=(%.2f,%.2f) cx=%.1f cy=%.1f canvas=%.0fx%.0f",
                frac_x,
                frac_y,
                cx,
                cy,
                box["width"],
                box["height"],
            )
            log.info(
                "SV_ARROW_PROTO før_klikk panoid=%s pano_lat=%.6f pano_lon=%.6f heading=%s",
                (before.panoid or "ukjent") if before else "ukjent",
                before.lat if before else float("nan"),
                before.lon if before else float("nan"),
                _heading_str(before),
            )

            page.mouse.click(cx, cy)
            page.wait_for_timeout(2800)

            after = parse_maps_pano_from_url(page.url)
            changed = _pano_changed(before, after)
            log.info(
                "SV_ARROW_PROTO etter_klikk panoid=%s pano_lat=%.6f pano_lon=%.6f heading=%s",
                (after.panoid or "ukjent") if after else "ukjent",
                after.lat if after else float("nan"),
                after.lon if after else float("nan"),
                _heading_str(after),
            )
            log.info("SV_ARROW_PROTO pano_endret=%s", changed)
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
