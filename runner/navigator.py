"""Google Street View i Playwright — deterministisk kamerasekvens."""

from __future__ import annotations

import logging
from typing import Literal

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeout

from runner import config

log = logging.getLogger(__name__)

CameraPreset = Literal["first_view", "slight_left", "slight_right", "slight_zoom"]

PRESET_ORDER: list[CameraPreset] = ["first_view", "slight_left", "slight_right", "slight_zoom"]


def open_streetview_near(page: Page, lat: float, lon: float) -> None:
    """Åpne Street View nær koordinat (cbll)."""
    url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
    log.info("Navigerer til Street View: %s", url)
    page.goto(url, wait_until="domcontentloaded", timeout=config.PAGE_TIMEOUT_MS)
    page.wait_for_timeout(config.VIEW_WAIT_MS)


def apply_camera_preset(page: Page, preset: CameraPreset) -> None:
    """Enkle tastetrykk — fungerer når Street View har fokus."""
    page.wait_for_timeout(300)
    body = page.locator("body")
    body.click(timeout=5000)
    page.wait_for_timeout(config.STABILIZE_MS)

    if preset == "first_view":
        return
    if preset == "slight_left":
        page.keyboard.press("ArrowLeft")
        page.wait_for_timeout(400)
        page.keyboard.press("ArrowLeft")
    elif preset == "slight_right":
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(400)
        page.keyboard.press("ArrowRight")
    elif preset == "slight_zoom":
        page.keyboard.press("+")
        page.wait_for_timeout(300)
    page.wait_for_timeout(config.STABILIZE_MS)


def wait_view_ready(page: Page, retries: int = 3) -> bool:
    for i in range(retries):
        try:
            canvas = page.locator("canvas").first
            if canvas.count() and canvas.is_visible(timeout=5000):
                return True
        except PlaywrightTimeout:
            pass
        log.warning("Vent på canvas, retry %s/%s", i + 1, retries)
        page.wait_for_timeout(2000)
    return page.locator("canvas").count() > 0
