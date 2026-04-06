"""Screenshot av viewport (Street View) — høy oppløsning, tapsfri PNG som standard."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page

from runner import config
from runner.timing import step_timer

log = logging.getLogger(__name__)


def _laplacian_mean_sq(gray: Image.Image) -> float:
    """Lav verdi ≈ lite kantdetalj i ROI (trygg, defensiv indikator — ingen ML)."""

    w, h = gray.size
    if w < 5 or h < 5:
        return 1e9
    max_side = 200
    if max(w, h) > max_side:
        r = max_side / float(max(w, h))
        nw, nh = max(5, int(w * r)), max(5, int(h * r))
        gray = gray.resize((nw, nh), Image.Resampling.BILINEAR)
        w, h = nw, nh
    px = gray.load()
    acc = 0.0
    n = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            c = px[x, y]
            lap = 4 * c - px[x - 1, y] - px[x + 1, y] - px[x, y - 1] - px[x, y + 1]
            acc += lap * lap
            n += 1
    return acc / max(1, n)


def _pre_capture_zoom_if_facade_seems_small(page: Page) -> None:
    """
    Rett før endelig screenshot: 0–2 små musehjul-inn på Street View-canvas,
    kun når en konservativ visuell probe tyder på mye himmel + lite struktur i fasadeområdet.
    Ingen page.goto, ingen sidebytte — ved tvil gjøres ingenting.
    """
    try:
        probe = page.screenshot(full_page=False, type="jpeg", quality=48)
        im = Image.open(BytesIO(probe)).convert("L")
    except Exception as e:
        log.debug("SV_CAPTURE pre_zoom probe skipped: %s", e)
        return

    w, h = im.size
    if w < 120 or h < 120:
        return

    top_h = max(1, int(h * 0.22))
    top = im.crop((0, 0, w, top_h))
    td = list(top.getdata())
    top_mean = sum(td) / max(1, len(td))

    x0, y0 = int(w * 0.20), int(h * 0.26)
    x1, y1 = int(w * 0.80), int(h * 0.90)
    if x1 <= x0 + 8 or y1 <= y0 + 8:
        return
    facade = im.crop((x0, y0, x1, y1))
    lap = _laplacian_mean_sq(facade)

    steps = 0
    if top_mean >= 200.0 and lap < 95.0:
        steps = 2
    elif top_mean >= 196.0 and lap < 132.0:
        steps = 1

    if steps <= 0:
        return

    log.info(
        "SV_CAPTURE view_milestone=pre_zoom_probe top_mean=%.1f lap_msq=%.1f steps=%s",
        top_mean,
        lap,
        steps,
    )

    canvas = page.locator("canvas").first
    try:
        if canvas.count() == 0:
            return
        canvas.focus(timeout=2000)
    except Exception:
        log.debug("SV_CAPTURE pre_zoom canvas.focus failed", exc_info=True)
        return

    try:
        box = canvas.bounding_box()
    except Exception:
        return
    if not box or box.get("width", 0) < 8 or box.get("height", 0) < 8:
        return

    cx = box["x"] + box["width"] * 0.5
    cy = box["y"] + box["height"] * 0.5
    try:
        page.mouse.move(cx, cy)
    except Exception:
        return

    for i in range(steps):
        try:
            page.mouse.wheel(0, -85)
        except Exception:
            log.debug("SV_CAPTURE pre_zoom wheel step %s failed", i, exc_info=True)
            break
        page.wait_for_timeout(340)

    page.wait_for_timeout(420)


def capture_viewport(page: Page, dest: Path, *, pre_stabilize_ms: int | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pre = (
        int(pre_stabilize_ms)
        if pre_stabilize_ms is not None
        else int(config.CAPTURE_PRE_STABILIZE_MS)
    )
    pre = max(0, pre)
    log.info(
        "SV_CAPTURE view_milestone=screenshot_imminent pre_stabilize_ms=%s file=%s",
        pre,
        dest.name,
    )
    if pre:
        page.wait_for_timeout(pre)
    _pre_capture_zoom_if_facade_seems_small(page)
    log.info("SV_CAPTURE view_milestone=screenshot_shutter file=%s", dest.name)
    suffix = dest.suffix.lower()
    with step_timer(log, "playwright_screenshot", dest=dest.name):
        if suffix == ".png":
            page.screenshot(path=str(dest), full_page=False, type="png")
            kind = "PNG tapsfri"
        elif suffix in (".jpg", ".jpeg"):
            q = max(1, min(100, int(config.CAPTURE_JPEG_QUALITY)))
            page.screenshot(path=str(dest), full_page=False, type="jpeg", quality=q)
            kind = f"JPEG kvalitet={q}"
        else:
            page.screenshot(path=str(dest), full_page=False)
            kind = "standard"

    dpr = config.CAPTURE_DEVICE_SCALE_FACTOR
    vw, vh = config.CAPTURE_VIEWPORT_WIDTH, config.CAPTURE_VIEWPORT_HEIGHT
    approx_px_w = int(vw * dpr)
    approx_px_h = int(vh * dpr)
    log.info(
        "Capture: %s | viewport_css=%dx%d device_scale_factor=%s (~%dx%d px) -> %s",
        kind,
        vw,
        vh,
        dpr,
        approx_px_w,
        approx_px_h,
        dest,
    )
    return dest
