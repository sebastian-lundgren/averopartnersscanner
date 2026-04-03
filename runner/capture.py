"""Screenshot av viewport (Street View)."""

from __future__ import annotations

import logging
from pathlib import Path

from playwright.sync_api import Page

from runner import config

log = logging.getLogger(__name__)


def capture_viewport(page: Page, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(config.STABILIZE_MS)
    page.screenshot(path=str(dest), full_page=False)
    log.info("Screenshot: %s", dest)
    return dest
