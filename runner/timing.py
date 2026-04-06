"""Enkel perf_counter-basert steg-timing for scanner-flyten."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

log = logging.getLogger(__name__)


@contextmanager
def step_timer(logger: logging.Logger, step: str, **ctx: object) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        extra = " ".join(f"{k}={v}" for k, v in ctx.items() if v is not None)
        logger.info("SCAN_STEP %s %.3fs%s", step, dt, f" {extra}" if extra else "")
