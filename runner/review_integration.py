"""Tynn wrapper: push YOLO-treff inn i appens review-kø via result_store."""

from __future__ import annotations

from pathlib import Path

from runner.result_store import ScanApi


def push_detection(
    api: ScanApi,
    screenshot: Path,
    *,
    scan_run_item_id: int,
    location_id: int,
    address: str,
    postcode: str,
    lat: float,
    lon: float,
    confidence: int,
    bbox: dict[str, float],
    rationale: str,
) -> dict:
    return api.ingest_yolo(
        screenshot,
        scan_run_item_id=scan_run_item_id,
        location_id=location_id,
        address=address,
        postcode=postcode,
        latitude=lat,
        longitude=lon,
        confidence=confidence,
        bbox=bbox,
        rationale=rationale,
    )
