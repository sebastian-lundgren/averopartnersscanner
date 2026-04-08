"""Tynn wrapper: push YOLO-treff eller rå scan-bilder inn i appens review-kø via result_store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.result_store import ScanApi


def push_scan_capture(
    api: ScanApi,
    screenshot: Path,
    *,
    scan_run_item_id: int,
    location_id: int,
    address: str,
    postcode: str,
    lat: float,
    lon: float,
    attempt_index: int,
    camera_state: str,
    bbox: dict[str, Any],
    confidence: int,
    yolo_note: str,
    decision_note: str,
) -> dict:
    """Lagre Street View-ramme uten godkjent deteksjon — samme API som treff, annerledes rationale."""
    rationale = (
        f"[Street View scan — bilde lagret | forsøk #{attempt_index} | {camera_state}] "
        f"YOLO: {yolo_note} | beslutning: {decision_note}"
    )
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
    bbox: dict[str, Any],
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
