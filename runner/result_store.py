"""HTTP-klient mot backend scanner-API."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from runner import config

log = logging.getLogger(__name__)


class ScanApi:
    def __init__(self) -> None:
        self.base = config.API_BASE
        self.headers = {}
        if config.SCANNER_TOKEN:
            self.headers["X-Scanner-Token"] = config.SCANNER_TOKEN

    def bulk_locations(self, locations: list[dict]) -> dict:
        r = httpx.post(
            f"{self.base}/api/scanner/locations/bulk",
            json={"locations": locations},
            headers=self.headers,
            timeout=120.0,
        )
        r.raise_for_status()
        d = r.json()
        log.info("Lagret %s lokasjoner", len(locations))
        return d

    def start_run(
        self,
        postcode: str,
        max_locations: int,
        *,
        location_ids: list[int] | None = None,
    ) -> tuple[int, list[dict]]:
        body: dict = {"postcode": postcode, "max_locations": max_locations}
        if location_ids is not None:
            body["location_ids"] = location_ids
        r = httpx.post(
            f"{self.base}/api/scanner/runs/start",
            json=body,
            headers=self.headers,
            timeout=120.0,
        )
        r.raise_for_status()
        d = r.json()
        return int(d["scan_run_id"]), list(d["items"])

    def log_attempt(
        self,
        run_id: int,
        item_id: int,
        attempt_index: int,
        *,
        screenshot_path: str | None,
        camera_state: str,
        prediction_status: str | None,
        confidence: int | None,
        bbox_json: dict | list | None,
        rationale: str | None,
    ) -> None:
        body = {
            "attempt_index": attempt_index,
            "screenshot_path": screenshot_path,
            "camera_state": camera_state,
            "prediction_status": prediction_status,
            "confidence": confidence,
            "bbox_json": bbox_json,
            "rationale": rationale,
        }
        r = httpx.post(
            f"{self.base}/api/scanner/runs/{run_id}/items/{item_id}/attempt",
            json=body,
            headers=self.headers,
            timeout=60.0,
        )
        r.raise_for_status()

    def complete_item(
        self,
        run_id: int,
        item_id: int,
        final_result: str,
        best_confidence: float | None = None,
        notes: str | None = None,
    ) -> None:
        r = httpx.post(
            f"{self.base}/api/scanner/runs/{run_id}/items/{item_id}/complete",
            json={
                "final_result": final_result,
                "best_confidence": best_confidence,
                "notes": notes,
            },
            headers=self.headers,
            timeout=60.0,
        )
        r.raise_for_status()

    def ingest_yolo(
        self,
        screenshot: Path,
        *,
        scan_run_item_id: int,
        location_id: int,
        address: str,
        postcode: str,
        latitude: float,
        longitude: float,
        confidence: int,
        bbox: dict,
        rationale: str,
    ) -> dict:
        raw = screenshot.read_bytes()
        mime = "image/png" if screenshot.suffix.lower() == ".png" else "image/jpeg"
        files = {"file": (screenshot.name, raw, mime)}
        data = {
            "scan_run_item_id": str(scan_run_item_id),
            "location_id": str(location_id),
            "address_line": address,
            "postcode": postcode,
            "latitude": str(latitude),
            "longitude": str(longitude),
            "confidence": str(confidence),
            "bbox_json": json.dumps(bbox),
            "rationale": rationale,
            "predicted_status": "uklart",
        }
        r = httpx.post(
            f"{self.base}/api/scanner/ingest-yolo",
            data=data,
            files=files,
            headers=self.headers,
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json()
