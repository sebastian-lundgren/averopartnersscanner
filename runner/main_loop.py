"""Sekvensiell hovedløkke: én adresse av gangen."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from runner import config
from runner.capture import capture_viewport
from runner.decision import evaluate
from runner.navigator import PRESET_ORDER, apply_camera_preset, open_streetview_near, wait_view_ready
from runner.result_store import ScanApi
from runner.review_integration import push_detection
from runner.yolo_detector import run_yolo

log = logging.getLogger(__name__)


def load_locations_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "locations" in data:
        return list(data["locations"])
    raise ValueError("Forventet liste eller {locations: [...]}")


def run_scan(
    *,
    locations_file: Path,
    postcode: str,
    max_addresses: int,
    max_attempts: int,
    api: ScanApi | None = None,
) -> None:
    api = api or ScanApi()
    locs = load_locations_json(locations_file)
    locs = [x for x in locs if str(x.get("postcode", "")) == postcode][:max_addresses]
    if not locs:
        log.error("Ingen lokasjoner for postkode %s i %s", postcode, locations_file)
        return

    api.bulk_locations(
        [
            {
                "address": x["address"],
                "postcode": str(x["postcode"]),
                "latitude": float(x["latitude"]),
                "longitude": float(x["longitude"]),
            }
            for x in locs
        ]
    )
    run_id, items = api.start_run(postcode, max_locations=len(locs))
    log.info("ScanRun %s med %s stopp", run_id, len(items))

    tmp_root = Path(tempfile.mkdtemp(prefix="sv_scan_"))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=config.HEADLESS)
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        page = context.new_page()

        for it in items:
            item_id = it["scan_run_item_id"]
            lid = it["location_id"]
            lat, lon = float(it["latitude"]), float(it["longitude"])
            addr = it["address"]
            log.info("=== Lokasjon %s (%s) ===", lid, addr)

            best_bbox: dict | None = None
            best_conf = 0.0
            pushed_hit = False
            notes_parts: list[str] = []

            for attempt_idx in range(min(max_attempts, len(PRESET_ORDER))):
                preset = PRESET_ORDER[attempt_idx]
                try:
                    open_streetview_near(page, lat, lon)
                    if not wait_view_ready(page):
                        notes_parts.append(f"{preset}: view ikke klar")
                        continue
                    apply_camera_preset(page, preset)
                    shot = tmp_root / f"r{run_id}_i{item_id}_a{attempt_idx}.jpg"
                    capture_viewport(page, shot)
                    det = run_yolo(shot, config.YOLO_MODEL_PATH)
                    dec = evaluate(det, best_bbox)

                    api.log_attempt(
                        run_id,
                        item_id,
                        attempt_idx,
                        screenshot_path=str(shot),
                        camera_state=preset,
                        prediction_status="hit" if det.has_detection else "no_hit",
                        confidence=det.confidence_pct,
                        bbox_json=det.bbox_norm_xywh,
                        rationale=det.rationale + " | " + dec.reason,
                    )

                    if det.has_detection and det.bbox_norm_xywh and det.confidence > best_conf:
                        best_conf = det.confidence
                        best_bbox = det.bbox_norm_xywh

                    if dec.save_hit and det.bbox_norm_xywh:
                        push_detection(
                            api,
                            shot,
                            scan_run_item_id=item_id,
                            location_id=lid,
                            address=addr,
                            postcode=postcode,
                            lat=lat,
                            lon=lon,
                            confidence=det.confidence_pct,
                            bbox=det.bbox_norm_xywh,
                            rationale=f"{det.rationale} ({dec.tier}) preset={preset}",
                        )
                        pushed_hit = True
                        log.info("Lagret treff via API for item %s", item_id)

                    if dec.stop_attempts:
                        break
                except Exception as e:
                    log.exception("Feil attempt %s: %s", attempt_idx, e)
                    notes_parts.append(f"{preset}: {e!s}")

            api.complete_item(
                run_id,
                item_id,
                "detection_found" if pushed_hit else "no_hit",
                best_confidence=best_conf if best_conf else None,
                notes="; ".join(notes_parts) or None,
            )

        browser.close()

    log.info("ScanRun %s ferdig", run_id)
