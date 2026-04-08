"""Sekvensiell hovedløkke: én adresse av gangen."""

from __future__ import annotations

import json
import logging
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

from runner import config
from runner.decision import evaluate
from runner.maps_streetview import open_placecard_hero_thumbnail
from runner.result_store import ScanApi
from runner.review_integration import push_detection, push_scan_capture
from runner.timing import step_timer
from runner.yolo_detector import DetectorResult, run_yolo

log = logging.getLogger(__name__)


def _bbox_payload_for_api(det: DetectorResult) -> dict:
    """Backend ingest forventer JSON med flere bokser eller legacy null-boks."""
    if det.all_bboxes_norm:
        return {
            "boxes": det.all_bboxes_norm,
            "v": 2,
            "yolo_trusted_primary": det.yolo_trusted_primary,
            "yolo_primary_gate_reason": det.yolo_primary_gate_reason,
        }
    return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}


def load_locations_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "locations" in data:
        return list(data["locations"])
    raise ValueError("Forventet liste eller {locations: [...]}")


def _postcode_match(row_pc: object, want: str) -> bool:
    a = str(row_pc or "").strip().replace(" ", "")
    b = str(want or "").strip().replace(" ", "")
    return a == b


def _download_url_image(url: str, out_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def run_scan(
    *,
    locations_file: Path,
    postcode: str,
    max_addresses: int,
    max_attempts: int,
    max_images_per_address: int | None = None,
    api: ScanApi | None = None,
) -> None:
    api = api or ScanApi()
    locs = load_locations_json(locations_file)
    locs = [x for x in locs if _postcode_match(x.get("postcode", ""), postcode)][:max_addresses]
    if not locs:
        log.error("Ingen lokasjoner for postkode %s i %s", postcode, locations_file)
        return

    bulk: list[dict] = []
    for x in locs:
        bulk.append(
            {
                "address": x["address"],
                "postcode": str(x["postcode"]),
                "latitude": float(x["latitude"]),
                "longitude": float(x["longitude"]),
            }
        )
    with step_timer(log, "api_bulk_locations_and_start_run", locations=len(bulk)):
        bulk_out = api.bulk_locations(bulk)
        ids_block = bulk_out.get("ids") if isinstance(bulk_out, dict) else None
        loc_ids: list[int] | None = None
        if isinstance(ids_block, list) and len(ids_block) == len(bulk):
            try:
                loc_ids = [int(x["id"]) for x in ids_block]
            except (KeyError, TypeError, ValueError):
                loc_ids = None
        if loc_ids is None:
            log.warning(
                "bulk_locations ga ikke forventet ids-liste — start_run uten location_ids (kan avvike fra JSON-rekkefølge)"
            )
        run_id, items = api.start_run(
            postcode,
            max_locations=len(locs),
            location_ids=loc_ids,
        )
    log.info("ScanRun %s med %s stopp", run_id, len(items))

    tmp_root = Path(tempfile.mkdtemp(prefix="sv_scan_"))
    scan_t0 = time.perf_counter()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=config.HEADLESS)
        context = browser.new_context(
            viewport={
                "width": config.CAPTURE_VIEWPORT_WIDTH,
                "height": config.CAPTURE_VIEWPORT_HEIGHT,
            },
            device_scale_factor=config.CAPTURE_DEVICE_SCALE_FACTOR,
        )
        log.info(
            "Playwright context: viewport=%dx%d device_scale_factor=%s",
            config.CAPTURE_VIEWPORT_WIDTH,
            config.CAPTURE_VIEWPORT_HEIGHT,
            config.CAPTURE_DEVICE_SCALE_FACTOR,
        )
        for addr_idx, it in enumerate(items):
            addr_t0 = time.perf_counter()
            page = context.new_page()
            item_id = it["scan_run_item_id"]
            lid = it["location_id"]
            lat, lon = float(it["latitude"]), float(it["longitude"])
            addr = it["address"]
            log.info("=== Lokasjon %s (%s) ===", lid, addr)
            subsequent_addr = addr_idx > 0

            tgt_lat, tgt_lon = lat, lon
            log.info("Kameraplan målpunkt (JSON): %.6f,%.6f — %s", tgt_lat, tgt_lon, addr)

            maps_main_ok = False
            maps_main_reason = ""
            notes_parts: list[str] = []
            pushed_hit = False
            best_conf = 0.0
            best_bbox: dict | None = None

            with step_timer(log, "maps_open_hero_thumbnail_total", item_id=lid):
                maps_main_ok, hero_thumb, maps_main_reason = open_placecard_hero_thumbnail(
                    page,
                    addr,
                    subsequent_address=subsequent_addr,
                )

            if not maps_main_ok:
                log.info("MAPS_SCAN_SUMMARY address=%r hovedbilde=hoppet (%s)", addr, maps_main_reason)
                notes_parts.append(f"standard_front hoppet: {maps_main_reason}")
            else:
                shot = tmp_root / f"r{run_id}_i{item_id}_a0.png"
                with step_timer(log, "maps_fetch_hero_thumbnail", item_id=lid):
                    _download_url_image(hero_thumb.fetch_url, shot)
                cam_label = (
                    f"standard_front|cam={tgt_lat:.5f},{tgt_lon:.5f}"
                    f"|source=maps_hero_thumbnail_direct|panoid={hero_thumb.panoid or 'na'}"
                    f"|yaw={hero_thumb.yaw}|pitch={hero_thumb.pitch}|thumbfov={hero_thumb.thumbfov}"
                )
                notes_parts.append("MAPS: hovedbilde=maps_hero_thumbnail_direct")
                with step_timer(log, "yolo_run_total", item_id=lid, attempt=0):
                    det = run_yolo(shot, config.YOLO_MODEL_PATH)
                dec = evaluate(det, None)
                bbox_for_api = _bbox_payload_for_api(det)
                with step_timer(log, "api_log_attempt", item_id=item_id, attempt=0):
                    api.log_attempt(
                        run_id,
                        item_id,
                        0,
                        screenshot_path=str(shot),
                        camera_state=cam_label,
                        prediction_status="hit" if det.has_detection else "no_hit",
                        confidence=det.confidence_pct,
                        bbox_json=bbox_for_api,
                        rationale=f"{det.rationale} | {dec.reason} | source=maps_hero_thumbnail_direct",
                    )
                with step_timer(log, "api_ingest_yolo", item_id=item_id, attempt=0):
                    if dec.save_hit and det.bbox_norm_xywh:
                        push_detection(
                            api,
                            shot,
                            scan_run_item_id=item_id,
                            location_id=lid,
                            address=addr,
                            postcode=postcode,
                            lat=tgt_lat,
                            lon=tgt_lon,
                            confidence=det.confidence_pct,
                            bbox=bbox_for_api,
                            rationale=f"{det.rationale} ({dec.tier}) view=standard_front",
                        )
                        pushed_hit = True
                        best_conf = det.confidence
                        best_bbox = det.bbox_norm_xywh
                    else:
                        push_scan_capture(
                            api,
                            shot,
                            scan_run_item_id=item_id,
                            location_id=lid,
                            address=addr,
                            postcode=postcode,
                            lat=tgt_lat,
                            lon=tgt_lon,
                            attempt_index=0,
                            camera_state=cam_label,
                            bbox=bbox_for_api,
                            confidence=int(det.confidence_pct) if det.has_detection else 0,
                            yolo_note=det.rationale,
                            decision_note=f"{dec.reason} ({dec.tier}) | source=maps_hero_thumbnail_direct",
                        )

            with step_timer(log, "api_complete_item", item_id=item_id):
                api.complete_item(
                    run_id,
                    item_id,
                    "detection_found" if pushed_hit else "no_hit",
                    best_confidence=best_conf if best_conf else None,
                    notes="; ".join(notes_parts) or None,
                )
            log.info(
                "SCAN_ADDR_SUMMARY item_id=%s total_s=%.3fs pushed_hit=%s",
                lid,
                time.perf_counter() - addr_t0,
                pushed_hit,
            )
            page.close()

        browser.close()

    log.info("ScanRun %s ferdig wall_clock_s=%.3fs", run_id, time.perf_counter() - scan_t0)
