"""Sekvensiell hovedløkke: én adresse av gangen."""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from runner import config
from runner.capture import capture_viewport
from runner.decision import evaluate
from runner.maps_streetview import (
    open_default_streetview_from_address,
    open_first_distinct_neighbor_pano,
    parse_maps_pano_from_url,
    refresh_pano_snapshot,
)
from runner.navigator import (
    StreetViewAttempt,
    focus_street_view_for_capture,
    open_streetview_view,
    settle_streetview_after_canvas_ready,
    wait_view_ready,
)
from runner.streetview_candidates import build_lateral_front_attempts
from runner.result_store import ScanApi
from runner.review_integration import push_detection, push_scan_capture
from runner.timing import step_timer
from runner.yolo_detector import DetectorResult, run_yolo

log = logging.getLogger(__name__)


def _ll_equal(a_lat: float, a_lon: float, b_lat: float, b_lon: float, nd: int = 5) -> bool:
    return round(a_lat, nd) == round(b_lat, nd) and round(a_lon, nd) == round(b_lon, nd)


def _log_sv_pano_audit(
    *,
    page,
    location_key: str,
    attempt_idx: int,
    view: StreetViewAttempt,
    heading_applied: float,
    ref_hovedpano: tuple[float, float] | None,
    ref_venstre: tuple[float, float] | None,
    ref_main_panoid: str | None,
    ref_venstre_panoid: str | None,
) -> tuple[float, float, str | None]:
    """
    Logg faktisk pano fra URL etter at view er lastet.
    Returnerer (lat, lon, panoid) brukt i sammenligning og ref-oppdatering.
    """
    parsed = parse_maps_pano_from_url(page.url)
    if parsed is not None:
        plat, plon = parsed.lat, parsed.lon
        heading_show = (
            parsed.heading_deg if parsed.heading_deg is not None else heading_applied
        )
        src = "maps_url"
        panoid = (parsed.panoid or "").strip() or None
    else:
        plat, plon = view.camera_lat, view.camera_lon
        heading_show = heading_applied
        src = "plan_fallback"
        panoid = None

    if ref_main_panoid and panoid:
        same_h = panoid == ref_main_panoid
    else:
        same_h = (
            ref_hovedpano is not None
            and _ll_equal(plat, plon, ref_hovedpano[0], ref_hovedpano[1])
        )

    if ref_venstre_panoid and panoid:
        same_l = panoid == ref_venstre_panoid
    else:
        same_l = (
            ref_venstre is not None
            and _ll_equal(plat, plon, ref_venstre[0], ref_venstre[1])
        )

    panoid_log = panoid or "ukjent"
    log.info(
        "SV_PANO_AUDIT item=%s attempt=%s view_id=%s "
        "pano_lat=%.6f pano_lon=%.6f panoid=%s heading=%.2f | "
        "same_as_hovedpano=%s same_as_venstre_nabo=%s | "
        "plan_lat=%.6f plan_lon=%.6f heading_applied=%.2f | src=%s",
        location_key,
        attempt_idx,
        view.view_id,
        plat,
        plon,
        panoid_log,
        heading_show,
        same_h,
        same_l,
        view.camera_lat,
        view.camera_lon,
        heading_applied,
        src,
    )
    if view.view_id in ("front_left", "front_right") and same_h:
        log.warning(
            "SV_PANO_AUDIT: %s same_as_hovedpano=True (panoid/koord) — ikke egen nabopano",
            view.view_id,
        )
    return plat, plon, panoid


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
    img_cap = max_images_per_address if max_images_per_address is not None else max_attempts
    per_address_iters = min(max_attempts, img_cap)

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
        page = context.new_page()

        for addr_idx, it in enumerate(items):
            addr_t0 = time.perf_counter()
            item_id = it["scan_run_item_id"]
            lid = it["location_id"]
            lat, lon = float(it["latitude"]), float(it["longitude"])
            addr = it["address"]
            log.info("=== Lokasjon %s (%s) ===", lid, addr)
            subsequent_addr = addr_idx > 0

            tgt_lat, tgt_lon = lat, lon
            log.info("Kameraplan målpunkt (JSON): %.6f,%.6f — %s", tgt_lat, tgt_lon, addr)

            lateral_status: dict[str, str] = {}
            views: list[StreetViewAttempt] = []
            left_chain: list[StreetViewAttempt] = []
            right_chain: list[StreetViewAttempt] = []
            want_left = False
            want_right = False
            maps_main_ok = False
            maps_main_reason = ""
            notes_parts: list[str] = []

            with step_timer(log, "maps_open_streetview_total", item_id=lid):
                maps_main_ok, snap0, maps_main_reason = open_default_streetview_from_address(
                    page,
                    addr,
                    subsequent_address=subsequent_addr,
                )

            if not maps_main_ok or snap0 is None:
                log.info(
                    "MAPS_SCAN_SUMMARY address=%r hovedbilde=hoppet (%s) | front_left=hoppet (krever hovedpano) | "
                    "front_right=hoppet (krever hovedpano)",
                    addr,
                    maps_main_reason,
                )
                notes_parts.append(f"standard_front hoppet: {maps_main_reason}")
            else:
                page.wait_for_timeout(500)
                snap = refresh_pano_snapshot(page) or snap0
                main_view = StreetViewAttempt(
                    view_id="standard_front",
                    camera_lat=snap.lat,
                    camera_lon=snap.lon,
                    heading_deg=snap.heading_deg,
                    pitch_deg=0.0,
                    fov_deg=75.0,
                    plan_reason=(
                        "Google Maps første Street View etter søk på JSON-adresse "
                        "(samme som når man søker manuelt og åpner forslaget)"
                    ),
                )
                extra = max(0, min(2, per_address_iters - 1))
                want_left = extra >= 1
                want_right = extra >= 2
                lateral_status, left_chain, right_chain = build_lateral_front_attempts(
                    tgt_lat,
                    tgt_lon,
                    snap.lat,
                    snap.lon,
                    want_left=want_left,
                    want_right=want_right,
                )
                views = [main_view]
                log.info(
                    "MAPS_SCAN_SUMMARY address=%r hovedbilde=Google Street View-forslag brukt (standard_front @ %.6f,%.6f) | "
                    "front_left=%s | front_right=%s",
                    addr,
                    snap.lat,
                    snap.lon,
                    lateral_status.get("front_left", ""),
                    lateral_status.get("front_right", ""),
                )
                log.info(
                    "SV_NEIGHBOR_SUMMARY hovedpano=%.6f,%.6f | venstre=%s | høyre=%s",
                    snap.lat,
                    snap.lon,
                    lateral_status.get("front_left", ""),
                    lateral_status.get("front_right", ""),
                )
                notes_parts.append(
                    f"MAPS: hovedbilde=Google SV-forslag; front_left={lateral_status.get('front_left')}; "
                    f"front_right={lateral_status.get('front_right')}"
                )

            best_bbox: dict | None = None
            best_conf = 0.0
            pushed_hit = False
            ref_hovedpano: tuple[float, float] | None = None
            ref_venstre: tuple[float, float] | None = None
            ref_main_panoid: str | None = None
            ref_venstre_panoid: str | None = None

            def _capture_block(
                attempt_idx: int,
                view: StreetViewAttempt,
                heading_applied: float,
                *,
                update_refs: bool,
            ) -> bool:
                nonlocal best_bbox, best_conf, pushed_hit, ref_hovedpano, ref_venstre
                nonlocal ref_main_panoid, ref_venstre_panoid, notes_parts
                pr_lat, pr_lon, pr_pid = _log_sv_pano_audit(
                    page=page,
                    location_key=str(lid),
                    attempt_idx=attempt_idx,
                    view=view,
                    heading_applied=heading_applied,
                    ref_hovedpano=ref_hovedpano,
                    ref_venstre=ref_venstre,
                    ref_main_panoid=ref_main_panoid,
                    ref_venstre_panoid=ref_venstre_panoid,
                )
                if update_refs:
                    if attempt_idx == 0:
                        ref_hovedpano = (pr_lat, pr_lon)
                        ref_main_panoid = pr_pid
                    elif attempt_idx == 1:
                        ref_venstre = (pr_lat, pr_lon)
                        ref_venstre_panoid = pr_pid
                settle_streetview_after_canvas_ready(
                    page,
                    first_attempt=(attempt_idx == 0),
                    subsequent_address=subsequent_addr and attempt_idx == 0,
                )
                with step_timer(log, "focus_street_view", item_id=lid, attempt=attempt_idx):
                    focus_street_view_for_capture(
                        page,
                        allow_scene_click=(
                            view.view_id not in ("front_left", "front_right")
                        ),
                    )
                shot = tmp_root / f"r{run_id}_i{item_id}_a{attempt_idx}.png"
                pre_cap = (
                    config.CAPTURE_PRE_STABILIZE_MS
                    if attempt_idx == 0
                    else config.CAPTURE_PRE_STABILIZE_MS_NEXT
                )
                with step_timer(log, "capture_viewport_total", item_id=lid, attempt=attempt_idx):
                    capture_viewport(page, shot, pre_stabilize_ms=pre_cap)
                with step_timer(log, "yolo_run_total", item_id=lid, attempt=attempt_idx):
                    det = run_yolo(shot, config.YOLO_MODEL_PATH)
                dec = evaluate(det, best_bbox)

                hd = f"{heading_applied:.1f}"
                h_plan_s = f"{view.heading_deg:.1f}" if view.heading_deg is not None else "na"
                cam_label = (
                    f"{view.view_id}|cam={view.camera_lat:.5f},{view.camera_lon:.5f}"
                    f"|h_applied={hd}|h_plan={h_plan_s}|p={view.pitch_deg}|fov={view.fov_deg}"
                )
                with step_timer(log, "api_log_attempt", item_id=item_id, attempt=attempt_idx):
                    api.log_attempt(
                        run_id,
                        item_id,
                        attempt_idx,
                        screenshot_path=str(shot),
                        camera_state=cam_label,
                        prediction_status="hit" if det.has_detection else "no_hit",
                        confidence=det.confidence_pct,
                        bbox_json=_bbox_payload_for_api(det),
                        rationale=(
                            det.rationale
                            + " | "
                            + dec.reason
                            + f" | view={view.view_id} | plan={view.plan_reason[:200]}"
                        ),
                    )

                route = (
                    "push_detection"
                    if (dec.save_hit and det.bbox_norm_xywh)
                    else "push_scan_capture"
                )
                bbox_for_api = _bbox_payload_for_api(det)
                has_meaningful_bbox = bool(det.all_bboxes_norm)
                log.info(
                    "YOLO etter capture: kjørt på bilde=%s | has_detection=%s | confidence=%.4f (%s%%) | "
                    "bbox_norm_xywh=%s | evaluate: save_hit=%s tier=%s reason=%s | ingest_route=%s | "
                    "bbox_sendes_til_api=%s",
                    shot.name,
                    det.has_detection,
                    det.confidence,
                    det.confidence_pct,
                    det.bbox_norm_xywh,
                    dec.save_hit,
                    dec.tier,
                    dec.reason,
                    route,
                    has_meaningful_bbox,
                )

                if det.has_detection and det.bbox_norm_xywh and det.confidence > best_conf:
                    best_conf = det.confidence
                    best_bbox = det.bbox_norm_xywh

                with step_timer(log, "api_ingest_yolo", item_id=item_id, attempt=attempt_idx, route=route):
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
                            rationale=f"{det.rationale} ({dec.tier}) view={view.view_id}",
                        )
                        pushed_hit = True
                        log.info(
                            "Ingest: %s (treff godkjent i evaluate) item=%s forsøk=%s",
                            route,
                            item_id,
                            attempt_idx,
                        )
                    else:
                        bbox_use = bbox_for_api
                        conf_use = int(det.confidence_pct) if det.has_detection else 0
                        push_scan_capture(
                            api,
                            shot,
                            scan_run_item_id=item_id,
                            location_id=lid,
                            address=addr,
                            postcode=postcode,
                            lat=tgt_lat,
                            lon=tgt_lon,
                            attempt_index=attempt_idx,
                            camera_state=cam_label,
                            bbox=bbox_use,
                            confidence=conf_use,
                            yolo_note=det.rationale,
                            decision_note=f"{dec.reason} ({dec.tier})",
                        )
                        log.info(
                            "Ingest: %s (samme /api/scanner/ingest-yolo; bbox=%s) item=%s forsøk=%s",
                            route,
                            "ja" if has_meaningful_bbox else "nei (0-boks)",
                            item_id,
                            attempt_idx,
                        )

                # Ikke stopp resten av planlagte bilder på adressen bare fordi YOLO/evaluate sier stop_attempts
                # (f.eks. sterkt treff kan være falsk positiv før modellen er trent på skilt).
                return True

            for attempt_idx, view in enumerate(views):
                try:
                    heading_applied = open_streetview_view(
                        page,
                        view,
                        quick_settle=(attempt_idx >= 1),
                        subsequent_address=subsequent_addr and attempt_idx == 0,
                        target_lat=tgt_lat,
                        target_lon=tgt_lon,
                        skip_navigation=(attempt_idx == 0),
                    )
                    if heading_applied is None:
                        notes_parts.append(
                            f"{view.view_id}: ingen heading mot mål — hopper forsøk (sjekk target_lat/lon)"
                        )
                        continue
                    if not wait_view_ready(page, fast=(attempt_idx >= 1)):
                        notes_parts.append(f"{view.view_id}: Street View ikke brukbar")
                        if attempt_idx == 0:
                            notes_parts.append("hopper til neste adresse")
                            break
                        continue
                    _capture_block(
                        attempt_idx,
                        view,
                        heading_applied,
                        update_refs=(attempt_idx in (0, 1)),
                    )
                except Exception as e:
                    log.exception("Feil attempt %s: %s", attempt_idx, e)
                    notes_parts.append(f"{view.view_id}: {e!s}")

            if (
                maps_main_ok
                and ref_hovedpano is not None
                and want_left
                and left_chain
            ):
                attempt_idx = 1
                try:
                    res_l = open_first_distinct_neighbor_pano(
                        page,
                        left_chain,
                        tgt_lat=tgt_lat,
                        tgt_lon=tgt_lon,
                        quick_settle=True,
                    )
                    if res_l is None:
                        lateral_status["front_left"] = (
                            "hoppet: ingen vellykket nabosteg i kjeden (åpning/canvas/URL-parse)"
                        )
                        notes_parts.append(
                            "front_left hoppet: ingen vellykket nabosteg — ikke lagret"
                        )
                    else:
                        view_l, h_l = res_l
                        did_l = _capture_block(
                            attempt_idx,
                            view_l,
                            h_l,
                            update_refs=True,
                        )
                        if did_l:
                            lateral_status["front_left"] = "front_left: lagret etter planlagt nabosteg"
                        else:
                            lateral_status["front_left"] = (
                                "hoppet: capture fullførte ikke (uvanlig)"
                            )
                except Exception as e:
                    log.exception("Feil front_left nabopano: %s", e)
                    notes_parts.append(f"front_left: {e!s}")

            if (
                maps_main_ok
                and ref_hovedpano is not None
                and want_right
                and right_chain
            ):
                attempt_idx = 2
                try:
                    res_r = open_first_distinct_neighbor_pano(
                        page,
                        right_chain,
                        tgt_lat=tgt_lat,
                        tgt_lon=tgt_lon,
                        quick_settle=True,
                    )
                    if res_r is None:
                        lateral_status["front_right"] = (
                            "hoppet: ingen vellykket nabosteg i kjeden (åpning/canvas/URL-parse)"
                        )
                        notes_parts.append(
                            "front_right hoppet: ingen vellykket nabosteg — ikke lagret"
                        )
                    else:
                        view_r, h_r = res_r
                        did_r = _capture_block(
                            attempt_idx,
                            view_r,
                            h_r,
                            update_refs=False,
                        )
                        if did_r:
                            lateral_status["front_right"] = "front_right: lagret etter planlagt nabosteg"
                        else:
                            lateral_status["front_right"] = (
                                "hoppet: capture fullførte ikke (uvanlig)"
                            )
                except Exception as e:
                    log.exception("Feil front_right nabopano: %s", e)
                    notes_parts.append(f"front_right: {e!s}")

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

        browser.close()

    log.info("ScanRun %s ferdig wall_clock_s=%.3fs", run_id, time.perf_counter() - scan_t0)
