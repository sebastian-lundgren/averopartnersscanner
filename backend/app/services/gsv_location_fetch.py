"""
Hent strukturerte adresser for norsk postnummer via OpenStreetMap Overpass API.

Kun objekter med addr:street + addr:housenumber + addr:postcode og gyldige koordinater i
Norge-boksen — deduplisert, sortert alfabetisk på gate og stigende på husnummer (inkl. suffiks).

Overpass: https://wiki.openstreetmap.org/wiki/Overpass_API — ikke massiv polling.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from app.config import settings

log = logging.getLogger(__name__)

# Offentlige instanser (roter ved 502/503/504/429). kumi er ofte mindre belastet enn overpass-api.de.
OVERPASS_ENDPOINTS: tuple[str, ...] = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
# Omtrentlig Norge (sør, vest, nord, øst) — samme som Overpass-filter
NO_BBOX = "57.9,4.5,71.3,31.3"
NO_LAT_MIN, NO_LON_MIN, NO_LAT_MAX, NO_LON_MAX = 57.9, 4.5, 71.3, 31.3
USER_AGENT = "AlarmskiltQC-StreetViewScanner/1.0 (overpass; internal)"
# Kortere pause mellom POSTs — fortsatt «snill» mot Overpass; sparer mye wall-clock ved retry.
REQUEST_PAUSE_S = float(os.environ.get("OVERPASS_REQUEST_PAUSE_S", "0.35"))
# Må være > Overpass [timeout:N] i spørringen slik at vi får JSON med remark ved server-timeout.
HTTP_TIMEOUT_S = int(os.environ.get("OVERPASS_HTTP_TIMEOUT_S", "140"))
RETRY_PER_ENDPOINT = int(os.environ.get("OVERPASS_RETRY_PER_ENDPOINT", "2"))
RETRY_BACKOFF_S = float(os.environ.get("OVERPASS_RETRY_BACKOFF_S", "1.35"))
# Gjenbruk rå JSON per postnummer (samme prosess / flere jobber innen TTL).
OVERPASS_CACHE_MAX_AGE_S = int(os.environ.get("OVERPASS_CACHE_MAX_AGE_S", str(6 * 3600)))
OVERPASS_CACHE_SUBDIR = "gsv_overpass_cache"


def _should_retry_overpass(error_msg: str) -> bool:
    m = error_msg.lower()
    return (
        any(f"http {c}" in m for c in ("429", "500", "502", "503", "504"))
        or "timeout" in m
        or "timed out" in m
        or "nettverksfeil" in m
    )


class OverpassFetchError(RuntimeError):
    """Overpass utilgjengelig, timeout i spørring, eller ugyldig svar."""


def _normalize_street(s: str) -> str:
    t = unicodedata.normalize("NFKC", (s or "").strip())
    t = re.sub(r"\s+", " ", t)
    return t


def _normalize_hn_key(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").strip()).casefold()


def _house_sort_key(housenumber: str) -> tuple:
    """Stigende: heltall først, deretter suffiks (12, 12a, 12b, 12-14 som tekst)."""
    raw = (housenumber or "").strip()
    s = unicodedata.normalize("NFKC", raw).upper().replace(" ", "")
    m = re.match(r"^(\d+)(.*)$", s)
    if m:
        try:
            return (int(m.group(1)), m.group(2).lower())
        except ValueError:
            pass
    return (10**9, raw.casefold())


def _in_norway_bbox(lat: float, lon: float) -> bool:
    return NO_LAT_MIN <= lat <= NO_LAT_MAX and NO_LON_MIN <= lon <= NO_LON_MAX


def _element_priority(el_type: str) -> int:
    # Ved duplikat: foretrekk node (ofte inngangspunkt) fremfor bygningsentroid.
    return {"node": 3, "way": 2, "relation": 1}.get(el_type, 0)


def _coords_from_element(el: dict) -> tuple[float, float] | None:
    t = el.get("type")
    lat: float | None
    lon: float | None
    if t == "node":
        try:
            lat = float(el["lat"])
            lon = float(el["lon"])
        except (KeyError, TypeError, ValueError):
            return None
    else:
        c = el.get("center")
        if not isinstance(c, dict):
            return None
        try:
            lat = float(c["lat"])
            lon = float(c["lon"])
        except (KeyError, TypeError, ValueError):
            return None
    if not _in_norway_bbox(lat, lon):
        return None
    return lat, lon


def _overpass_query(pc: str, *, include_relation: bool, timeout_s: int) -> str:
    rel = ""
    if include_relation:
        rel = f'  relation["addr:postcode"="{pc}"]["addr:street"]["addr:housenumber"]({NO_BBOX});\n'
    return f"""
[out:json][timeout:{timeout_s}];
(
  node["addr:postcode"="{pc}"]["addr:street"]["addr:housenumber"]({NO_BBOX});
  way["addr:postcode"="{pc}"]["addr:street"]["addr:housenumber"]({NO_BBOX});
{rel});
out center;
"""


def _overpass_cache_path(pc: str) -> Path:
    return Path(settings.upload_dir).resolve() / OVERPASS_CACHE_SUBDIR / f"{pc}.json"


def _read_overpass_cache(pc: str) -> str | None:
    p = _overpass_cache_path(pc)
    if not p.is_file():
        return None
    age = time.time() - p.stat().st_mtime
    if age > OVERPASS_CACHE_MAX_AGE_S:
        return None
    log.info("Overpass: bruker disk-cache for %s (alder %.0fs)", pc, age)
    return p.read_text(encoding="utf-8")


def _write_overpass_cache(pc: str, raw_text: str) -> None:
    try:
        p = _overpass_cache_path(pc)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(raw_text, encoding="utf-8")
    except OSError as e:
        log.debug("Overpass: kunne ikke skrive cache: %s", e)


def _post_overpass(endpoint: str, query: str) -> str:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
    )
    time.sleep(REQUEST_PAUSE_S)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return resp.read().decode("utf-8")
    except TimeoutError as e:
        raise OverpassFetchError(f"Overpass HTTP-timeout etter {HTTP_TIMEOUT_S}s mot {endpoint}: {e}") from e
    except urllib.error.HTTPError as e:
        raise OverpassFetchError(f"Overpass HTTP {e.code} mot {endpoint}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise OverpassFetchError(f"Overpass nettverksfeil mot {endpoint}: {e}") from e


def _fetch_overpass_with_retries(postcode: str) -> str:
    """
    Prøv flere endepunkter, retry ved gateway/rate-limit, deretter spissere spørring uten relation.
    Returnerer rå JSON-tekst eller kaster OverpassFetchError med sammendrag av siste feil.
    """
    pc = postcode.strip().replace(" ", "")
    log.info("Overpass: endepunkter=%s", list(OVERPASS_ENDPOINTS))
    # Lett spørring først (node+way) — dekker nesten alle adresser raskere; relation er treg og sjelden nødvendig.
    # [timeout:N] = server-side max kjøretid sek (tungt postnr + treig instans → 70s feilet ofte i prod).
    q_light = int(os.environ.get("OVERPASS_QUERY_TIMEOUT_LIGHT_S", "85"))
    q_full = int(os.environ.get("OVERPASS_QUERY_TIMEOUT_FULL_S", "115"))
    strategies: tuple[tuple[str, bool, int], ...] = (
        ("light_node_way_only", False, q_light),
        ("full_node_way_relation", True, q_full),
    )
    errors: list[str] = []

    for strat_name, include_rel, q_timeout in strategies:
        query = _overpass_query(pc, include_relation=include_rel, timeout_s=q_timeout)
        for endpoint in OVERPASS_ENDPOINTS:
            for attempt in range(1, RETRY_PER_ENDPOINT + 1):
                log.info(
                    "Overpass: strategi=%s timeout_in_query=%ss endpoint=%s forsøk=%s/%s",
                    strat_name,
                    q_timeout,
                    endpoint,
                    attempt,
                    RETRY_PER_ENDPOINT,
                )
                try:
                    if attempt > 1:
                        pause = RETRY_BACKOFF_S * (attempt - 1)
                        log.info("Overpass: retry-pause %.1fs før ny POST mot %s", pause, endpoint)
                        time.sleep(pause)
                    return _post_overpass(endpoint, query)
                except OverpassFetchError as e:
                    msg = str(e)
                    errors.append(f"[{strat_name} {endpoint} #{attempt}] {msg}")
                    log.warning("Overpass feilet: %s", msg)
                    if not _should_retry_overpass(msg):
                        log.info("Overpass: ikke retrybar feil — neste endepunkt")
                        break
                    if attempt >= RETRY_PER_ENDPOINT:
                        log.info("Overpass: retry oppbrukt for %s — neste endepunkt", endpoint)
                        break

    raise OverpassFetchError(
        "Overpass: alle endepunkter/strategier feilet. Siste feil: " + " | ".join(errors[-8:])
    )


def _overpass_fetch_raw(postcode: str) -> tuple[list[dict], list[str]]:
    """
    Returnerer (kandidatrader, advarsler). Kan være tom liste ved tomt treff.
    """
    warnings: list[str] = []
    pc = postcode.strip().replace(" ", "")
    if not pc or any(c in pc for c in '["\\]'):
        warnings.append("Ugyldig postnummer — avbrutt.")
        return [], warnings

    raw_text = _read_overpass_cache(pc)
    from_disk = raw_text is not None
    if raw_text is None:
        raw_text = _fetch_overpass_with_retries(pc)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise OverpassFetchError(f"Overpass returnerte ugyldig JSON: {e}") from e

    remark = data.get("remark")
    if isinstance(remark, str) and remark.strip():
        rlow = remark.lower()
        warnings.append(f"Overpass: {remark.strip()}")
        if "timeout" in rlow or "timed out" in rlow or "too many requests" in rlow:
            raise OverpassFetchError(remark.strip())

    elements = data.get("elements")
    if not isinstance(elements, list):
        warnings.append("Overpass-svar manglet elementliste — behandles som tomt.")
        return [], warnings

    if len(elements) == 0 and from_disk:
        log.info("Overpass: cache for %s var tom — henter live på nytt", pc)
        raw_text = _fetch_overpass_with_retries(pc)
        from_disk = False
        data = json.loads(raw_text)
        elements = data.get("elements")
        if not isinstance(elements, list):
            warnings.append("Overpass-svar manglet elementliste — behandles som tomt.")
            return [], warnings

    if not from_disk and isinstance(elements, list) and len(elements) > 0:
        _write_overpass_cache(pc, raw_text)

    if len(elements) == 0:
        warnings.append(f"Ingen OSM-objekter med full gateadresse for postnummer {pc}.")

    rows: list[dict] = []
    for el in elements:
        tags = el.get("tags") or {}
        street = _normalize_street(str(tags.get("addr:street") or ""))
        hn = str(tags.get("addr:housenumber") or "").strip()
        if not street or not hn:
            continue
        el_type = str(el.get("type") or "")
        coords = _coords_from_element(el)
        if coords is None:
            continue
        lat, lon = coords
        rows.append(
            {
                "street": street,
                "housenumber": hn,
                "latitude": lat,
                "longitude": lon,
                "postcode": pc,
                "_type": el_type,
                "_prio": _element_priority(el_type),
            }
        )

    if len(rows) == 0 and len(elements) > 0:
        warnings.append(
            "Treff i OSM, men ingen komplette adresser med brukbare koordinater i Norge-boksen "
            "(krever addr:street, addr:housenumber og gyldig punkt/centroid)."
        )
    elif 0 < len(rows) < 4:
        warnings.append(f"Kun {len(rows)} unike adresser i OSM for dette postnummeret — lav dekning.")

    return rows, warnings


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    """Én rad per (gate, husnr, postnr); foretrekk node-koordinat fremfor way/relation."""
    best: dict[tuple[str, str, str], dict] = {}
    for r in rows:
        key = (
            _normalize_hn_key(r["street"]),
            _normalize_hn_key(r["housenumber"]),
            str(r["postcode"]),
        )
        cur = best.get(key)
        if cur is None or int(r["_prio"]) > int(cur["_prio"]):
            best[key] = r
    out = list(best.values())
    for r in out:
        r.pop("_type", None)
        r.pop("_prio", None)
    return out


def _ordered_locations(rows: list[dict], limit: int) -> list[dict]:
    want = max(1, min(int(limit), 500))
    by_street: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_street[r["street"]].append(r)

    ordered: list[dict] = []
    for street in sorted(by_street.keys(), key=lambda s: s.casefold()):
        chunk = sorted(by_street[street], key=lambda x: _house_sort_key(x["housenumber"]))
        for r in chunk:
            addr_line = f"{r['street']} {r['housenumber']}, {r['postcode']}"
            ordered.append(
                {
                    "address": addr_line,
                    "postcode": r["postcode"],
                    "latitude": r["latitude"],
                    "longitude": r["longitude"],
                }
            )
            if len(ordered) >= want:
                return ordered
    return ordered


def fetch_locations_bundle(postcode: str, limit: int) -> tuple[list[dict], dict]:
    """
    Hent og sorter adresser. Returnerer (liste til runner-JSON, metadata til UI/lagring).

    metadata keys: source, postcode, unique_address_count, planned_count, truncated_to_max_locations,
                    warnings, rows (plan med order 1..n)
    """
    pc = postcode.strip().replace(" ", "")
    raw_rows, warnings = _overpass_fetch_raw(pc)
    deduped = _dedupe_rows(raw_rows)
    unique_n = len(deduped)
    want = max(1, min(int(limit), 500))
    truncated = unique_n > want
    locs = _ordered_locations(deduped, limit)

    plan_rows = [
        {
            "order": i + 1,
            "address": x["address"],
            "postcode": str(x["postcode"]),
            "latitude": float(x["latitude"]),
            "longitude": float(x["longitude"]),
        }
        for i, x in enumerate(locs)
    ]
    meta = {
        "source": "overpass",
        "postcode": pc,
        "unique_address_count": unique_n,
        "planned_count": len(locs),
        "truncated_to_max_locations": truncated,
        "warnings": warnings,
        "rows": plan_rows,
    }
    return locs, meta


def write_locations_file_for_postcode(upload_dir: Path, job_id: int, postcode: str, limit: int) -> tuple[Path, dict]:
    try:
        locs, meta = fetch_locations_bundle(postcode, limit)
    except OverpassFetchError as e:
        raise ValueError(str(e)) from e
    if not locs:
        raise ValueError(
            f"Ingen adresser for postnummer «{postcode.strip()}» etter Overpass og filtrering. "
            + (
                " ".join(meta.get("warnings") or [])
                or "Sjekk OSM-dekning eller bruk statisk JSON (use_dynamic_locations=false)."
            )
        )
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"gsv_dynamic_job_{job_id}.json"
    path.write_text(json.dumps(locs, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, meta


def plan_from_static_file(path: Path, postcode: str, max_locations: int) -> dict:
    """Samme plan-format som Overpass for statisk JSON (fallback)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "locations" in raw:
        raw = raw["locations"]
    if not isinstance(raw, list):
        raise ValueError("Statisk lokasjonsfil må være en JSON-liste eller {locations: [...]}")
    pc = postcode.strip().replace(" ", "")
    want = max(1, min(int(max_locations), 500))
    matching = [x for x in raw if str(x.get("postcode", "")).strip().replace(" ", "") == pc]
    locs = matching[:want]
    plan_rows = []
    for i, x in enumerate(locs):
        plan_rows.append(
            {
                "order": i + 1,
                "address": str(x.get("address", "")),
                "postcode": str(x.get("postcode", "")),
                "latitude": float(x["latitude"]),
                "longitude": float(x["longitude"]),
            }
        )
    return {
        "source": "file",
        "postcode": pc,
        "unique_address_count": len(matching),
        "planned_count": len(locs),
        "truncated_to_max_locations": len(matching) > want,
        "warnings": [],
        "rows": plan_rows,
    }
