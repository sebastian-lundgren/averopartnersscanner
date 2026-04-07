"""Bakgrunnstråd: kjører `python -m runner` (Street View + Playwright), ikke i FastAPI-request."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.database import SessionLocal
from app.models import StreetViewScanJobStatus

GSV_DYNAMIC_MARKER = "__dynamic__"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_runner_python_executable(repo: Path) -> str:
    """Ikke .resolve() på interpretator: venv/bin/python-symlink kan hoppe ut av venv."""
    raw = (settings.gsv_scan_runner_python or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = repo / p
        if p.is_file():
            return os.fspath(p)
        raise FileNotFoundError(
            f"GSV_SCAN_RUNNER_PYTHON peker på manglende fil ({p}). "
            "Sjekk sti eller fjern variabelen for automatisk valg."
        )

    if platform.system() == "Windows":
        candidates = (
            repo / "backend" / ".venv-runner" / "Scripts" / "python.exe",
            repo / ".venv-runner" / "Scripts" / "python.exe",
        )
    else:
        candidates = (
            repo / "backend" / ".venv-runner" / "bin" / "python",
            repo / ".venv-runner" / "bin" / "python",
        )
    for cand in candidates:
        if cand.is_file():
            return os.fspath(cand)

    raise FileNotFoundError(
        "Mangler runner-Python. Forventet backend/.venv-runner (f.eks. Render) eller "
        ".venv-runner i repo-rot (lokalt), med Playwright — eller sett GSV_SCAN_RUNNER_PYTHON."
    )


def create_gsv_scan_job(
    db: Session,
    *,
    postcode: str,
    max_locations: int | None,
    max_attempts: int | None,
    max_images_per_address: int | None,
    locations_json_path: str | None,
    use_dynamic_locations: bool,
) -> models.StreetViewScanJob:
    if use_dynamic_locations:
        path = GSV_DYNAMIC_MARKER
    else:
        path = (locations_json_path or settings.gsv_scan_locations_path or "").strip()
        if not path:
            path = "runner/data/example_locations.json"
    ma = int(max_attempts if max_attempts is not None else settings.gsv_scan_max_attempts_default)
    mi_default = int(settings.gsv_scan_max_images_per_address_default)
    mi = int(max_images_per_address if max_images_per_address is not None else mi_default)
    job = models.StreetViewScanJob(
        status=StreetViewScanJobStatus.QUEUED,
        postcode=postcode.strip(),
        max_locations=int(max_locations if max_locations is not None else settings.gsv_scan_max_locations_default),
        max_attempts=ma,
        max_images_per_address=mi,
        locations_json_path=path,
    )
    db.add(job)
    db.flush()
    return job


def start_gsv_scan_job_thread(job_id: int) -> None:
    t = threading.Thread(target=run_gsv_scan_job_sync, args=(job_id,), daemon=True)
    t.start()


def run_gsv_scan_job_sync(job_id: int) -> None:
    temp_loc_file: Path | None = None
    try:
        db = SessionLocal()
        try:
            job = db.get(models.StreetViewScanJob, job_id)
            if not job:
                return
            job.status = StreetViewScanJobStatus.RUNNING
            job.started_at = datetime.utcnow()
            job.error_message = None
            db.commit()

            run_job_id = job.id
            run_postcode = job.postcode.strip()
            run_max_locations = job.max_locations
            run_max_attempts = job.max_attempts
            run_max_images_per_address = job.max_images_per_address
            run_locations_json_path = job.locations_json_path
        finally:
            db.close()

        repo = _repo_root()
        if run_locations_json_path == GSV_DYNAMIC_MARKER:
            from app.services.gsv_location_fetch import write_locations_file_for_postcode

            dyn_dir = Path(settings.upload_dir).resolve() / "gsv_dynamic"
            temp_loc_file, plan_meta = write_locations_file_for_postcode(
                dyn_dir, run_job_id, run_postcode, run_max_locations
            )
            loc = temp_loc_file.resolve()
        else:
            loc = Path(run_locations_json_path)
            if not loc.is_absolute():
                loc = repo / loc
            loc = loc.resolve()
            if not loc.is_file():
                raise FileNotFoundError(f"Mangler locations-fil: {loc}")
            from app.services.gsv_location_fetch import plan_from_static_file

            plan_meta = plan_from_static_file(loc, run_postcode, run_max_locations)

        db = SessionLocal()
        try:
            job = db.get(models.StreetViewScanJob, job_id)
            if not job:
                return
            job.locations_plan_json = json.dumps(plan_meta, ensure_ascii=False)
            db.commit()
        finally:
            db.close()

        env = os.environ.copy()
        env["SCANNER_API_BASE"] = settings.gsv_scan_runner_api_base.rstrip("/")
        tok = (settings.scanner_api_token or "").strip()
        if tok:
            env["SCANNER_API_TOKEN"] = tok

        py = _resolve_runner_python_executable(repo)
        cmd = [
            py,
            "-m",
            "runner",
            "--locations",
            str(loc),
            "--postcode",
            run_postcode,
            "--max-addresses",
            str(run_max_locations),
            "--max-attempts",
            str(run_max_attempts),
            "--max-images",
            str(run_max_images_per_address),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=None,
        )

        combined_log = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        m = re.search(r"ScanRun\s+(\d+)\s+med", combined_log)
        parsed_run_id = int(m.group(1)) if m else None

        db = SessionLocal()
        try:
            job = db.get(models.StreetViewScanJob, job_id)
            if not job:
                return
            if proc.returncode == 0:
                job.status = StreetViewScanJobStatus.DONE
                job.error_message = None
                if parsed_run_id is not None:
                    job.scan_run_id = parsed_run_id
            else:
                job.status = StreetViewScanJobStatus.FAILED
                tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()[-8000:]
                job.error_message = tail or f"Runner avsluttet med kode {proc.returncode}"
            job.finished_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()
    except Exception as e:
        db = SessionLocal()
        try:
            job = db.get(models.StreetViewScanJob, job_id)
            if job:
                job.status = StreetViewScanJobStatus.FAILED
                job.error_message = str(e)[:8000]
                job.finished_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    finally:
        if temp_loc_file is not None and temp_loc_file.is_file():
            try:
                temp_loc_file.unlink(missing_ok=True)
            except OSError:
                pass
