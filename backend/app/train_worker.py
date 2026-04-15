from __future__ import annotations

import os
import time

from app.database import SessionLocal, init_db
from app.services.train_pipeline import claim_next_queued_train_job, recover_stale_train_jobs, run_train_job_sync


def run_worker_loop() -> None:
    poll_s = float(os.getenv("TRAIN_WORKER_POLL_SECONDS", "3").strip() or "3")
    init_db()
    while True:
        db = SessionLocal()
        try:
            recover_stale_train_jobs(db)
            job_id = claim_next_queued_train_job(db, runner_kind="render-worker")
        finally:
            db.close()
        if job_id is None:
            time.sleep(poll_s)
            continue
        run_train_job_sync(job_id)


if __name__ == "__main__":
    run_worker_loop()
