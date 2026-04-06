"""
API for manuell kvalitetskontroll av alarmskilt — kun autoriserte use cases.

IKKE for massekartlegging eller "ingen alarm"-lister. Tre tillatte sluttstatuser.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import (
    addresses,
    dashboard,
    export,
    files,
    images,
    model_versions,
    reviews,
    scanner,
    settings as settings_router,
    streetview_scan_jobs,
    train_jobs,
    training,
    yolo_admin,
)
from app.seed import seed_if_empty


def create_app() -> FastAPI:
    app = FastAPI(
        title="Alarmskilt QC API",
        description="Review- og treningsverktøy. Unknown-first; menneske i løkken.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.evidence_dir).mkdir(parents=True, exist_ok=True)
    Path("./data").mkdir(parents=True, exist_ok=True)
    Path("./data/models").mkdir(parents=True, exist_ok=True)
    Path(settings.yolo_dataset_export_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.yolo_train_output_dir).mkdir(parents=True, exist_ok=True)

    init_db()
    seed_if_empty()

    app.include_router(images.router)
    app.include_router(addresses.router)
    app.include_router(reviews.router)
    app.include_router(dashboard.router)
    app.include_router(export.router)
    app.include_router(settings_router.router)
    app.include_router(model_versions.router)
    app.include_router(training.router)
    app.include_router(train_jobs.router)
    app.include_router(files.router)
    app.include_router(scanner.router)
    app.include_router(streetview_scan_jobs.router)
    app.include_router(yolo_admin.router)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
