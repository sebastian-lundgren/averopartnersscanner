from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    pass


def _normalize_database_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    if u.startswith("postgres://"):
        return "postgresql+psycopg2://" + u[len("postgres://") :]
    if u.startswith("postgresql://") and not u.startswith("postgresql+"):
        return "postgresql+psycopg2://" + u[len("postgresql://") :]
    return u


_db_url = _normalize_database_url(settings.database_url)
if _db_url.startswith("sqlite"):
    _connect_args: dict = {"check_same_thread": False}
elif _db_url.startswith("postgresql"):
    _connect_args = {}
    if settings.database_sslmode.strip() and "sslmode" not in _db_url.lower():
        _connect_args["sslmode"] = settings.database_sslmode.strip()
else:
    _connect_args = {}

engine_kwargs: dict = {}
if _db_url.startswith("postgresql"):
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_recycle"] = 300

engine = create_engine(
    _db_url,
    connect_args=_connect_args,
    **engine_kwargs,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _column_names(engine, table: str) -> set[str]:
    insp = inspect(engine)
    if not insp.has_table(table):
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def apply_schema_patches():
    """Legg til nye kolonner på eksisterende tabeller (SQLite/Postgres)."""
    tc_json = "TEXT" if engine.dialect.name == "sqlite" else "JSON"

    def _add(table: str, col: str, ddl: str) -> None:
        cols = _column_names(engine, table)
        if not cols or col in cols:
            return
        with engine.begin() as conn:
            conn.execute(text(ddl))

    _add("model_versions", "weights_path", "ALTER TABLE model_versions ADD COLUMN weights_path VARCHAR(1024)")
    _add(
        "model_versions",
        "training_config_json",
        f"ALTER TABLE model_versions ADD COLUMN training_config_json {tc_json}",
    )
    _add("model_versions", "train_image_count", "ALTER TABLE model_versions ADD COLUMN train_image_count INTEGER")
    _add("predictions", "claimed_by", "ALTER TABLE predictions ADD COLUMN claimed_by VARCHAR(128)")
    _add("predictions", "claimed_at", "ALTER TABLE predictions ADD COLUMN claimed_at TIMESTAMP")
    _add("training_examples", "annotated_by", "ALTER TABLE training_examples ADD COLUMN annotated_by VARCHAR(128)")
    _add("streetview_scan_jobs", "scan_run_id", "ALTER TABLE streetview_scan_jobs ADD COLUMN scan_run_id INTEGER")
    _add(
        "streetview_scan_jobs",
        "max_images_per_address",
        "ALTER TABLE streetview_scan_jobs ADD COLUMN max_images_per_address INTEGER DEFAULT 4",
    )
    _add(
        "streetview_scan_jobs",
        "locations_plan_json",
        "ALTER TABLE streetview_scan_jobs ADD COLUMN locations_plan_json TEXT",
    )


def init_db():
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    try:
        apply_schema_patches()
    except Exception:
        pass
