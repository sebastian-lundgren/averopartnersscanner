import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ReviewStatus(str, enum.Enum):
    """Kun disse tre — aldri 'ingen alarm' eller fravær som kategori."""

    SKILT_FUNNET = "skilt_funnet"
    UKLART = "uklart"
    TRENGER_MANUELL = "trenger_manuell"


class ErrorType(str, enum.Enum):
    FEIL_OBJEKT = "feil_objekt"
    DARLIG_VINKEL = "darlig_vinkel"
    FOR_LANGT_UNNA = "for_langt_unna"
    REFLEKS_SKYGGE = "refleks_skygge"
    LIGNENDE_FASADE = "lignende_fasadedetalj"
    SKILT_DELVIS_SKJULT = "skilt_delvis_skjult"
    INNGANG_IKKE_SYNLIG = "inngang_ikke_synlig"
    BILDEKVALITET = "bildekvalitet_for_darlig"
    ANNET = "annet"


class TrainingLibraryCategory(str, enum.Enum):
    POSITIVE = "positive"
    NEGATIVE_IRRELEVANT = "negative_irrelevant"
    VANSKELIG = "vanskelig"
    VINKEL_VARIASJON = "vinkel_variasjon"
    LYS_VARIASJON = "lys_variasjon"
    DELVIS_SKJULT = "delvis_skjult"


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_tag: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    deployed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    weights_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    training_config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    train_image_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_version")


class AddressRecord(Base):
    __tablename__ = "address_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    address_line: Mapped[str | None] = mapped_column(String(512), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    best_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    selected_image_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("image_assets.id"), nullable=True)
    final_human_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    selection_metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    images: Mapped[list["ImageAsset"]] = relationship(
        back_populates="address",
        foreign_keys="ImageAsset.address_id",
    )


class ImageAsset(Base):
    __tablename__ = "image_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    address_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("address_records.id"), nullable=True, index=True)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    evidence_crop_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(128), default="image/jpeg")
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_temporary_candidate: Mapped[bool] = mapped_column(Boolean, default=False)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    discard_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_primary_for_address: Mapped[bool] = mapped_column(Boolean, default=False)

    address: Mapped["AddressRecord | None"] = relationship(
        back_populates="images",
        foreign_keys=[address_id],
    )
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="image")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(Integer, ForeignKey("image_assets.id"), nullable=False, index=True)
    model_version_id: Mapped[int] = mapped_column(Integer, ForeignKey("model_versions.id"), nullable=False)
    predicted_status: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)  # 0–100
    bbox_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # {x,y,w,h} normalized 0–1
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=True)
    priority_score: Mapped[float] = mapped_column(Float, default=0.0)
    review_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    claimed_by: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    image: Mapped["ImageAsset"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")
    review: Mapped["ReviewDecision | None"] = relationship(back_populates="prediction", uselist=False)


class ReviewDecision(Base):
    __tablename__ = "review_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    final_status: Mapped[str] = mapped_column(String(32), nullable=False)
    was_override: Mapped[bool] = mapped_column(Boolean, default=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prediction: Mapped["Prediction"] = relationship(back_populates="review")


class TrainingExample(Base):
    """Læring fra rettinger og merkede eksempler — ikke en database over 'mangler alarm'."""

    __tablename__ = "training_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(Integer, ForeignKey("image_assets.id"), nullable=False)
    source_prediction_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("predictions.id"), nullable=True)
    human_status: Mapped[str] = mapped_column(String(32), nullable=False)
    original_model_guess: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version_tag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence_at_time: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_crop_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    tags_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    annotated_by: Mapped[str | None] = mapped_column(String(128), nullable=True)


class TrainingLibraryEntry(Base):
    __tablename__ = "training_library_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(Integer, ForeignKey("image_assets.id"), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    tags_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TestLocation(Base):
    """Adresse/koordinat for Street View-scan."""

    __tablename__ = "test_locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    address: Mapped[str] = mapped_column(String(512), nullable=False)
    postcode: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ScanRun(Base):
    __tablename__ = "scan_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    test_postcode: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    total_locations: Mapped[int] = mapped_column(Integer, default=0)
    completed_locations: Mapped[int] = mapped_column(Integer, default=0)
    detections_found: Mapped[int] = mapped_column(Integer, default=0)
    failed_locations: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="running")

    items: Mapped[list["ScanRunItem"]] = relationship(back_populates="scan_run")


class ScanRunItem(Base):
    __tablename__ = "scan_run_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("scan_runs.id"), nullable=False, index=True)
    location_id: Mapped[int] = mapped_column(Integer, ForeignKey("test_locations.id"), nullable=False, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    attempts_used: Mapped[int] = mapped_column(Integer, default=0)
    final_result: Mapped[str | None] = mapped_column(String(64), nullable=True)
    best_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    scan_run: Mapped["ScanRun"] = relationship(back_populates="items")


class ScanAttempt(Base):
    __tablename__ = "scan_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_run_item_id: Mapped[int] = mapped_column(Integer, ForeignKey("scan_run_items.id"), nullable=False, index=True)
    attempt_index: Mapped[int] = mapped_column(Integer, nullable=False)
    screenshot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    camera_state: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prediction_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DetectionHit(Base):
    __tablename__ = "detection_hits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_run_item_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("scan_run_items.id"), nullable=True, index=True)
    location_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("test_locations.id"), nullable=True, index=True)
    image_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("image_assets.id"), nullable=True)
    prediction_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("predictions.id"), nullable=True)
    screenshot_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class YoloDatasetEntry(Base):
    """Kobling fra review-treningsrad til train/val/rejected for YOLO-eksport."""

    __tablename__ = "yolo_dataset_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_example_id: Mapped[int] = mapped_column(Integer, ForeignKey("training_examples.id"), nullable=False, unique=True)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TrainJobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainJob(Base):
    __tablename__ = "train_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=TrainJobStatus.QUEUED, index=True)
    trigger: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    export_counts_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    new_annotations_snapshot: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidate_model_version_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model_versions.id"), nullable=True
    )
    activated_new_model: Mapped[bool] = mapped_column(Boolean, default=False)


class StreetViewScanJobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class StreetViewScanJob(Base):
    """Kø for Google Street View-scan via ekstern runner (Playwright), ikke i HTTP-tråden."""

    __tablename__ = "streetview_scan_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=StreetViewScanJobStatus.QUEUED, index=True
    )
    postcode: Mapped[str] = mapped_column(String(16), nullable=False)
    max_locations: Mapped[int] = mapped_column(Integer, default=10)
    max_attempts: Mapped[int] = mapped_column(Integer, default=4)
    max_images_per_address: Mapped[int] = mapped_column(Integer, default=4)
    locations_json_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    scan_run_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("scan_runs.id"), nullable=True, index=True)
    locations_plan_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
