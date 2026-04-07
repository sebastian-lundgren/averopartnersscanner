from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ModelVersionOut(BaseModel):
    id: int
    version_tag: str
    description: str | None
    deployed_at: datetime
    is_active: bool
    weights_path: str | None = None
    metrics_json: dict[str, Any] | None = None
    training_config_json: dict[str, Any] | None = None
    train_image_count: int | None = None

    model_config = {"from_attributes": True}


class ImageAssetOut(BaseModel):
    id: int
    address_id: int | None
    original_filename: str
    stored_path: str
    evidence_crop_path: str | None
    width: int | None
    height: int | None
    uploaded_at: datetime
    is_temporary_candidate: bool
    quality_score: float | None
    discard_reason: str | None
    is_primary_for_address: bool

    model_config = {"from_attributes": True}


class LibraryImageOut(ImageAssetOut):
    """Bibliotek-liste: tilknyttet adresse, rekkefølgenummer og visningsnavn (ikke filnavn)."""

    address_line: str | None = None
    address_final_status: str | None = None
    sequence_within_address: int
    display_name: str


class PredictionOut(BaseModel):
    id: int
    image_id: int
    model_version_id: int
    predicted_status: str
    confidence: int
    # Én bbox (legacy), liste, eller {"boxes":[...],"v":2}
    bbox_json: Any | None
    rationale: str | None
    needs_review: bool
    priority_score: float
    review_completed: bool
    created_at: datetime
    claimed_by: str | None = None
    claimed_at: datetime | None = None
    model_version: ModelVersionOut | None = None

    model_config = {"from_attributes": True}


class ReviewDecisionOut(BaseModel):
    id: int
    prediction_id: int
    final_status: str
    was_override: bool
    comment: str | None
    error_type: str | None
    decided_at: datetime

    model_config = {"from_attributes": True}


class QueueItemOut(BaseModel):
    prediction: PredictionOut
    image: ImageAssetOut
    review: ReviewDecisionOut | None


class AnnotatorBody(BaseModel):
    annotator_id: str = Field(..., min_length=1, max_length=128)


class AddressOut(BaseModel):
    id: int
    customer_id: str | None
    address_line: str | None
    notes: str | None
    attempt_count: int
    best_quality_score: float | None
    selected_image_id: int | None
    final_human_status: str | None
    selection_metadata_json: dict | None
    image_count: int = 0
    highest_confidence: int | None = None
    last_prediction_at: datetime | None = None

    model_config = {"from_attributes": True}


class ReviewSubmit(BaseModel):
    final_status: str = Field(..., pattern="^(skilt_funnet|uklart|trenger_manuell)$")
    # Valgfri datasett-label; når satt overstyrer den final_status (kartes til ReviewStatus).
    annotation_label: str | None = Field(
        None,
        pattern="^(alarm_sign|not_alarm_sign|unclear)$",
    )
    # Normalisert bbox {x,y,w,h} 0–1 for treningsdata; default er modellens forslag.
    annotation_bbox_json: dict[str, float] | None = None
    # Flere skilt i samme bilde — overstyrer enkelt-bbox når satt.
    annotation_bboxes_json: list[dict[str, float]] | None = Field(None, max_length=48)
    # YOLO datasett-split ved lagring av treningsrad (train/val/rejected).
    yolo_dataset_split: str | None = Field(None, pattern="^(train|val|rejected)$")
    comment: str | None = None
    error_type: str | None = None
    approve_without_change: bool = False
    annotator_id: str | None = Field(None, max_length=128)


class ScannerLocationIn(BaseModel):
    address: str
    postcode: str
    latitude: float
    longitude: float


class ScannerLocationBulk(BaseModel):
    locations: list[ScannerLocationIn]


class ScanRunStart(BaseModel):
    postcode: str
    max_locations: int = 50
    # Når satt: bruk eksakt disse test_locations-radene i denne rekkefølgen (fra bulk_locations-respons).
    location_ids: list[int] | None = None


class ScanRunStartOut(BaseModel):
    scan_run_id: int
    items: list[dict]


class ScanAttemptIn(BaseModel):
    attempt_index: int
    screenshot_path: str | None = None
    camera_state: str | None = None
    prediction_status: str | None = None
    confidence: int | None = None
    bbox_json: Any | None = None
    rationale: str | None = None


class ScanItemComplete(BaseModel):
    final_result: str
    best_confidence: float | None = None
    notes: str | None = None


class YoloTrainRequest(BaseModel):
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    base_model: str = "yolov8s.pt"
    name: str = "train_run"
    device: str | None = None


class TrainJobStartBody(BaseModel):
    base_model: str | None = None
    epochs: int | None = None
    imgsz: int | None = None
    batch: int | None = None
    device: str | None = None


class StreetViewScanJobStartBody(BaseModel):
    postcode: str | None = None
    max_locations: int | None = Field(default=None, ge=1, le=500)
    max_attempts: int | None = Field(default=None, ge=1, le=20)
    max_images_per_address: int | None = Field(default=None, ge=1, le=20)
    locations_json_path: str | None = None
    # True (standard): hent gate+husnr. via OSM Overpass før runner. False: bruk JSON-fil.
    use_dynamic_locations: bool = True


class ScanJobLocationPlanRow(BaseModel):
    order: int
    address: str
    postcode: str
    latitude: float
    longitude: float


class ScanJobLocationsPlan(BaseModel):
    source: str = "overpass"
    postcode: str
    unique_address_count: int
    planned_count: int
    truncated_to_max_locations: bool
    warnings: list[str] = Field(default_factory=list)
    rows: list[ScanJobLocationPlanRow] = Field(default_factory=list)


class ScanJobAddressOutcome(BaseModel):
    order: int
    location_id: int
    address: str
    final_result: str | None = None
    notes: str | None = None
    images_saved: int = 0


class ScanJobImageDebug(BaseModel):
    image_id: int
    bbox_count: int = 0
    used_stored_path: str | None = None
    annotated: bool = False


class StreetViewScanJobResultSummary(BaseModel):
    scan_run_id: int
    run_status: str
    total_locations: int
    completed_locations: int
    locations_with_detection: int
    images_saved: int
    image_ids: list[int]
    predictions_pending_review: int
    address_outcomes: list[ScanJobAddressOutcome] = Field(default_factory=list)
    image_debug: list[ScanJobImageDebug] = Field(default_factory=list)


class StreetViewScanJobOut(BaseModel):
    id: int
    status: str
    postcode: str
    max_locations: int
    max_attempts: int
    max_images_per_address: int
    locations_json_path: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    scan_run_id: int | None = None
    result_summary: StreetViewScanJobResultSummary | None = None
    locations_plan: ScanJobLocationsPlan | None = None

    model_config = {"from_attributes": True}


class TrainJobOut(BaseModel):
    id: int
    status: str
    trigger: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    config_json: dict[str, Any] | None
    export_counts_json: dict[str, Any] | None
    metrics_json: dict[str, Any] | None
    new_annotations_snapshot: int | None
    candidate_model_version_id: int | None
    activated_new_model: bool

    model_config = {"from_attributes": True}


class YoloDatasetAssign(BaseModel):
    training_example_id: int
    split: str = Field(..., pattern="^(train|val|rejected)$")


class TrainingLibraryUpsert(BaseModel):
    category: str
    tags: dict[str, Any] | None = None
    notes: str | None = None


class ThresholdsUpdate(BaseModel):
    threshold_strong_sign: int | None = None
    threshold_unclear_high: int | None = None
    threshold_unclear_low: int | None = None
    max_best_view_attempts: int | None = None
    quality_threshold: float | None = None


class DashboardOut(BaseModel):
    total_images: int
    total_predictions: int
    count_skilt_funnet: int
    count_uklart: int
    count_trenger_manuell: int
    overrides_count: int
    pending_review: int
    error_rate_last_7d: float | None
    by_model_version: list[dict[str, Any]]
