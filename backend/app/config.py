import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_default_port = os.environ.get("PORT", "8000").strip() or "8000"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///./data/app.db"
    # Postgres (Supabase): URL kan være postgres:// — normaliseres i database.py til psycopg2
    database_sslmode: str = ""  # f.eks. require hvis ikke allerede i DATABASE_URL
    upload_dir: Path = Path("./data/uploads")
    evidence_dir: Path = Path("./data/evidence")
    cors_origins: str = "http://localhost:3000"
    # local | r2 — R2 krever bucket + nøkler (+ account_id eller endpoint)
    storage_backend: str = "local"
    r2_account_id: str = ""
    r2_bucket_name: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_endpoint_url: str = Field(
        default="",
        validation_alias=AliasChoices("R2_ENDPOINT_URL", "R2_ENDPOINT"),
    )  # valgfri; ellers https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com
    max_best_view_attempts: int = 5
    quality_threshold: float = 0.45
    # Justerbare terskler (lagres også i DB settings)
    threshold_strong_sign: int = 80
    threshold_unclear_high: int = 79
    threshold_unclear_low: int = 50
    # Grounding DINO (Hugging Face Hub — offisiell IDEA-Research-vekter, ikke kopi av repo)
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base"
    grounding_dino_box_threshold: float = 0.35
    grounding_dino_text_threshold: float = 0.28
    # Kommaseparerte fraser (open-set); engelsk fungerer best med forhåndstrening
    grounding_dino_phrases: str = (
        "alarm sticker,alarm sign,security alarm label,burglar alarm plate,"
        "red alarm sign,door sticker,small sign on door,glass door sign"
    )
    # OpenAI vision (plausibility-gate). Tom nøkkel = hopp over GPT og bruk kun DINO.
    openai_api_key: str = ""
    # Standard: gpt-4o-mini (rask/ rimelig); sett OPENAI_VISION_MODEL=gpt-4o for tyngre vurdering.
    openai_vision_model: str = "gpt-4o-mini"
    gpt_plausibility_direct_threshold: int = 90
    # YOLO (inferens i API, eksport, trening)
    yolo_model_path: Path = Path("./data/models/yolov8s.pt")
    yolo_confidence_strong: float = 0.65
    yolo_confidence_weak: float = 0.35
    # Under disse: lagre alle bbox-forslag, men ikke behandle rangert primær som pålitelig (evidens/review).
    yolo_primary_trust_min_conf: float = 0.45
    yolo_primary_trust_min_composite: float = 0.30
    yolo_dataset_export_dir: Path = Path("./data/yolo_dataset")
    yolo_train_output_dir: Path = Path("./data/yolo_runs")
    # YOLO auto-treningssløyfe (ultralytics)
    yolo_train_base_model: str = "yolov8s.pt"
    yolo_train_epochs: int = 50
    yolo_train_imgsz: int = 640
    yolo_train_batch: int = 8
    yolo_train_device: str = ""  # tom = auto
    yolo_train_min_train_images: int = 5
    yolo_train_min_val_images: int = 2
    yolo_train_trigger_min_new_annotations: int = 25
    yolo_train_auto_enabled: bool = False
    # Sammenligning for auto-aktivering (metric: map50_95 | map50 | precision | recall)
    yolo_activation_metric: str = "map50_95"
    yolo_activation_min_delta: float = 0.0
    review_claim_ttl_seconds: int = 3600
    # False på små instanser (f.eks. Render 2 GB): ingen torch/HF/DINO i web-prosessen for opplasting/seed/auto-YOLO-kø.
    ml_inference_enabled: bool = True
    # Street View scan-runner (Playwright) kaller API
    scanner_api_token: str = ""
    # Web-MVP: start av runner som subprocess (python -m runner), stier relativt repo-rot
    # Statisk fallback når use_dynamic_locations=false. Dynamisk modus bruker OSM Overpass (se gsv_location_fetch).
    gsv_scan_locations_path: str = "runner/data/example_locations.json"
    gsv_scan_default_postcode: str = "0154"
    gsv_scan_max_locations_default: int = 5
    gsv_scan_max_attempts_default: int = 4
    gsv_scan_max_images_per_address_default: int = 4
    # Base-URL runner bruker mot dette API-et (må nå backend fra runner-prosessen)
    gsv_scan_runner_api_base: str = f"http://127.0.0.1:{_default_port}"
    # Tom = bruk repo/.venv-runner/bin/python (eller Scripts\python.exe på Windows)
    gsv_scan_runner_python: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
