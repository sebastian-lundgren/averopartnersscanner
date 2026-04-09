"""
Cloudflare R2 (S3-kompatibel) eller lokal disk.
DB lagrer prefiks r2:<object_key> for objekter i bucket; ellers lokal sti.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.config import settings
from app.services.path_resolve import resolve_stored_path

R2_PREFIX = "r2:"


def is_r2_ref(stored: str) -> bool:
    return bool(stored) and stored.startswith(R2_PREFIX)


def r2_object_key(stored: str) -> str:
    return stored[len(R2_PREFIX) :] if is_r2_ref(stored) else stored


def r2_enabled() -> bool:
    has_r2_creds = bool(
        settings.r2_bucket_name and settings.r2_access_key_id and settings.r2_secret_access_key
    )
    if not has_r2_creds:
        return False
    backend = settings.storage_backend.strip().lower()
    # Prefer explicit r2, but auto-enable when R2 is configured to avoid
    # writing ephemeral local paths in production after deploy/restart.
    return backend == "r2" or backend == "" or backend == "local"


def _s3_client():
    import boto3
    from botocore.config import Config

    endpoint = (settings.r2_endpoint_url or "").strip() or (
        f"https://{settings.r2_account_id}.r2.cloudflarestorage.com"
        if settings.r2_account_id
        else ""
    )
    if not endpoint:
        raise RuntimeError("R2: sett R2_ENDPOINT_URL eller R2_ACCOUNT_ID")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def put_bytes(key: str, data: bytes, content_type: str) -> str:
    """Last opp til R2; returner r2:key."""
    client = _s3_client()
    client.put_object(
        Bucket=settings.r2_bucket_name,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    return f"{R2_PREFIX}{key}"


def get_bytes(ref_or_key: str) -> bytes:
    """Hent objekt; ref kan være r2:key eller bare key hvis R2."""
    key = r2_object_key(ref_or_key) if is_r2_ref(ref_or_key) else ref_or_key
    client = _s3_client()
    obj = client.get_object(Bucket=settings.r2_bucket_name, Key=key)
    return obj["Body"].read()


def materialize_local_path(stored_path: str, *, suffix: str = "") -> tuple[Path, bool]:
    """
    Returner (lokal Path, må_slettes_etterpå).
    For r2: laster ned til tempfil.
    """
    if is_r2_ref(stored_path):
        data = get_bytes(stored_path)
        fd, name = tempfile.mkstemp(suffix=suffix or ".bin", prefix="r2dl_")
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        return Path(name), True
    return resolve_stored_path(stored_path), False


def store_upload_bytes(data: bytes, original_filename: str, content_type: str) -> tuple[str, str]:
    """Returner (stored_path_ref, safe_original_name)."""
    safe = Path(original_filename or "image.jpg").name
    name = f"{uuid.uuid4().hex[:12]}_{safe}"
    if r2_enabled():
        key = f"uploads/{name}"
        ref = put_bytes(key, data, content_type or "image/jpeg")
        return ref, safe
    dest_dir = Path(settings.upload_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / name
    path.write_bytes(data)
    return str(path.resolve()), safe


def stream_r2_object(ref: str):
    """Body stream for FastAPI StreamingResponse."""
    key = r2_object_key(ref)
    client = _s3_client()
    return client.get_object(Bucket=settings.r2_bucket_name, Key=key)["Body"]
