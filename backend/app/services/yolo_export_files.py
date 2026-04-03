"""Skriver YOLO dataset (images + labels) fra DB til disk."""

from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app import models
from app.services.blob_storage import is_r2_ref, materialize_local_path
from app.services.path_resolve import resolve_stored_path
from app.services.yolo_service import bbox_to_yolo_line


def write_yolo_dataset(
    db: Session,
    export_root: Path,
    *,
    clear_first: bool = False,
) -> dict[str, int]:
    """
    Kopier bilder og .txt-labels for train/val. rejected: kun bilde i rejected/ (uten label).
    """
    export_root = Path(export_root)
    for sub in ("images/train", "images/val", "images/rejected", "labels/train", "labels/val"):
        (export_root / sub).mkdir(parents=True, exist_ok=True)
    if clear_first:
        for sub in ("images/train", "images/val", "images/rejected", "labels/train", "labels/val"):
            p = export_root / sub
            for f in p.glob("*"):
                if f.is_file():
                    f.unlink()

    rows = db.query(models.YoloDatasetEntry).all()
    counts = {"train": 0, "val": 0, "rejected": 0}
    seen_stem: set[str] = set()

    for ent in rows:
        te = db.get(models.TrainingExample, ent.training_example_id)
        if not te or not te.tags_json:
            continue
        img = db.get(models.ImageAsset, te.image_id)
        if not img:
            continue
        tags = te.tags_json
        ann = str(tags.get("annotation_label") or "")
        bbox = tags.get("bbox_norm")
        split = ent.split
        if split not in counts:
            continue
        stem = f"img_{img.id}_{te.id}"
        if stem in seen_stem:
            continue
        seen_stem.add(stem)
        if is_r2_ref(img.stored_path):
            local_src, tmp_del = materialize_local_path(img.stored_path, suffix=".yolo")
            try:
                if not local_src.is_file():
                    continue
                ext = local_src.suffix or ".jpg"
                dst_img = export_root / "images" / split / f"{stem}{ext}"
                shutil.copy2(local_src, dst_img)
            finally:
                if tmp_del:
                    local_src.unlink(missing_ok=True)
        else:
            src_img = resolve_stored_path(img.stored_path)
            if not src_img.is_file():
                continue
            ext = src_img.suffix or ".jpg"
            dst_img = export_root / "images" / split / f"{stem}{ext}"
            shutil.copy2(src_img, dst_img)

        if split == "rejected":
            counts["rejected"] += 1
            continue

        label_path = export_root / "labels" / split / f"{stem}.txt"
        if ann == "alarm_sign" and isinstance(bbox, dict) and all(k in bbox for k in ("x", "y", "w", "h")):
            label_path.write_text(bbox_to_yolo_line(0, bbox), encoding="utf-8")
        else:
            # not_alarm_sign / unclear: tom label (negativt eksempel) eller tom ved unclear
            label_path.write_text("", encoding="utf-8")
        counts[split] += 1

    # dataset.yaml
    yaml_path = export_root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: alarm_sign",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return counts
