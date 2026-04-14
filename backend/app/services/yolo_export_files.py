"""Skriver YOLO dataset (images + labels) fra DB til disk."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from sqlalchemy.orm import Session

from app import models
from app.services.blob_storage import is_r2_ref, materialize_local_path
from app.services.path_resolve import resolve_stored_path
from app.services.bbox_multi import is_valid_box, normalize_box
from app.services.yolo_service import bbox_to_yolo_line


def _move_one_example_between_splits(export_root: Path, *, src: str, dst: str) -> bool:
    src_img_dir = export_root / "images" / src
    dst_img_dir = export_root / "images" / dst
    src_lbl_dir = export_root / "labels" / src
    dst_lbl_dir = export_root / "labels" / dst
    for img in src_img_dir.glob("*"):
        if not img.is_file():
            continue
        stem = img.stem
        src_lbl = src_lbl_dir / f"{stem}.txt"
        if not src_lbl.is_file():
            continue
        img.rename(dst_img_dir / img.name)
        src_lbl.rename(dst_lbl_dir / src_lbl.name)
        return True
    return False


def write_yolo_dataset(
    db: Session,
    export_root: Path,
    *,
    clear_first: bool = False,
) -> dict[str, int | dict[str, int]]:
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
    split_by_te_id = {int(r.training_example_id): str(r.split) for r in rows}
    training_rows = db.query(models.TrainingExample).all()
    counts: dict[str, int | dict[str, int]] = {"train": 0, "val": 0, "rejected": 0}
    skip_reasons: dict[str, int] = {}
    seen_stem: set[str] = set()
    valid_labels = {"alarm_sign", "not_alarm_sign", "unclear"}
    examples_total = len(training_rows)
    examples_with_tags = 0
    examples_with_valid_label = 0
    examples_with_valid_bbox = 0
    examples_with_existing_split = 0
    examples_backfilled_split = 0
    written_train = 0
    written_val = 0

    def _skip(reason: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for te in training_rows:
        if not te.tags_json:
            _skip("missing_tags_json")
            continue
        examples_with_tags += 1
        img = db.get(models.ImageAsset, te.image_id)
        if not img:
            _skip("missing_image_asset")
            continue
        tags = te.tags_json
        ann = str(tags.get("annotation_label") or "")
        bbox = tags.get("bbox_norm")
        bboxes_multi = tags.get("bboxes_norm")
        if ann in valid_labels:
            examples_with_valid_label += 1
        has_valid_bbox = False
        if isinstance(bboxes_multi, list) and bboxes_multi:
            has_valid_bbox = any(isinstance(b, dict) and is_valid_box(b) for b in bboxes_multi)
        elif isinstance(bbox, dict) and is_valid_box(bbox):
            has_valid_bbox = True
        if has_valid_bbox:
            examples_with_valid_bbox += 1
        split = split_by_te_id.get(te.id)
        if split is None:
            split = "val" if te.id % 5 == 0 else "train"
            db.add(models.YoloDatasetEntry(training_example_id=te.id, split=split))
            split_by_te_id[te.id] = split
            examples_backfilled_split += 1
        else:
            examples_with_existing_split += 1
        if split not in counts:
            _skip("invalid_split")
            continue
        stem = f"img_{img.id}_{te.id}"
        if stem in seen_stem:
            _skip("duplicate_stem")
            continue
        seen_stem.add(stem)
        if is_r2_ref(img.stored_path):
            local_src, tmp_del = materialize_local_path(img.stored_path, suffix=".yolo")
            try:
                if not local_src.is_file():
                    _skip("missing_source_image_r2")
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
                _skip("missing_source_image_local")
                continue
            ext = src_img.suffix or ".jpg"
            dst_img = export_root / "images" / split / f"{stem}{ext}"
            shutil.copy2(src_img, dst_img)

        if split == "rejected":
            counts["rejected"] = int(counts["rejected"]) + 1
            continue

        label_path = export_root / "labels" / split / f"{stem}.txt"
        lines: list[str] = []
        if ann == "alarm_sign":
            if isinstance(bboxes_multi, list) and bboxes_multi:
                for b in bboxes_multi:
                    if isinstance(b, dict) and is_valid_box(b):
                        lines.append(bbox_to_yolo_line(0, normalize_box(b)))
            elif isinstance(bbox, dict) and is_valid_box(bbox):
                lines.append(bbox_to_yolo_line(0, normalize_box(bbox)))
            else:
                _skip("alarm_sign_without_valid_bbox")
        label_path.write_text("".join(lines), encoding="utf-8")
        counts[split] = int(counts[split]) + 1
        if split == "train":
            written_train += 1
        elif split == "val":
            written_val += 1

    # Auto-reparer ugyldig split i eksporten (minst ett eksempel i både train og val).
    if int(counts["train"]) > 1 and int(counts["val"]) == 0:
        if _move_one_example_between_splits(export_root, src="train", dst="val"):
            counts["train"] = int(counts["train"]) - 1
            counts["val"] = int(counts["val"]) + 1
            written_train -= 1
            written_val += 1
    elif int(counts["val"]) > 1 and int(counts["train"]) == 0:
        if _move_one_example_between_splits(export_root, src="val", dst="train"):
            counts["val"] = int(counts["val"]) - 1
            counts["train"] = int(counts["train"]) + 1
            written_val -= 1
            written_train += 1

    # dataset.yaml: bruk eksplisitte absolutte train/val-mapper. Ultralytics 8.x kan fortsatt
    # slå sammen «path» + relative train/val feil (cwd), som ga …/backend/images/val.
    yaml_path = export_root / "dataset.yaml"
    root_res = export_root.resolve()
    train_dir = root_res / "images" / "train"
    val_dir = root_res / "images" / "val"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {json.dumps(str(root_res))}",
                f"train: {json.dumps(str(train_dir))}",
                f"val: {json.dumps(str(val_dir))}",
                "",
                "names:",
                "  0: alarm_sign",
                "",
            ]
        ),
        encoding="utf-8",
    )
    counts["training_examples_total"] = examples_total
    counts["training_examples_with_tags_json"] = examples_with_tags
    counts["training_examples_with_valid_annotation_label"] = examples_with_valid_label
    counts["training_examples_with_valid_bbox_or_bboxes"] = examples_with_valid_bbox
    counts["training_examples_with_existing_yolo_dataset_entry"] = examples_with_existing_split
    counts["training_examples_backfilled_to_yolo_dataset_entry"] = examples_backfilled_split
    counts["written_to_train"] = written_train
    counts["written_to_val"] = written_val
    counts["skipped_total"] = sum(skip_reasons.values())
    counts["skipped_reasons"] = skip_reasons
    return counts
