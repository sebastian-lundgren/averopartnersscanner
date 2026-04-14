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
    split_by_te_id = {int(r.training_example_id): str(r.split) for r in rows}
    training_rows = db.query(models.TrainingExample).all()
    counts = {"train": 0, "val": 0, "rejected": 0}
    seen_stem: set[str] = set()

    for te in training_rows:
        if not te.tags_json:
            continue
        img = db.get(models.ImageAsset, te.image_id)
        if not img:
            continue
        tags = te.tags_json
        ann = str(tags.get("annotation_label") or "")
        bbox = tags.get("bbox_norm")
        bboxes_multi = tags.get("bboxes_norm")
        split = split_by_te_id.get(te.id)
        if split is None:
            split = "val" if te.id % 5 == 0 else "train"
            db.add(models.YoloDatasetEntry(training_example_id=te.id, split=split))
            split_by_te_id[te.id] = split
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
        lines: list[str] = []
        if ann == "alarm_sign":
            if isinstance(bboxes_multi, list) and bboxes_multi:
                for b in bboxes_multi:
                    if isinstance(b, dict) and is_valid_box(b):
                        lines.append(bbox_to_yolo_line(0, normalize_box(b)))
            elif isinstance(bbox, dict) and is_valid_box(bbox):
                lines.append(bbox_to_yolo_line(0, normalize_box(bbox)))
        label_path.write_text("".join(lines), encoding="utf-8")
        counts[split] += 1

    # Auto-reparer ugyldig split i eksporten (minst ett eksempel i både train og val).
    if counts["train"] > 1 and counts["val"] == 0:
        if _move_one_example_between_splits(export_root, src="train", dst="val"):
            counts["train"] -= 1
            counts["val"] += 1
    elif counts["val"] > 1 and counts["train"] == 0:
        if _move_one_example_between_splits(export_root, src="val", dst="train"):
            counts["val"] -= 1
            counts["train"] += 1

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
    return counts
