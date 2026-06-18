#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from firevision.manifest import write_manifest
from firevision.splits import stable_group_split, validate_group_splits


def labels_for(label_path: Path, fire_class_id: int, smoke_class_id: int) -> tuple[int, int]:
    fire = smoke = 0
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            class_id = int(line.split()[0])
            fire |= class_id == fire_class_id
            smoke |= class_id == smoke_class_id
    return int(fire), int(smoke)


def infer_group(path: Path) -> str:
    # D-Fire is image-based and does not publish source-video grouping for every image.
    return path.stem


def find_layout(root: Path, images_name: str, labels_name: str):
    """Return (declared split, image root, label root) for flat or pre-split D-Fire layouts."""
    flat_images = root / images_name
    if flat_images.is_dir():
        return [(None, flat_images, root / labels_name)]

    split_roots = []
    for split in ("train", "val", "test"):
        image_root = root / split / images_name
        if image_root.is_dir():
            split_roots.append((split, image_root, root / split / labels_name))
    return split_roots


def training_split(group_id: str, val_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if value < val_fraction else "train"


def main():
    parser = argparse.ArgumentParser(description="Build a unified manifest from D-Fire YOLO files")
    parser.add_argument("--root", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--output", default="data/manifests/dfire.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction carved from official train when no val directory exists",
    )
    parser.add_argument("--fire-class-id", type=int, default=1)
    parser.add_argument("--smoke-class-id", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    root = Path(args.root).resolve()
    layout = find_layout(root, args.images, args.labels)
    if not layout:
        expected = (
            f"{root / args.images} or "
            f"{root / 'train' / args.images} plus {root / 'test' / args.images}"
        )
        raise FileNotFoundError(f"No D-Fire image directories found. Expected {expected}")

    records = []
    for declared_split, image_root, label_root in layout:
        if not label_root.is_dir():
            raise FileNotFoundError(f"Label directory is missing: {label_root}")
        images = sorted(
            p for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for image in images:
            relative = image.relative_to(image_root)
            label_path = (label_root / relative).with_suffix(".txt")
            fire, smoke = labels_for(
                label_path,
                args.fire_class_id,
                args.smoke_class_id,
            )
            records.append(
                {
                    "path": str(image),
                    "group_id": (
                        f"{declared_split}:{infer_group(image)}"
                        if declared_split
                        else infer_group(image)
                    ),
                    "fire": fire,
                    "smoke": smoke,
                    "annotation_path": str(label_path),
                    "dataset": "d_fire",
                    "source_split": declared_split or "unsplit",
                }
            )
    if not records:
        roots = ", ".join(str(image_root) for _, image_root, _ in layout)
        raise FileNotFoundError(f"No images found below: {roots}")

    frame = pd.DataFrame(records)
    if layout[0][0] is None:
        frame = stable_group_split(frame, seed=args.seed)
    else:
        has_validation = "val" in set(frame["source_split"])
        frame["split"] = frame.apply(
            lambda row: (
                row["source_split"]
                if row["source_split"] != "train" or has_validation
                else training_split(row["group_id"], args.val_fraction, args.seed)
            ),
            axis=1,
        )
        validate_group_splits(frame)

    write_manifest(frame, args.output)
    print("Detected layout:")
    for declared_split, image_root, label_root in layout:
        print(f"  {declared_split or 'unsplit':7} images={image_root} labels={label_root}")
    print(frame.groupby(["split", "fire", "smoke"]).size())
    print(f"Wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
