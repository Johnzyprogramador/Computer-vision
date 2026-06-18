#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from firevision.manifest import write_manifest
from firevision.splits import stable_group_split


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
    # D-Fire is image-based. Prefix grouping reduces duplicate/burst leakage where names encode a source.
    stem = path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def main():
    parser = argparse.ArgumentParser(description="Build a unified manifest from D-Fire YOLO files")
    parser.add_argument("--root", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--output", default="data/manifests/dfire.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fire-class-id", type=int, default=1)
    parser.add_argument("--smoke-class-id", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    image_root, label_root = root / args.images, root / args.labels
    records = []
    for image in sorted(p for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        relative = image.relative_to(image_root)
        fire, smoke = labels_for(
            (label_root / relative).with_suffix(".txt"),
            args.fire_class_id,
            args.smoke_class_id,
        )
        records.append(
            {
                "path": str(image),
                "group_id": infer_group(image),
                "fire": fire,
                "smoke": smoke,
                "annotation_path": str((label_root / relative).with_suffix(".txt")),
                "dataset": "d_fire",
            }
        )
    if not records:
        raise FileNotFoundError(f"No images found below {image_root}")
    frame = stable_group_split(pd.DataFrame(records), seed=args.seed)
    write_manifest(frame, args.output)
    print(frame.groupby(["split", "fire", "smoke"]).size())
    print(f"Wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
