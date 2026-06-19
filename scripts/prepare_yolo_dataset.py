#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib

import pandas as pd

from firevision.manifest import write_manifest
from firevision.splits import validate_group_splits
from firevision.yolo import (
    class_names,
    images_from_source,
    infer_label_path,
    labels_for,
    load_yolo_config,
    resolve_split_sources,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a standard YOLO data.yaml (including PYRONEAR) to FireVision manifest"
    )
    parser.add_argument("--data", required=True, help="Source YOLO data.yaml")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root, config = load_yolo_config(args.data)
    names = class_names(config)
    records = []
    declared_splits = [split for split in ("train", "val", "test") if config.get(split)]
    if not declared_splits:
        raise ValueError("YOLO YAML does not define train, val, or test")

    for split in declared_splits:
        for source in resolve_split_sources(dataset_root, config[split]):
            for image in images_from_source(source):
                if not image.exists():
                    raise FileNotFoundError(f"Image listed by YOLO config is missing: {image}")
                label = infer_label_path(image)
                fire, smoke = labels_for(label, names)
                records.append(
                    {
                        "path": str(image),
                        "group_id": f"{split}:{image.stem}",
                        "split": split,
                        "fire": fire,
                        "smoke": smoke,
                        "annotation_path": str(label),
                        "dataset": args.dataset_name,
                        "source_class_names": repr(names),
                    }
                )
    if not records:
        raise ValueError("No images were discovered from the YOLO YAML")

    frame = pd.DataFrame(records)
    if "val" not in set(frame["split"]):
        def train_or_val(group_id: str) -> str:
            digest = hashlib.sha256(f"{args.seed}:{group_id}".encode()).digest()
            value = int.from_bytes(digest[:8], "big") / 2**64
            return "val" if value < 0.15 else "train"

        train_mask = frame["split"] == "train"
        frame.loc[train_mask, "split"] = frame.loc[train_mask, "group_id"].map(train_or_val)
    if "test" not in set(frame["split"]):
        raise ValueError(
            "Source dataset has no test split. Create an event/camera-level test split before "
            "reporting final metrics."
        )
    validate_group_splits(frame)
    write_manifest(frame, args.output)
    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {names}")
    print(frame.groupby(["split", "fire", "smoke"]).size())
    print(f"Wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
