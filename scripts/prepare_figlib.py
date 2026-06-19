#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from firevision.manifest import write_manifest
from firevision.splits import validate_group_splits
from firevision.yolo import IMAGE_SUFFIXES


def image_index(root: Path) -> dict[str, Path]:
    index = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        index[str(relative.with_suffix(""))] = path.resolve()
        index[path.stem] = path.resolve()
    return index


def frame_index(key: str) -> int:
    stem = Path(key).name
    first = stem.split("_", 1)[0]
    return int(first) if first.lstrip("-").isdigit() else 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert official FIgLib/SmokeyNet split JSON files to a temporal manifest"
    )
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--output", default="data/manifests/figlib.csv")
    args = parser.parse_args()

    root = Path(args.images_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"FIgLib image root not found: {root}")
    index = image_index(root)
    records, missing = [], []
    for split, json_path in (
        ("train", args.train_json),
        ("val", args.val_json),
        ("test", args.test_json),
    ):
        entries = json.loads(Path(json_path).read_text())
        if not isinstance(entries, dict):
            raise ValueError(f"Expected object mapping in {json_path}")
        for key, metadata in entries.items():
            normalized = str(Path(key).with_suffix(""))
            image = index.get(normalized) or index.get(Path(key).name)
            if image is None:
                missing.append(key)
                continue
            event = str(metadata.get("camera_name") or Path(key).parts[0])
            records.append(
                {
                    "path": str(image),
                    "group_id": event,
                    "frame_index": frame_index(key),
                    "split": split,
                    "fire": 0,
                    "smoke": int(metadata["image_gt"]),
                    "dataset": "figlib",
                    "camera_name": event,
                }
            )
    if missing:
        example = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)} FIgLib JSON entries have no matching image below {root}; "
            f"examples: {example}"
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No FIgLib images matched the supplied JSON files")
    validate_group_splits(frame)
    write_manifest(frame, args.output)
    print(frame.groupby(["split", "smoke"]).size())
    print(f"Wrote {len(frame)} temporal frames to {args.output}")


if __name__ == "__main__":
    main()
