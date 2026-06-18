#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml


def link_or_copy(source: Path, target: Path, copy: bool):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    shutil.copy2(source, target) if copy else os.symlink(source.resolve(), target)


def main():
    parser = argparse.ArgumentParser(description="Materialize YOLO train/val/test folders from a manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy rather than symlink")
    parser.add_argument("--fire-class-id", type=int, default=1)
    parser.add_argument("--smoke-class-id", type=int, default=0)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    required = {"path", "annotation_path", "split"}
    if missing := required - set(frame):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    root = Path(args.output).resolve()
    for index, row in frame.iterrows():
        image = Path(row.path)
        label = Path(row.annotation_path)
        unique_name = f"{index:08d}_{image.name}"
        link_or_copy(image, root / "images" / row.split / unique_name, args.copy)
        target_label = root / "labels" / row.split / Path(unique_name).with_suffix(".txt")
        if label.exists():
            link_or_copy(label, target_label, args.copy)
        else:
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.touch()
    config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {args.smoke_class_id: "smoke", args.fire_class_id: "fire"},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote YOLO dataset configuration to {root / 'data.yaml'}")


if __name__ == "__main__":
    main()
