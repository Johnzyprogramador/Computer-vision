#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
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


def write_remapped_label(source: Path, target: Path, source_names: dict[int, str]):
    target.parent.mkdir(parents=True, exist_ok=True)
    output = []
    if source.exists():
        for line in source.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            name = source_names.get(int(fields[0]), "").lower()
            if "smoke" in name:
                class_id = 0
            elif "fire" in name or "flame" in name:
                class_id = 1
            else:
                continue
            output.append(" ".join([str(class_id), *fields[1:]]))
    target.write_text("\n".join(output) + ("\n" if output else ""))


def main():
    parser = argparse.ArgumentParser(description="Materialize YOLO train/val/test folders from a manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy rather than symlink")
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
        if "source_class_names" in frame and pd.notna(row.source_class_names):
            source_names = {
                int(index): str(name)
                for index, name in ast.literal_eval(row.source_class_names).items()
            }
            write_remapped_label(label, target_label, source_names)
        elif label.exists():
            link_or_copy(label, target_label, args.copy)
        else:
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.touch()
    config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "smoke", 1: "fire"},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote YOLO dataset configuration to {root / 'data.yaml'}")


if __name__ == "__main__":
    main()
