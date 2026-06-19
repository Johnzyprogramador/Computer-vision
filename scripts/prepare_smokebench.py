#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from firevision.manifest import write_manifest
from firevision.splits import stable_group_split
from firevision.yolo import IMAGE_SUFFIXES


def images(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root).with_suffix("")): path.resolve()
        for path in root.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SmokeBench paired smoky/clean images for smoke classification"
    )
    parser.add_argument("--smoky-dir", required=True)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--output", default="data/manifests/smokebench.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    smoky_root, clean_root = Path(args.smoky_dir).resolve(), Path(args.clean_dir).resolve()
    smoky, clean = images(smoky_root), images(clean_root)
    common = sorted(set(smoky) & set(clean))
    if not common:
        raise ValueError(
            "No paired images share the same relative filename between smoky and clean directories"
        )
    missing_smoky, missing_clean = set(clean) - set(smoky), set(smoky) - set(clean)
    if missing_smoky or missing_clean:
        raise ValueError(
            f"Unpaired SmokeBench files: missing smoky={len(missing_smoky)}, "
            f"missing clean={len(missing_clean)}"
        )
    records = []
    for pair_id in common:
        records.extend(
            [
                {
                    "path": str(smoky[pair_id]),
                    "group_id": pair_id,
                    "fire": 0,
                    "smoke": 1,
                    "dataset": "smokebench",
                    "pair_type": "smoky",
                },
                {
                    "path": str(clean[pair_id]),
                    "group_id": pair_id,
                    "fire": 0,
                    "smoke": 0,
                    "dataset": "smokebench",
                    "pair_type": "clean",
                },
            ]
        )
    frame = stable_group_split(pd.DataFrame(records), seed=args.seed)
    write_manifest(frame, args.output)
    print(frame.groupby(["split", "smoke"]).size())
    print(f"Wrote {len(frame)} paired samples to {args.output}")


if __name__ == "__main__":
    main()

