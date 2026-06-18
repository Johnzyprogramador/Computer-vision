#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", help="JSON metric files")
    parser.add_argument("--output", default="runs/leaderboard.csv")
    args = parser.parse_args()
    rows = []
    for filename in args.metrics:
        data = json.loads(Path(filename).read_text())
        for label in ("fire", "smoke"):
            if label in data:
                rows.append({"run": filename, "label": label, **data[label]})
        if "tp" in data:  # single-mask segmentation run
            rows.append({"run": filename, "label": "mask", **data})
    frame = pd.DataFrame(rows)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    columns = [c for c in ("run", "label", "tp", "tn", "fp", "fn", "f1", "iou") if c in frame]
    print(frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()

