#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from firevision.manifest import read_manifest
from firevision.metrics import binary_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Images loaded per inference call; reduce to 1 on CUDA out-of-memory",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, cpu")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", default="runs/detection/presence")
    args = parser.parse_args()
    from ultralytics import YOLO

    frame = read_manifest(args.manifest)
    frame = frame[frame["split"] == args.split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows found for split={args.split}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    model = YOLO(args.weights)
    names = model.names
    records = []
    total = len(frame)
    for start in range(0, total, args.batch_size):
        batch = frame.iloc[start : start + args.batch_size]
        results = model.predict(
            source=batch["resolved_path"].tolist(),
            conf=args.confidence,
            imgsz=args.image_size,
            device=args.device,
            verbose=False,
        )
        for row, result in zip(batch.itertuples(index=False), results, strict=True):
            scores = {"fire": 0.0, "smoke": 0.0}
            for class_id, confidence in zip(
                result.boxes.cls.tolist(), result.boxes.conf.tolist()
            ):
                name = str(names[int(class_id)]).lower()
                if name in scores:
                    scores[name] = max(scores[name], float(confidence))
            records.append(
                {
                    "path": row.path,
                    "fire_true": row.fire,
                    "smoke_true": row.smoke,
                    "fire_score": scores["fire"],
                    "smoke_score": scores["smoke"],
                }
            )
        done = min(start + args.batch_size, total)
        print(f"\rEvaluated {done}/{total} images", end="", flush=True)
    print()
    predictions = pd.DataFrame(records)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    report = {
        label: binary_metrics(
            predictions[f"{label}_true"],
            predictions[f"{label}_score"],
            threshold=args.confidence,
        ).to_dict()
        for label in ("fire", "smoke")
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
