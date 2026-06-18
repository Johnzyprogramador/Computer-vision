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
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", default="runs/detection/presence")
    args = parser.parse_args()
    from ultralytics import YOLO

    frame = read_manifest(args.manifest)
    frame = frame[frame["split"] == args.split].reset_index(drop=True)
    model = YOLO(args.weights)
    names = model.names
    records = []
    for row, result in zip(
        frame.itertuples(index=False),
        model.predict(frame["resolved_path"].tolist(), conf=args.confidence, imgsz=args.image_size, stream=True),
    ):
        scores = {"fire": 0.0, "smoke": 0.0}
        for class_id, confidence in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
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
