#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Run YOLO and save annotated images")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Image, directory, or glob")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--output", default="runs/inference/images")
    args = parser.parse_args()
    from ultralytics import YOLO

    output = Path(args.output)
    annotated_dir = output / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    records = []
    results = model.predict(
        source=args.source,
        conf=args.confidence,
        imgsz=args.image_size,
        stream=True,
        verbose=False,
    )
    for index, result in enumerate(results):
        source = Path(result.path)
        target = annotated_dir / f"{index:06d}_{source.name}"
        cv2.imwrite(str(target), result.plot())
        labels = {"fire": 0.0, "smoke": 0.0}
        for class_id, confidence in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            name = str(model.names[int(class_id)]).lower()
            if name in labels:
                labels[name] = max(labels[name], float(confidence))
        records.append(
            {
                "source": str(source),
                "annotated": str(target),
                "fire_score": labels["fire"],
                "smoke_score": labels["smoke"],
            }
        )
        print(target)
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["source", "annotated", "fire_score", "smoke_score"]
        )
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()

