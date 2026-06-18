#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Ultralytics YOLO fire/smoke detection")
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="runs/detection")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        project=args.output,
        name="train",
    )
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.image_size,
        device=args.device,
        project=args.output,
        name="test",
    )
    result = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "note": (
            "Object detectors do not have a natural TN count. Use evaluate_detector_presence.py "
            "for image-level TP/TN/FP/FN in addition to mAP."
        ),
    }
    target = Path(args.output) / "test_metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

