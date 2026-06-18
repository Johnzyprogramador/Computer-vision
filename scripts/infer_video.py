#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from firevision.alerts import PersistentAlert


def main():
    parser = argparse.ArgumentParser(description="YOLO video inference with persistence filtering")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--required", type=int, default=6)
    parser.add_argument("--cooldown", type=int, default=30)
    parser.add_argument("--output", default="runs/inference/events.csv")
    parser.add_argument("--output-video", default="runs/inference/annotated.mp4")
    parser.add_argument("--frame-log", default="runs/inference/frame_predictions.csv")
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video source: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_target = Path(args.output_video)
    video_target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {video_target}")
    filters = {
        "fire": PersistentAlert(args.window, args.required, args.cooldown),
        "smoke": PersistentAlert(args.window, args.required, args.cooldown),
    }
    events, frame_records, frame_index = [], [], 0
    alert_until = {"fire": -1, "smoke": -1}
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, conf=args.confidence, verbose=False)[0]
        present = {"fire": False, "smoke": False}
        scores = {"fire": 0.0, "smoke": 0.0}
        for class_id, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            name = str(model.names[int(class_id)]).lower()
            if name in present:
                present[name] = True
                scores[name] = max(scores[name], float(score))
        for label, value in present.items():
            if filters[label].update(value):
                alert_until[label] = frame_index + max(1, round(fps * 2))
                events.append(
                    {"frame_index": frame_index, "seconds": frame_index / fps, "label": label}
                )
        annotated = result.plot()
        active = [label.upper() for label, end in alert_until.items() if frame_index <= end]
        if active:
            cv2.rectangle(annotated, (0, 0), (width, 55), (0, 0, 180), -1)
            cv2.putText(
                annotated,
                f"PERSISTENT ALERT: {', '.join(active)}",
                (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        writer.write(annotated)
        frame_records.append(
            {
                "frame_index": frame_index,
                "seconds": frame_index / fps,
                "fire_score": scores["fire"],
                "smoke_score": scores["smoke"],
                "fire_present": int(present["fire"]),
                "smoke_present": int(present["smoke"]),
            }
        )
        frame_index += 1
    capture.release()
    writer.release()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "seconds", "label"])
        writer.writeheader()
        writer.writerows(events)
    frame_target = Path(args.frame_log)
    frame_target.parent.mkdir(parents=True, exist_ok=True)
    with frame_target.open("w", newline="") as handle:
        frame_writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "seconds",
                "fire_score",
                "smoke_score",
                "fire_present",
                "smoke_present",
            ],
        )
        frame_writer.writeheader()
        frame_writer.writerows(frame_records)
    print(f"Wrote {len(events)} persistent events to {target}")
    print(f"Wrote annotated video to {video_target}")
    print(f"Wrote per-frame scores to {frame_target}")


if __name__ == "__main__":
    main()
