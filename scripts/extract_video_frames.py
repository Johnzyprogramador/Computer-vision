#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Extract frames while preserving video group IDs")
    parser.add_argument("--videos", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--manifest", default="data/manifests/unlabeled_video_frames.csv")
    args = parser.parse_args()
    video_root, output = Path(args.videos), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for video in sorted(p for p in video_root.rglob("*") if p.suffix.lower() in suffixes):
        capture = cv2.VideoCapture(str(video))
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, round(source_fps / args.fps))
        group_id = str(video.relative_to(video_root).with_suffix("")).replace("/", "__")
        group_dir = output / group_id
        group_dir.mkdir(parents=True, exist_ok=True)
        source_index = extracted_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % interval == 0:
                target = group_dir / f"{extracted_index:07d}.jpg"
                cv2.imwrite(str(target), frame)
                records.append(
                    {
                        "path": str(target.resolve()),
                        "group_id": group_id,
                        "frame_index": extracted_index,
                        "timestamp_seconds": source_index / source_fps,
                        "fire": 0,
                        "smoke": 0,
                        "split": "train",
                    }
                )
                extracted_index += 1
            source_index += 1
        capture.release()
    target = Path(args.manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(target, index=False)
    print(f"Wrote {len(records)} frames to {target}; labels and group-level splits must be reviewed.")


if __name__ == "__main__":
    main()

