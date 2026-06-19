#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from firevision.splits import stable_group_split


def main():
    parser = argparse.ArgumentParser(
        description="Extract paired FireSentry infrared/mask frames for segmentation"
    )
    parser.add_argument("--region", required=True, help="Path such as 'Region A'")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", default="data/manifests/firesentry_masks.csv")
    parser.add_argument("--every", type=int, default=10, help="Keep every Nth frame")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import cv2

    if args.every < 1:
        raise ValueError("--every must be at least 1")

    region = Path(args.region).resolve()
    infrared_root = region / "Infrared Videos"
    mask_root = region / "Fire Mask Videos"
    if not infrared_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(
            f"Expected '{infrared_root}' and '{mask_root}' from the official FireSentry layout"
        )
    output = Path(args.output_root).resolve()
    image_output, mask_output = output / "images", output / "masks"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)
    records = []
    for infrared_path in sorted(infrared_root.glob("*.mp4")):
        mask_path = mask_root / infrared_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing paired mask video: {mask_path}")
        infrared_video, mask_video = cv2.VideoCapture(str(infrared_path)), cv2.VideoCapture(
            str(mask_path)
        )
        frame_index = 0
        while True:
            infrared_ok, infrared = infrared_video.read()
            mask_ok, mask = mask_video.read()
            if infrared_ok != mask_ok:
                raise ValueError(f"Frame-count mismatch for {infrared_path.name}")
            if not infrared_ok:
                break
            if frame_index % args.every == 0:
                name = f"{infrared_path.stem}_{frame_index:07d}.png"
                image_target, mask_target = image_output / name, mask_output / name
                cv2.imwrite(str(image_target), infrared)
                gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
                cv2.imwrite(str(mask_target), binary_mask)
                records.append(
                    {
                        "image": str(image_target),
                        "mask": str(mask_target),
                        "group_id": infrared_path.stem,
                        "frame_index": frame_index,
                    }
                )
            frame_index += 1
        infrared_video.release()
        mask_video.release()
    if not records:
        raise ValueError("No paired FireSentry frames were extracted")
    frame = stable_group_split(pd.DataFrame(records), seed=args.seed)
    target = Path(args.manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    print(frame.groupby("split").size())
    print(f"Wrote {len(frame)} infrared/mask pairs to {target}")


if __name__ == "__main__":
    main()
