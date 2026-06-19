#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from firevision.yolo import IMAGE_SUFFIXES


def main():
    parser = argparse.ArgumentParser(description="Inspect an extracted dataset before preparation")
    parser.add_argument("--root", required=True)
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    suffixes = Counter()
    image_count = label_count = video_count = 0
    videos = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        suffixes[suffix or "<none>"] += 1
        image_count += suffix in IMAGE_SUFFIXES
        label_count += suffix == ".txt"
        video_count += suffix in videos

    print(f"Root: {root}")
    print(f"Images: {image_count}")
    print(f"YOLO/text labels: {label_count}")
    print(f"Videos: {video_count}")
    print("Top file types:")
    for suffix, count in suffixes.most_common(12):
        print(f"  {suffix:10} {count}")
    print("Directory tree:")
    directories = sorted(path for path in root.rglob("*") if path.is_dir())
    for path in directories:
        depth = len(path.relative_to(root).parts)
        if depth <= args.max_depth:
            print(f"  {'  ' * (depth - 1)}{path.name}/")


if __name__ == "__main__":
    main()
