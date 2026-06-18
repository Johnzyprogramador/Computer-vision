#!/usr/bin/env python3
from __future__ import annotations

import argparse

from firevision.manifest import read_manifest
from firevision.splits import validate_group_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    frame = read_manifest(args.manifest)
    validate_group_splits(frame)
    print(frame.groupby(["split", "fire", "smoke"]).size())
    print(f"OK: {len(frame)} samples, {frame.group_id.nunique()} independent groups")


if __name__ == "__main__":
    main()
