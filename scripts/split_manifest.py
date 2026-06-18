#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from firevision.manifest import write_manifest
from firevision.splits import stable_group_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-column", default="group_id")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = stable_group_split(
        pd.read_csv(args.input),
        group_column=args.group_column,
        train=args.train,
        val=args.val,
        seed=args.seed,
    )
    write_manifest(frame, args.output)
    print(frame.groupby("split").size())


if __name__ == "__main__":
    main()

