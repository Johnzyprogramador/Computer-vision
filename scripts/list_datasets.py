#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="configs/datasets.yaml")
    args = parser.parse_args()
    datasets = yaml.safe_load(Path(args.catalog).read_text())["datasets"]
    for name, item in datasets.items():
        print(f"{name:24} {item['task']:24} {item['source']}")
        print(f"{'':24} {item['notes']}")


if __name__ == "__main__":
    main()

