from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import binary_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate binary fire/smoke CSV predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    report = {
        label: binary_metrics(
            frame[f"{label}_true"], frame[f"{label}_score"], threshold=args.threshold
        ).to_dict()
        for label in ("fire", "smoke")
    }
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
