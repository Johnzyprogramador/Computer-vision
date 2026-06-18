#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from firevision.metrics import binary_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Tune thresholds on validation predictions only; never tune on test"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--objective", choices=["f1", "balanced_accuracy"], default="f1")
    parser.add_argument("--minimum-specificity", type=float, default=0.0)
    parser.add_argument("--output", default="runs/thresholds.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    report = {}
    for label in ("fire", "smoke"):
        truth, score = frame[f"{label}_true"], frame[f"{label}_score"]
        candidates = np.unique(np.concatenate(([0.0], score.to_numpy(), [1.0])))
        evaluated = []
        for threshold in candidates:
            metrics = binary_metrics(truth, score, threshold=float(threshold))
            if metrics.specificity >= args.minimum_specificity:
                evaluated.append((getattr(metrics, args.objective), float(threshold), metrics))
        if not evaluated:
            raise ValueError(
                f"No {label} threshold reaches specificity {args.minimum_specificity:.3f}"
            )
        _, threshold, metrics = max(evaluated, key=lambda item: (item[0], item[2].recall))
        report[label] = {"threshold": threshold, **metrics.to_dict()}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

