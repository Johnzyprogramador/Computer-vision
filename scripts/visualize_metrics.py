#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def main():
    parser = argparse.ArgumentParser(
        description="Create interactive confusion, ROC, PR, and score reports"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="runs/plots")
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for label in ("fire", "smoke"):
        truth_column, score_column = f"{label}_true", f"{label}_score"
        if truth_column not in frame or score_column not in frame:
            continue
        truth = frame[truth_column].astype(int).to_numpy()
        score = frame[score_column].astype(float).to_numpy()
        prediction = (score >= args.threshold).astype(int)
        matrix = confusion_matrix(truth, prediction, labels=[0, 1])

        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Confusion matrix @ {args.threshold:.2f}",
                "ROC curve",
                "Precision–recall curve",
                "Score distribution",
            ),
        )
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=["predicted negative", f"predicted {label}"],
                y=["actual negative", f"actual {label}"],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            ),
            row=1,
            col=1,
        )
        if len(np.unique(truth)) == 2:
            false_positive_rate, true_positive_rate, _ = roc_curve(truth, score)
            precision, recall, _ = precision_recall_curve(truth, score)
            figure.add_trace(
                go.Scatter(
                    x=false_positive_rate,
                    y=true_positive_rate,
                    mode="lines",
                    name="ROC",
                ),
                row=1,
                col=2,
            )
            figure.add_trace(
                go.Scatter(x=recall, y=precision, mode="lines", name="PR"),
                row=2,
                col=1,
            )
        for value, name, color in ((0, "negative", "#4472C4"), (1, label, "#C00000")):
            figure.add_trace(
                go.Histogram(
                    x=score[truth == value],
                    name=name,
                    opacity=0.65,
                    marker_color=color,
                    nbinsx=25,
                    histnorm="probability density",
                ),
                row=2,
                col=2,
            )
        figure.add_vline(x=args.threshold, line_dash="dash", row=2, col=2)
        figure.update_xaxes(title_text="False-positive rate", row=1, col=2)
        figure.update_yaxes(title_text="True-positive rate", row=1, col=2)
        figure.update_xaxes(title_text="Recall", row=2, col=1)
        figure.update_yaxes(title_text="Precision", row=2, col=1)
        figure.update_xaxes(title_text="Model score", range=[0, 1], row=2, col=2)
        figure.update_layout(
            title=f"{label.title()} evaluation",
            template="plotly_white",
            height=850,
            barmode="overlay",
        )
        target = output / f"{label}_evaluation.html"
        figure.write_html(target, include_plotlyjs=True, full_html=True)
        print(target)


if __name__ == "__main__":
    main()
