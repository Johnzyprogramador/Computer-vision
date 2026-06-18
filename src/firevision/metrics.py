from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    iou: float
    false_positive_rate: float
    false_negative_rate: float
    balanced_accuracy: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float] | None = None,
    *,
    y_pred: Iterable[int] | None = None,
    threshold: float = 0.5,
) -> BinaryMetrics:
    truth = np.asarray(list(y_true), dtype=np.int8)
    if y_pred is None:
        if y_score is None:
            raise ValueError("Provide y_score or y_pred")
        pred = (np.asarray(list(y_score), dtype=float) >= threshold).astype(np.int8)
    else:
        pred = np.asarray(list(y_pred), dtype=np.int8)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: truth={truth.shape}, prediction={pred.shape}")
    if not set(np.unique(truth)).issubset({0, 1}) or not set(np.unique(pred)).issubset({0, 1}):
        raise ValueError("Binary metrics require labels in {0, 1}")

    tp = int(np.sum((truth == 1) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    precision = _divide(tp, tp + fp)
    return BinaryMetrics(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=_divide(tp + tn, len(truth)),
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=_divide(2 * precision * recall, precision + recall),
        iou=_divide(tp, tp + fp + fn),
        false_positive_rate=_divide(fp, fp + tn),
        false_negative_rate=_divide(fn, fn + tp),
        balanced_accuracy=(recall + specificity) / 2,
    )


def multilabel_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    threshold: float = 0.5,
) -> dict[str, dict[str, int | float]]:
    if y_true.shape != y_score.shape or y_true.shape[1] != len(class_names):
        raise ValueError("Expected matching [samples, classes] arrays and class names")
    results = {
        name: binary_metrics(y_true[:, i], y_score[:, i], threshold=threshold).to_dict()
        for i, name in enumerate(class_names)
    }
    results["macro"] = {
        key: float(np.mean([results[name][key] for name in class_names]))
        for key in ("accuracy", "precision", "recall", "specificity", "f1", "iou")
    }
    return results

