import numpy as np

from firevision.metrics import binary_metrics, multilabel_metrics


def test_binary_counts_and_rates():
    result = binary_metrics([1, 1, 0, 0], y_pred=[1, 0, 1, 0])
    assert (result.tp, result.tn, result.fp, result.fn) == (1, 1, 1, 1)
    assert result.accuracy == 0.5
    assert result.f1 == 0.5
    assert result.iou == 1 / 3


def test_multilabel_has_each_class_and_macro():
    truth = np.array([[1, 0], [0, 1]])
    score = np.array([[0.9, 0.1], [0.2, 0.8]])
    result = multilabel_metrics(truth, score, ["fire", "smoke"])
    assert result["fire"]["tp"] == 1
    assert result["smoke"]["tn"] == 1
    assert result["macro"]["f1"] == 1

