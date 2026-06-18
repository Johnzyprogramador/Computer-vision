import pytest

from firevision.device import best_device


def test_explicit_cpu_device():
    pytest.importorskip("torch")
    assert str(best_device("cpu")) == "cpu"
