from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_dfire.py"
SPEC = spec_from_file_location("prepare_dfire", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_detects_presplit_layout(tmp_path):
    for split in ("train", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    layout = MODULE.find_layout(tmp_path, "images", "labels")
    assert [item[0] for item in layout] == ["train", "test"]


def test_detects_flat_layout(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    layout = MODULE.find_layout(tmp_path, "images", "labels")
    assert layout == [(None, tmp_path / "images", tmp_path / "labels")]


def test_validation_assignment_is_stable():
    first = MODULE.training_split("image-123", 0.15, 42)
    second = MODULE.training_split("image-123", 0.15, 42)
    assert first == second
