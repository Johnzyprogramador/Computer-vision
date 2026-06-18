import pandas as pd
import pytest

from firevision.splits import stable_group_split, validate_group_splits


def test_group_split_never_leaks():
    frame = pd.DataFrame(
        {"group_id": ["video_a"] * 10 + ["video_b"] * 10 + ["video_c"] * 10}
    )
    result = stable_group_split(frame)
    assert result.groupby("group_id")["split"].nunique().max() == 1


def test_leakage_is_rejected():
    frame = pd.DataFrame(
        {"group_id": ["same_video", "same_video"], "split": ["train", "test"]}
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_group_splits(frame)

