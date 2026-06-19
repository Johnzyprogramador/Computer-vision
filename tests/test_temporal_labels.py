import pytest

from firevision.data import temporal_dataset


def test_invalid_temporal_label_mode_is_rejected(tmp_path):
    pytest.importorskip("torch")
    import pandas as pd
    from PIL import Image
    from torchvision.transforms import ToTensor

    rows = []
    for index in range(2):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (4, 4)).save(path)
        rows.append(
            {
                "resolved_path": str(path),
                "path": str(path),
                "group_id": "event",
                "frame_index": index,
                "fire": 0,
                "smoke": index,
            }
        )
    dataset = temporal_dataset(
        pd.DataFrame(rows), ToTensor(), sequence_length=2, label_mode="invalid"
    )
    with pytest.raises(ValueError, match="label mode"):
        dataset[0]
