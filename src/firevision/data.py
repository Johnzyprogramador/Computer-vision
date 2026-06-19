from __future__ import annotations

from pathlib import Path

from PIL import Image


def image_dataset(manifest, transform):
    import torch
    from torch.utils.data import Dataset

    class ManifestImages(Dataset):
        def __len__(self):
            return len(manifest)

        def __getitem__(self, index):
            row = manifest.iloc[index]
            image = Image.open(Path(row["resolved_path"])).convert("RGB")
            target = torch.tensor([row["fire"], row["smoke"]], dtype=torch.float32)
            return transform(image), target, str(row["path"])

    return ManifestImages()


def temporal_dataset(
    manifest,
    transform,
    sequence_length: int,
    stride: int = 1,
    label_mode: str = "last",
):
    import torch
    from torch.utils.data import Dataset

    sequences = []
    sort_columns = ["group_id"] + (["frame_index"] if "frame_index" in manifest else [])
    ordered = manifest.sort_values(sort_columns)
    for _, group in ordered.groupby("group_id", sort=False):
        rows = group.reset_index(drop=True)
        width = (sequence_length - 1) * stride + 1
        for start in range(0, len(rows) - width + 1):
            indices = [start + offset * stride for offset in range(sequence_length)]
            sequences.append(rows.iloc[indices])

    class ManifestSequences(Dataset):
        def __len__(self):
            return len(sequences)

        def __getitem__(self, index):
            rows = sequences[index]
            frames = [
                transform(Image.open(path).convert("RGB"))
                for path in rows["resolved_path"].tolist()
            ]
            if label_mode == "last":
                labels = rows.iloc[-1][["fire", "smoke"]].to_numpy()
            elif label_mode == "max":
                labels = rows[["fire", "smoke"]].max(axis=0).to_numpy()
            else:
                raise ValueError(f"Unsupported temporal label mode: {label_mode}")
            return torch.stack(frames), torch.tensor(labels, dtype=torch.float32), rows.iloc[-1]["path"]

    return ManifestSequences()
