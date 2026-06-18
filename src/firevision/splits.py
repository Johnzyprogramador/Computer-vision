from __future__ import annotations

import hashlib

import pandas as pd

SPLITS = ("train", "val", "test")


def stable_group_split(
    frame: pd.DataFrame,
    *,
    group_column: str = "group_id",
    train: float = 0.7,
    val: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Split whole videos/cameras/scenes, never adjacent frames."""
    if train <= 0 or val < 0 or train + val >= 1:
        raise ValueError("Expected train > 0, val >= 0, and train + val < 1")
    if group_column not in frame:
        raise ValueError(f"Missing required group column: {group_column}")

    def assign(group: object) -> str:
        digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        return "train" if value < train else "val" if value < train + val else "test"

    result = frame.copy()
    result["split"] = result[group_column].map(assign)
    validate_group_splits(result, group_column=group_column)
    return result


def validate_group_splits(frame: pd.DataFrame, *, group_column: str = "group_id") -> None:
    required = {group_column, "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    invalid = set(frame["split"].dropna().unique()) - set(SPLITS)
    if invalid:
        raise ValueError(f"Invalid splits: {sorted(invalid)}")
    leaked = frame.groupby(group_column)["split"].nunique()
    leaked = leaked[leaked > 1]
    if not leaked.empty:
        examples = ", ".join(map(str, leaked.index[:5]))
        raise ValueError(f"Temporal/group leakage detected for: {examples}")

