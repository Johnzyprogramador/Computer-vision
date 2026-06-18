from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = {"path", "group_id", "split", "fire", "smoke"}


def read_manifest(path: str | Path, *, require_files: bool = True) -> pd.DataFrame:
    manifest_path = Path(path)
    frame = pd.read_csv(manifest_path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    for column in ("fire", "smoke"):
        frame[column] = frame[column].astype(int)
        if not set(frame[column].unique()).issubset({0, 1}):
            raise ValueError(f"{column} must contain only 0/1")
    root = manifest_path.parent
    frame["resolved_path"] = frame["path"].map(
        lambda value: str((root / str(value)).resolve()) if not Path(str(value)).is_absolute() else value
    )
    if require_files:
        missing_files = [p for p in frame["resolved_path"] if not Path(p).exists()]
        if missing_files:
            raise FileNotFoundError(f"{len(missing_files)} files are missing; first: {missing_files[0]}")
    return frame


def write_manifest(frame: pd.DataFrame, path: str | Path) -> None:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)

