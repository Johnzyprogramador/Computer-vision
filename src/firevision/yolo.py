from __future__ import annotations

from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yolo_config(path: str | Path) -> tuple[Path, dict]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YOLO YAML: {config_path}")
    dataset_root = Path(config.get("path", config_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (config_path.parent / dataset_root).resolve()
    return dataset_root, config


def class_names(config: dict) -> dict[int, str]:
    names = config.get("names")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError("YOLO YAML must define names as a list or mapping")


def resolve_split_sources(dataset_root: Path, value) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    sources = []
    for item in values:
        source = Path(str(item))
        if not source.is_absolute():
            source = dataset_root / source
        sources.append(source.resolve())
    return sources


def images_from_source(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if source.is_file() and source.suffix.lower() == ".txt":
        images = []
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            image = Path(line.strip())
            if not image.is_absolute():
                image = source.parent / image
            images.append(image.resolve())
        return images
    raise FileNotFoundError(f"YOLO split source does not exist or is unsupported: {source}")


def infer_label_path(image: Path) -> Path:
    parts = list(image.parts)
    image_positions = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if image_positions:
        parts[image_positions[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.parent.parent / "labels" / image.name.replace(image.suffix, ".txt")


def labels_for(label_path: Path, names: dict[int, str]) -> tuple[int, int]:
    fire = smoke = 0
    if not label_path.exists():
        return fire, smoke
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        class_name = names.get(int(fields[0]), "").strip().lower()
        fire |= "fire" in class_name or "flame" in class_name
        smoke |= "smoke" in class_name
    return int(fire), int(smoke)

