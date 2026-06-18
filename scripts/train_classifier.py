#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from firevision.data import image_dataset
from firevision.device import best_device
from firevision.manifest import read_manifest
from firevision.models import build_classifier
from firevision.splits import validate_group_splits
from firevision.train import run_epoch, save_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="runs/classifier")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, cuda:0, ...")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    frame = read_manifest(args.manifest)
    validate_group_splits(frame)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), normalize]
    )
    loaders = {}
    for split in ("train", "val", "test"):
        subset = frame[frame["split"] == split].reset_index(drop=True)
        loaders[split] = DataLoader(
            image_dataset(subset, train_transform if split == "train" else eval_transform),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.workers,
            pin_memory=True,
        )

    device = best_device(args.device)
    print(f"device={device}")
    model = build_classifier(args.model, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, *_ = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        val_loss, truth, score, paths = run_epoch(model, loaders["val"], loss_fn, device)
        metrics = save_evaluation(output, "val", val_loss, truth, score, paths, args.threshold)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args), "metrics": metrics}, output / "best.pt")

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    loss, truth, score, paths = run_epoch(model, loaders["test"], loss_fn, device)
    save_evaluation(output, "test", loss, truth, score, paths, args.threshold)
    (output / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
