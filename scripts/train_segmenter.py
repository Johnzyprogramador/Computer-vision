#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from firevision.metrics import binary_metrics
from firevision.device import best_device


class SegmentationDataset(Dataset):
    def __init__(self, frame, size):
        self.frame, self.size = frame, size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(row["image"]).convert("RGB").resize((self.size, self.size))
        mask = Image.open(row["mask"]).convert("L").resize(
            (self.size, self.size), resample=Image.Resampling.NEAREST
        )
        return TF.to_tensor(image), (TF.pil_to_tensor(mask) > 0).float(), row["image"]


def main():
    parser = argparse.ArgumentParser(description="Train U-Net/DeepLab-style smoke/fire masks")
    parser.add_argument("--manifest", required=True, help="CSV: image,mask,split,group_id")
    parser.add_argument("--architecture", choices=["unet", "deeplabv3plus"], default="unet")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="runs/segmentation")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, cuda:0, ...")
    args = parser.parse_args()
    import pandas as pd
    import segmentation_models_pytorch as smp

    frame = pd.read_csv(args.manifest)
    leaked = frame.groupby("group_id")["split"].nunique()
    if (leaked > 1).any():
        raise ValueError("group_id leakage across segmentation splits")
    model_class = smp.Unet if args.architecture == "unet" else smp.DeepLabV3Plus
    model = model_class(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=1)
    device = best_device(args.device)
    print(f"device={device}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
    loaders = {
        split: DataLoader(
            SegmentationDataset(frame[frame.split == split].reset_index(drop=True), args.image_size),
            batch_size=args.batch_size,
            shuffle=split == "train",
        )
        for split in ("train", "val", "test")
    }
    output, best = Path(args.output), float("inf")
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        for images, masks, _ in loaders["train"]:
            loss = loss_fn(model(images.to(device)), masks.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for images, masks, _ in loaders["val"]:
                losses.append(loss_fn(model(images.to(device)), masks.to(device)).item())
        val_loss = float(np.mean(losses))
        print(f"epoch={epoch + 1} val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), output / "best.pt")

    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    truth, scores = [], []
    model.eval()
    with torch.no_grad():
        for images, masks, _ in loaders["test"]:
            truth.extend(masks.numpy().reshape(-1).astype(int))
            scores.extend(torch.sigmoid(model(images.to(device))).cpu().numpy().reshape(-1))
    report = binary_metrics(truth, scores, threshold=args.threshold).to_dict()
    (output / "test_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
