from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import multilabel_metrics


def run_epoch(model, loader, loss_fn, device, optimizer=None, *, video_layout=False):
    import torch

    training = optimizer is not None
    model.train(training)
    losses, truths, scores, paths = [], [], [], []
    for inputs, targets, batch_paths in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if video_layout:
            inputs = inputs.permute(0, 2, 1, 3, 4)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        truths.append(targets.detach().cpu().numpy())
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        paths.extend(batch_paths)
    return float(np.mean(losses)), np.concatenate(truths), np.concatenate(scores), paths


def save_evaluation(output_dir, split, loss, truth, score, paths, threshold=0.5):
    import pandas as pd

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = multilabel_metrics(truth, score, ["fire", "smoke"], threshold)
    metrics["loss"] = loss
    (output / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(
        {
            "path": paths,
            "fire_true": truth[:, 0],
            "smoke_true": truth[:, 1],
            "fire_score": score[:, 0],
            "smoke_score": score[:, 1],
        }
    ).to_csv(output / f"{split}_predictions.csv", index=False)
    return metrics

