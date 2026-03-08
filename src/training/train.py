"""
Training loop: two-view contrastive with VICReg.
DataLoader is assumed to yield (view1_bins, view1_dist_ids, view2_bins, view2_dist_ids)
with aligned batch: view1[i] and view2[i] are the same neuron (match_idx = arange(B)).
"""

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml


def train_one_epoch(
    encoder: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    encoder.train()
    criterion.train()
    total_loss = 0.0
    total_inv = 0.0
    total_var = 0.0
    total_cov = 0.0
    n_batches = 0

    for batch in data_loader:
        (
            view1_bins,
            view1_dist_ids,
            view2_bins,
            view2_dist_ids,
        ) = batch[:4]
        B = view1_bins.size(0)
        view1_bins = view1_bins.to(device)
        view1_dist_ids = view1_dist_ids.to(device)
        view2_bins = view2_bins.to(device)
        view2_dist_ids = view2_dist_ids.to(device)
        match_idx = torch.stack([
            torch.arange(B, device=device),
            torch.arange(B, device=device),
        ])

        h1 = encoder(view1_bins, view1_dist_ids)
        h2 = encoder(view2_bins, view2_dist_ids)
        loss, loss_dict = criterion(h1, h2, match_idx)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            params = list(encoder.parameters()) + list(criterion.parameters())
            torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_inv += loss_dict["inv"]
        total_var += loss_dict["var"]
        total_cov += loss_dict["cov"]
        n_batches += 1

    n_batches = max(n_batches, 1)
    return {
        "loss": total_loss / n_batches,
        "inv": total_inv / n_batches,
        "var": total_var / n_batches,
        "cov": total_cov / n_batches,
    }


def run_training(
    encoder: nn.Module,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    device: torch.device,
    checkpoint_dir: Optional[Path] = None,
) -> None:
    train_cfg = config.get("train", {})
    lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    num_epochs = train_cfg.get("num_epochs", 100)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.05)

    params = list(encoder.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    warmup_steps = int(warmup_ratio * num_epochs * steps_per_epoch)

    for epoch in range(num_epochs):
        global_step = epoch * steps_per_epoch
        if global_step < warmup_steps:
            scale = (global_step + 1) / warmup_steps
            for g in optimizer.param_groups:
                g["lr"] = lr * scale
        metrics = train_one_epoch(
            encoder, criterion, optimizer, device, train_loader, grad_clip
        )
        print(f"Epoch {epoch + 1}/{num_epochs}  loss={metrics['loss']:.4f}  "
              f"inv={metrics['inv']:.4f}  var={metrics['var']:.4f}  cov={metrics['cov']:.4f}")
        if checkpoint_dir and (epoch + 1) % 10 == 0:
            ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
