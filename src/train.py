"""
train.py – Training script for the multimodal story model.

Run:
    python src/train.py --config config.yaml
"""
import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model import MultimodalStoryModel
from src.utils import load_config, set_seed, get_device, save_checkpoint, compute_bleu


# ────────────────────────────────────────────────
# Dataset  (import lazily to avoid hard dependency at import time)
# ────────────────────────────────────────────────
def get_dataloaders(cfg: dict):
    from src.dataset import StoryReasoningDataset, collate_fn
    ds_cfg = cfg["dataset"]
    full_ds = StoryReasoningDataset(cfg)
    n = len(full_ds)
    n_train = int(n * ds_cfg["train_split"])
    n_val   = int(n * ds_cfg["val_split"])
    n_test  = n - n_train - n_val
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["project"]["seed"])
    )
    mk = lambda ds, shuffle: DataLoader(
        ds, batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle, collate_fn=collate_fn,
        num_workers=ds_cfg["num_workers"], pin_memory=True
    )
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)


# ────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Three objectives:
      1. Image feature reconstruction  (MSE)
      2. Text token prediction         (Cross-entropy)
      3. Multimodal alignment          (Contrastive / cosine)   ← INNOVATION
    """

    def __init__(self, weights: dict):
        super().__init__()
        self.w_img   = weights["image_reconstruction"]
        self.w_text  = weights["text_generation"]
        self.w_align = weights["multimodal_alignment"]

    def forward(self, outputs: dict, targets: dict) -> dict:
        # 1. Image feature MSE
        img_loss = F.mse_loss(outputs["pred_img_feat"], targets["next_img_feat"])

        # 2. Text cross-entropy  (predict first token of next description as proxy)
        text_loss = F.cross_entropy(
            outputs["pred_text_logits"],
            targets["next_token"],
            ignore_index=0,
        )

        # 3. Alignment: cosine similarity between predicted image feature
        #    and aligned text context (maximise agreement)
        pred_img_norm  = F.normalize(outputs["pred_img_feat"], dim=-1)
        context_norm   = F.normalize(outputs["context"], dim=-1)
        align_loss = 1.0 - (pred_img_norm * context_norm).sum(dim=-1).mean()

        total = (self.w_img * img_loss
                 + self.w_text * text_loss
                 + self.w_align * align_loss)

        return {
            "total":     total,
            "img_loss":  img_loss.item(),
            "text_loss": text_loss.item(),
            "align_loss": align_loss.item(),
        }


# ────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    totals = {"total": 0., "img_loss": 0., "text_loss": 0., "align_loss": 0.}
    n = 0
    for batch in loader:
        images = batch["images"].to(device)      # (B, K, 3, H, W)
        tokens = batch["tokens"].to(device)      # (B, K, L)
        targets = {
            "next_img_feat": batch["next_img_feat"].to(device),   # (B, D)
            "next_token":    batch["next_token"].to(device),      # (B,)
        }
        optimizer.zero_grad()
        outputs = model(images, tokens)
        losses  = criterion(outputs, targets)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        b = images.size(0)
        n += b
        for k in totals:
            totals[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()

    return {k: v / max(len(loader), 1) for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    totals = {"total": 0., "img_loss": 0., "text_loss": 0., "align_loss": 0.}
    for batch in loader:
        images = batch["images"].to(device)
        tokens = batch["tokens"].to(device)
        targets = {
            "next_img_feat": batch["next_img_feat"].to(device),
            "next_token":    batch["next_token"].to(device),
        }
        outputs = model(images, tokens)
        losses  = criterion(outputs, targets)
        for k in totals:
            totals[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
    return {k: v / max(len(loader), 1) for k, v in totals.items()}


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    device = get_device()
    print(f"[Train] Device: {device}")

    os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    print(f"[Train] Splits → train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    model     = MultimodalStoryModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"],
        eta_min=1e-6,
    )
    criterion = MultiTaskLoss(cfg["training"]["loss_weights"])

    history = []
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion,
                                    device, cfg["training"]["grad_clip"])
        val_metrics   = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
              f"train_loss={train_metrics['total']:.4f} | "
              f"val_loss={val_metrics['total']:.4f} | "
              f"img={val_metrics['img_loss']:.4f} | "
              f"text={val_metrics['text_loss']:.4f} | "
              f"align={val_metrics['align_loss']:.4f} | "
              f"{elapsed:.1f}s")

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            patience_counter = 0
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "val_loss": best_val},
                os.path.join(cfg["logging"]["checkpoint_dir"], "best.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                print(f"[Train] Early stopping at epoch {epoch}")
                break

    # Save history
    with open(os.path.join(cfg["logging"]["log_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("[Train] Done. History saved.")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
