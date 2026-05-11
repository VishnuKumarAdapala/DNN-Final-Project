"""
explainability.py – Attention visualisation for the temporal cross-modal attention.

Produces per-sample heatmaps showing which input frames the model
focused on when predicting the next story element.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms.functional as TF
from pathlib import Path


# ────────────────────────────────────────────────
# Un-normalise helper (for display)
# ────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def unnorm(t: torch.Tensor) -> torch.Tensor:
    """(3,H,W) normalised → (3,H,W) in [0,1]"""
    return (t * _STD + _MEAN).clamp(0., 1.)


# ────────────────────────────────────────────────
# Core visualisation
# ────────────────────────────────────────────────
def visualise_attention(model, batch: dict, device: torch.device,
                        save_dir: str = "results/figures/attention_maps",
                        n_samples: int = 5):
    """
    Run the model on `batch`, extract attention weights, and save figures.

    Each figure shows:
      Row 1: K input frames
      Row 2: Attention weight bar chart over K frames
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    images = batch["images"].to(device)   # (B, K, 3, H, W)
    tokens = batch["tokens"].to(device)   # (B, K, L)

    with torch.no_grad():
        outputs = model(images, tokens)

    attn_weights = outputs["attn_weights"].cpu()  # (B, K)
    B, K = images.shape[:2]
    n_samples = min(n_samples, B)

    saved_paths = []
    for i in range(n_samples):
        fig = plt.figure(figsize=(4 * K, 5), facecolor="#1a1a2e")
        gs  = gridspec.GridSpec(2, K, height_ratios=[3, 1.5], hspace=0.4)

        weights = attn_weights[i].numpy()   # (K,)
        # Softmax already applied inside MHA, but re-normalise for display
        weights = weights / (weights.sum() + 1e-8)

        for k in range(K):
            ax_img = fig.add_subplot(gs[0, k])
            img = unnorm(images[i, k].cpu()).permute(1, 2, 0).numpy()
            ax_img.imshow(img)
            ax_img.set_title(f"Frame {k+1}", color="white", fontsize=9)
            ax_img.axis("off")
            # Overlay attention as alpha tint
            alpha_overlay = np.ones((*img.shape[:2], 4))
            alpha_overlay[..., :3] = [1, 0.8, 0]    # yellow tint
            alpha_overlay[..., 3]  = weights[k] * 0.6
            ax_img.imshow(alpha_overlay, interpolation="nearest")

        # Bar chart row
        ax_bar = fig.add_subplot(gs[1, :])
        bar_colors = [plt.cm.YlOrRd(w) for w in weights]
        bars = ax_bar.bar(range(1, K + 1), weights, color=bar_colors, edgecolor="white",
                          linewidth=0.8)
        ax_bar.set_xticks(range(1, K + 1))
        ax_bar.set_xticklabels([f"F{k}" for k in range(1, K + 1)], color="white")
        ax_bar.set_ylabel("Attention\nWeight", color="white", fontsize=8)
        ax_bar.tick_params(colors="white")
        ax_bar.set_facecolor("#16213e")
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#444")

        fig.suptitle("Temporal Cross-Modal Attention — Which frames matter for story continuation?",
                     color="white", fontsize=10, y=1.02)

        path = os.path.join(save_dir, f"attn_sample_{i:03d}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        saved_paths.append(path)
        print(f"[Explainability] Saved → {path}")

    return saved_paths


# ────────────────────────────────────────────────
# Aggregate: mean attention across dataset
# ────────────────────────────────────────────────
def plot_mean_attention(all_weights: list, K: int,
                        save_path: str = "results/figures/mean_attention.png"):
    """
    all_weights: list of (K,) numpy arrays
    Plots the average attention weight per temporal position.
    """
    arr = np.stack(all_weights, axis=0)   # (N, K)
    mean_w = arr.mean(axis=0)
    std_w  = arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 3), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    x = np.arange(1, K + 1)
    ax.bar(x, mean_w, yerr=std_w, color="#e94560", edgecolor="white",
           linewidth=0.8, capsize=4, error_kw={"ecolor": "white", "linewidth": 1})
    ax.set_xlabel("Frame position", color="white")
    ax.set_ylabel("Mean attention weight", color="white")
    ax.set_title("Temporal Attention Distribution (mean ± std)", color="white", pad=12)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i}" for i in x], color="white")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Explainability] Mean attention saved → {save_path}")
    return save_path


# ────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from src.utils import load_config, get_device, load_checkpoint
    from src.model import MultimodalStoryModel
    from src.dataset import StoryReasoningDataset, collate_fn
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="results/checkpoints/best.pt")
    parser.add_argument("--n_samples",  type=int, default=10)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device()

    model = MultimodalStoryModel(cfg).to(device)
    load_checkpoint(args.checkpoint, model)

    ds     = StoryReasoningDataset(cfg)
    loader = DataLoader(ds, batch_size=args.n_samples, collate_fn=collate_fn, shuffle=True)
    batch  = next(iter(loader))

    # Compute next_img_feat target
    with torch.no_grad():
        batch["next_img_feat"] = model.encode_image(batch["next_frames"].to(device)).cpu()

    saved = visualise_attention(model, batch, device,
                                save_dir=cfg["explainability"]["save_dir"],
                                n_samples=args.n_samples)
    print(f"\n[Explainability] {len(saved)} figures saved.")
