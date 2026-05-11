"""
utils.py – Shared utilities: config loading, reproducibility, logging, metrics.
"""
import os
import random
import yaml
import numpy as np
import torch
from pathlib import Path


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_bleu(references, hypotheses):
    """Sentence-level BLEU-4 averaged across examples."""
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk.download("punkt", quiet=True)
    smoother = SmoothingFunction().method1
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def compute_rouge_l(references, hypotheses):
    """ROUGE-L F1 averaged across examples."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
    return float(np.mean(scores)) if scores else 0.0


def compute_image_mse(pred_features, target_features):
    """MSE between predicted and target image feature vectors (used as proxy metric)."""
    if isinstance(pred_features, torch.Tensor):
        return torch.nn.functional.mse_loss(pred_features, target_features).item()
    return float(np.mean((np.array(pred_features) - np.array(target_features)) ** 2))


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[Checkpoint] Loaded ← {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ─────────────────────────────────────────────
# Story-coherence metric (novel evaluation innovation)
# ─────────────────────────────────────────────
def compute_story_coherence(texts: list[str]) -> float:
    """
    Novel metric: semantic similarity between consecutive generated texts
    as a proxy for narrative coherence.  Returns mean cosine similarity.
    Higher = more coherent story continuation.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if len(texts) < 2:
        return 1.0
    vec = TfidfVectorizer().fit_transform(texts)
    sims = []
    for i in range(len(texts) - 1):
        sim = cosine_similarity(vec[i], vec[i + 1])[0][0]
        sims.append(sim)
    return float(np.mean(sims))
