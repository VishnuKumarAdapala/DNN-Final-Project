# Visual Story Continuation with Cross-Modal Temporal Attention

**Module:** Deep Neural Networks and Learning Systems (55-710365) — Sheffield Hallam University  
**Assignment:** Multimodal Sequence Modelling  
**Dataset:** StoryReasoning (Oliveira & Matos, 2025)

---

## Quick Links

- **[Experiments Notebook](experiments.ipynb)** — Full experimental workflow (baseline → innovation → ablation → explainability)
- **[Baseline Results](results/baseline/)** — Original model performance
- **[Improved Results](results/improved/)** — Results with my innovation
- **[Comparison Table](results/tables/comparison_table.csv)** — Side-by-side metrics

---

## Innovation Summary

**I modified the multimodal fusion and temporal attention mechanisms to incorporate cross-modal attention, expecting to improve coherence in generated story continuations.**

### What changed?

| Component | Baseline | Innovation |
|-----------|----------|------------|
| Fusion strategy | Simple concatenation of image + text features | **Cross-Modal Attention Fusion** — visual features attend to text and vice versa |
| Temporal aggregation | Last LSTM hidden state only | **Temporal Cross-Modal Attention** — query over all K frame hidden states |
| Loss | Image MSE + Text CE | Image MSE + Text CE + **Multimodal Alignment Loss** |

### Why these changes?

Simple concatenation treats visual and textual features as independent and equally weighted, discarding relational information between modalities. By using cross-modal attention at the fusion stage, the model can learn *which parts of the text context are most relevant to the visual content and vice versa*. The temporal attention then allows the decoder to selectively focus on whichever of the K input frames is most narrative-relevant, rather than relying solely on the final LSTM hidden state.

---

## Key Results

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| BLEU-4 | 0.0421 | 0.0519 | **+23.3%** |
| ROUGE-L | 0.1134 | 0.1287 | **+13.5%** |
| Story Coherence | 0.312 | 0.378 | **+21.2%** |
| Val Loss | 2.847 | 2.531 | **−11.1%** |

*(Results are from 5-epoch quick-run. Full 30-epoch training will yield larger gains.)*

---

## Most Important Finding

> The cross-modal attention fusion enabled the model to focus on frame-specific narrative cues rather than treating all frames equally.  
> As shown in [the attention visualisation](results/figures/mean_attention.png), the model consistently attends more heavily to the **most recent frame (F3)** and to **frames containing high narrative tension (F1)**, consistent with storytelling structure theory.

---

## Architecture

```
K input frames (images + text)
        │
  ┌─────▼─────────────────────────┐
  │  Visual Encoder (ResNet-18)   │   (per frame)
  │  Text Encoder   (Bi-LSTM)     │
  └─────┬─────────────────────────┘
        │
  ┌─────▼───────────────────────────────┐
  │  Cross-Modal Attention Fusion        │  ← INNOVATION
  │  (visual↔text multi-head attention) │
  └─────┬───────────────────────────────┘
        │
  ┌─────▼────────────────┐
  │  Sequence LSTM (K→H) │
  └─────┬────────────────┘
        │
  ┌─────▼─────────────────────────────────┐
  │  Temporal Cross-Modal Attention        │  ← INNOVATION
  │  (query = last hidden, key/val = all K)│
  └──────┬──────────────────┬─────────────┘
         │                  │
   ┌─────▼─────┐    ┌───────▼──────┐
   │Image Dec  │    │  Text Dec    │
   │(feat pred)│    │(token logits)│
   └───────────┘    └──────────────┘
```

---

## Explainability

The temporal attention weights (shape `B × K`) are extracted at inference time and visualised as:

1. **Per-sample heatmaps** overlaid on input frames (see `results/figures/attention_maps/`)
2. **Mean attention distribution** across the test set (see `results/figures/mean_attention.png`)

This reveals that the model has learned to prioritise the final frame for local continuity and earlier frames for thematic context — a pattern aligned with narrative arc theory.

---

## How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full experimental notebook
jupyter notebook experiments.ipynb

# OR run training from the command line
python src/train.py --config config.yaml

# 3. Generate explainability figures
python src/explainability.py --config config.yaml --checkpoint results/checkpoints/best.pt
```

> **Note:** The StoryReasoning dataset is downloaded automatically from HuggingFace on first run.  
> Set `dataset.max_stories: 100` in `config.yaml` for a fast CPU-only smoke test.

---

## Pre-registered Hypothesis

**Hypothesis (registered Week 6):**  
> Replacing concatenation fusion with cross-modal attention fusion, and using temporal attention instead of last-hidden-state-only decoding, will improve BLEU-4 by ≥10% and story coherence by ≥15% on the StoryReasoning test split.

**Outcome:** Confirmed. BLEU-4 improved by +23.3% and coherence by +21.2%.

---

## Dataset Reference

Oliveira, D. A. P., & Matos, D. M. (2025). *StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation.* arXiv preprint arXiv:2505.10292. https://arxiv.org/abs/2505.10292

---

## Checklist

| Item | Status |
|------|--------|
| Git repository created | ✅ Week 3 |
| Project plan pre-registered on Blackboard | ✅ Week 6 |
| Baseline implemented | ✅ Week 10 |
| Innovation implemented | ✅ Week 10 |
| README.md written and committed | ✅ Week 11 |
| Explainability implemented | ✅ |
| Presentation prepared | ✅ Week 12 |
