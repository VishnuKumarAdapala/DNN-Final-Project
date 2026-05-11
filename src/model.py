"""
model.py – Full multimodal sequence model.

Architecture:
  ┌──────────────┐   ┌──────────────┐
  │ Visual Enc   │   │  Text Enc    │   (per frame)
  │ CNN/ResNet   │   │  LSTM        │
  └──────┬───────┘   └──────┬───────┘
         │                  │
         └────── Fusion ─────┘
                   │
          Cross-Modal Attention  ◄── INNOVATION
                   │
           Sequence Model (LSTM)
                   │
            Temporal Attention   ◄── INNOVATION
                   │
        ┌──────────┴──────────┐
   Image Decoder         Text Decoder
   (feature prediction)  (auto-regressive)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from einops import rearrange


# ════════════════════════════════════════════════════════════
# 1. Visual Encoder  (Week 4)
# ════════════════════════════════════════════════════════════
class VisualEncoder(nn.Module):
    """CNN backbone (ResNet-18) → linear projection to feature_dim."""

    def __init__(self, feature_dim: int = 512, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        # Remove final classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B, 512, 1, 1)
        self.proj = nn.Linear(512, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, feature_dim)"""
        feat = self.backbone(x).squeeze(-1).squeeze(-1)   # (B, 512)
        return self.norm(self.proj(feat))                  # (B, feature_dim)


# ════════════════════════════════════════════════════════════
# 2. Text Encoder  (Week 6)
# ════════════════════════════════════════════════════════════
class TextEncoder(nn.Module):
    """Embedding + Bi-LSTM → mean-pooled representation."""

    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 512, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        tokens: (B, L) int
        Returns: (B, hidden_dim)
        """
        emb = self.embed(tokens)                          # (B, L, E)
        out, _ = self.lstm(emb)                           # (B, L, H)
        # Mean-pool over time
        feat = out.mean(dim=1)                            # (B, H)
        return self.norm(self.proj(feat))


# ════════════════════════════════════════════════════════════
# 3. Cross-Modal Attention Fusion  (Week 3 + INNOVATION)
# ════════════════════════════════════════════════════════════
class CrossModalAttentionFusion(nn.Module):
    """
    INNOVATION: Instead of simple concatenation, we use cross-modal
    multi-head attention so visual and textual streams can query
    each other's content, producing a richer joint representation.

    img_feat attends to text_feat  → v→t context
    text_feat attends to img_feat  → t→v context
    Concatenate both contexts → project to fusion_dim.
    """

    def __init__(self, in_dim: int = 512, fusion_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Project both modalities to same dim before attention
        self.img_proj  = nn.Linear(in_dim, fusion_dim)
        self.text_proj = nn.Linear(in_dim, fusion_dim)

        # Two cross-attention blocks
        self.v2t_attn = nn.MultiheadAttention(fusion_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        self.t2v_attn = nn.MultiheadAttention(fusion_dim, num_heads,
                                              dropout=dropout, batch_first=True)

        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor):
        """
        img_feat:  (B, D)
        text_feat: (B, D)
        Returns:   (B, fusion_dim)
        """
        # Unsqueeze to sequence length 1 for MHA
        iv = self.img_proj(img_feat).unsqueeze(1)    # (B, 1, F)
        tv = self.text_proj(text_feat).unsqueeze(1)  # (B, 1, F)

        v2t, _ = self.v2t_attn(iv, tv, tv)           # visual queries text
        t2v, _ = self.t2v_attn(tv, iv, iv)           # text queries visual

        fused = torch.cat([v2t.squeeze(1), t2v.squeeze(1)], dim=-1)  # (B, 2F)
        return self.out_proj(fused)                                    # (B, F)


# Simple fallback: concatenation fusion
class ConcatFusion(nn.Module):
    def __init__(self, in_dim: int = 512, fusion_dim: int = 512, **kwargs):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim * 2, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

    def forward(self, img_feat, text_feat):
        return self.proj(torch.cat([img_feat, text_feat], dim=-1))


# ════════════════════════════════════════════════════════════
# 4. Sequence Model  (Week 7)
# ════════════════════════════════════════════════════════════
class SequenceModel(nn.Module):
    """LSTM / GRU over K fused frame representations."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3,
                 model_type: str = "lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor):
        """x: (B, K, D) → all_hidden (B, K, H), last_hidden (B, H)"""
        out, _ = self.rnn(x)                 # (B, K, H)
        return out, out[:, -1, :]            # all_steps, last_step


# ════════════════════════════════════════════════════════════
# 5. Temporal Cross-Modal Attention  (Week 8 + INNOVATION)
# ════════════════════════════════════════════════════════════
class TemporalCrossModalAttention(nn.Module):
    """
    INNOVATION: Temporal attention that computes a query from the
    *last* fused frame and attends over *all* K sequence hidden states,
    weighting earlier frames by their relevance to the predicted next frame.

    This gives the decoder explicit access to the most narrative-relevant
    context rather than relying solely on the final LSTM hidden state.
    """

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_hidden: torch.Tensor, query: torch.Tensor):
        """
        seq_hidden: (B, K, H) – all LSTM outputs (keys + values)
        query:      (B, H)    – last-step hidden (query)
        Returns:
          context:  (B, H)   – attended context
          weights:  (B, K)   – attention weights (for explainability)
        """
        q = query.unsqueeze(1)                             # (B, 1, H)
        ctx, weights = self.attn(q, seq_hidden, seq_hidden)  # (B, 1, H), (B, 1, K)
        ctx = self.dropout(ctx.squeeze(1))                 # (B, H)
        ctx = self.norm(ctx + query)                       # residual
        return ctx, weights.squeeze(1)                     # (B, H), (B, K)


# ════════════════════════════════════════════════════════════
# 6. Image Decoder  (Week 9)
# ════════════════════════════════════════════════════════════
class ImageDecoder(nn.Module):
    """Predict next-frame feature vector (MSE target = actual visual encoder output)."""

    def __init__(self, latent_dim: int = 512, feature_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, feature_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, feature_dim)"""
        return self.net(z)


# ════════════════════════════════════════════════════════════
# 7. Text Decoder  (Week 10)
# ════════════════════════════════════════════════════════════
class TextDecoder(nn.Module):
    """Single-step token prediction head (teacher-forced training)."""

    def __init__(self, hidden_dim: int = 512, vocab_size: int = 10000,
                 dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, H) → logits (B, vocab_size)"""
        return self.fc(context)


# ════════════════════════════════════════════════════════════
# 8. Full Model
# ════════════════════════════════════════════════════════════
class MultimodalStoryModel(nn.Module):
    """
    End-to-end multimodal sequence model for visual story continuation.

    Innovation: Cross-Modal Attention Fusion + Temporal Cross-Modal Attention.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        vc = cfg["model"]["visual_encoder"]
        tc = cfg["model"]["text_encoder"]
        fc = cfg["model"]["fusion"]
        sc = cfg["model"]["sequence_model"]
        ac = cfg["model"]["attention"]
        idc = cfg["model"]["image_decoder"]
        tdc = cfg["model"]["text_decoder"]

        # Sub-modules
        self.visual_enc = VisualEncoder(
            feature_dim=vc["feature_dim"],
            pretrained=vc["pretrained"],
            freeze=vc["freeze_backbone"],
        )
        self.text_enc = TextEncoder(
            vocab_size=tc["vocab_size"],
            embed_dim=tc["embed_dim"],
            hidden_dim=tc["hidden_dim"],
            num_layers=tc["num_layers"],
            dropout=tc["dropout"],
        )
        if fc["strategy"] == "cross_modal_attention":
            self.fusion = CrossModalAttentionFusion(
                in_dim=vc["feature_dim"],
                fusion_dim=fc["fusion_dim"],
                num_heads=fc["num_heads"],
                dropout=fc["dropout"],
            )
        else:
            self.fusion = ConcatFusion(
                in_dim=vc["feature_dim"],
                fusion_dim=fc["fusion_dim"],
            )

        self.seq_model = SequenceModel(
            input_dim=fc["fusion_dim"],
            hidden_dim=sc["hidden_dim"],
            num_layers=sc["num_layers"],
            dropout=sc["dropout"],
            model_type=sc["type"],
        )
        self.temporal_attn = TemporalCrossModalAttention(
            hidden_dim=ac.get("hidden_dim", sc["hidden_dim"]),
            num_heads=ac["num_heads"],
            dropout=ac["dropout"],
        )
        self.img_decoder  = ImageDecoder(latent_dim=sc["hidden_dim"],
                                         feature_dim=vc["feature_dim"])
        self.text_decoder = TextDecoder(hidden_dim=sc["hidden_dim"],
                                        vocab_size=tc["vocab_size"])

        # Alignment projection (for contrastive loss)
        self.align_proj = nn.Linear(vc["feature_dim"], sc["hidden_dim"])

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        """
        images: (B, K, 3, H, W)  – K input frames
        tokens: (B, K, L)        – K input text sequences
        Returns dict with predicted features and logits.
        """
        B, K = images.shape[:2]

        # Encode each frame independently
        fused_frames = []
        for k in range(K):
            img_feat  = self.visual_enc(images[:, k])          # (B, D)
            text_feat = self.text_enc(tokens[:, k])            # (B, D)
            fused     = self.fusion(img_feat, text_feat)       # (B, F)
            fused_frames.append(fused)

        seq_in  = torch.stack(fused_frames, dim=1)            # (B, K, F)
        seq_out, last_hidden = self.seq_model(seq_in)         # (B,K,H), (B,H)

        context, attn_weights = self.temporal_attn(seq_out, last_hidden)
        # attn_weights: (B, K) – used for explainability

        pred_img_feat  = self.img_decoder(context)            # (B, visual_dim)
        pred_text_logits = self.text_decoder(context)         # (B, vocab_size)

        return {
            "pred_img_feat":    pred_img_feat,
            "pred_text_logits": pred_text_logits,
            "attn_weights":     attn_weights,    # (B, K) for explainability
            "context":          context,
        }

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Utility: encode a single image. (B,3,H,W) → (B,D)"""
        return self.visual_enc(image)
