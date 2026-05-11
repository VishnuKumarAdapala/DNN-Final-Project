"""
dataset.py – StoryReasoning dataset wrapper.

Downloads from HuggingFace Hub: daniel3303/StoryReasoning
Each sample is a story with K+1 frames; we use K as input, predict K+1.
"""
import re
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from collections import Counter


# ────────────────────────────────────────────────
# Simple character-level tokeniser (no external NLP deps needed)
# ────────────────────────────────────────────────
class SimpleTokenizer:
    """Word-level tokenizer built from training corpus."""

    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word2idx = {w: i for i, w in enumerate(self.SPECIAL)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def _tokenize(self, text: str):
        return re.findall(r"\b\w+\b", text.lower())

    def build_vocab(self, texts: list[str]):
        counter = Counter()
        for t in texts:
            counter.update(self._tokenize(t))
        for word, _ in counter.most_common(self.vocab_size - len(self.SPECIAL)):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = 128) -> torch.Tensor:
        tokens = [self.BOS] + [
            self.word2idx.get(w, self.UNK) for w in self._tokenize(text)
        ] + [self.EOS]
        tokens = tokens[:max_len]
        tokens += [self.PAD] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, ids) -> str:
        words = []
        for i in ids:
            i = int(i)
            if i in (self.PAD, self.BOS):
                continue
            if i == self.EOS:
                break
            words.append(self.idx2word.get(i, "<UNK>"))
        return " ".join(words)


# ────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────
class StoryReasoningDataset(Dataset):
    """
    Wraps the HuggingFace StoryReasoning dataset.

    Each __getitem__ returns:
      images:       (K, 3, H, W)  – input frames
      tokens:       (K, L)        – tokenised input descriptions
      next_img_feat:  placeholder – will be filled during forward pass
                      (actual target = visual_enc(next_frame))
      next_frame:   (3, H, W)    – raw next frame for target encoding
      next_token:   scalar long  – first token of next description (text target)
      next_text:    str          – full next description (for eval)
    """

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(self, cfg: dict, split: str = "train"):
        from datasets import load_dataset
        ds_cfg = cfg["dataset"]
        self.seq_len   = ds_cfg["sequence_length"]
        self.max_text  = ds_cfg["max_text_length"]
        self.img_size  = ds_cfg["image_size"]

        print(f"[Dataset] Loading StoryReasoning from HuggingFace …")
        raw = load_dataset(ds_cfg["hf_path"], split="train", trust_remote_code=True)

        # Optionally subsample for fast experiments
        if ds_cfg.get("max_stories"):
            raw = raw.select(range(min(ds_cfg["max_stories"], len(raw))))

        self.stories = list(raw)
        print(f"[Dataset] Loaded {len(self.stories)} stories.")

        # Build tokenizer from all text in the dataset
        self.tokenizer = SimpleTokenizer(vocab_size=ds_cfg.get("vocab_size",
                                         cfg["model"]["text_encoder"]["vocab_size"]))
        all_texts = []
        for s in self.stories:
            all_texts.extend(self._get_texts(s))
        self.tokenizer.build_vocab(all_texts)
        print(f"[Dataset] Vocab size: {len(self.tokenizer.word2idx)}")

        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=self.IMG_MEAN, std=self.IMG_STD),
        ])

        # Filter stories that have enough frames
        self.stories = [s for s in self.stories if self._count_frames(s) >= self.seq_len + 1]
        print(f"[Dataset] Stories with ≥{self.seq_len+1} frames: {len(self.stories)}")

    # ── helpers ──────────────────────────────────
    def _count_frames(self, story: dict) -> int:
        """Count how many image frames are in a story entry."""
        # StoryReasoning stores images as a list under 'images' key
        imgs = story.get("images", [])
        return len(imgs) if isinstance(imgs, list) else 0

    def _get_texts(self, story: dict) -> list[str]:
        """Extract all text descriptions from a story."""
        texts = []
        # 'story' field contains the narrative text; 'scene_analyses' per frame
        if isinstance(story.get("story"), str):
            texts.append(story["story"])
        if isinstance(story.get("scene_analyses"), list):
            for sa in story["scene_analyses"]:
                if isinstance(sa, str):
                    texts.append(sa)
                elif isinstance(sa, dict):
                    texts.append(str(sa))
        return texts or [""]

    def _get_frame_text(self, story: dict, idx: int) -> str:
        """Return text for frame idx."""
        analyses = story.get("scene_analyses", [])
        if isinstance(analyses, list) and idx < len(analyses):
            sa = analyses[idx]
            return sa if isinstance(sa, str) else str(sa)
        # fallback: use portion of story text
        return story.get("story", "")[:200]

    def _load_image(self, img_obj) -> torch.Tensor:
        """img_obj may be a PIL Image or a dict with 'bytes'."""
        if isinstance(img_obj, Image.Image):
            pil = img_obj.convert("RGB")
        elif isinstance(img_obj, dict) and "bytes" in img_obj:
            import io
            pil = Image.open(io.BytesIO(img_obj["bytes"])).convert("RGB")
        else:
            # Create a blank image as fallback
            pil = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))
        return self.transform(pil)

    # ── main interface ────────────────────────────
    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx: int) -> dict:
        story = self.stories[idx]
        imgs_raw = story["images"]                          # list of image objects

        # Pick K input frames + 1 target frame
        K    = self.seq_len
        imgs = [self._load_image(imgs_raw[k]) for k in range(K)]      # K×(3,H,W)
        next_frame = self._load_image(imgs_raw[K])                     # (3,H,W)

        tokens = [
            self.tokenizer.encode(self._get_frame_text(story, k), self.max_text)
            for k in range(K)
        ]
        next_text   = self._get_frame_text(story, K)
        next_tok_id = self.tokenizer.encode(next_text, self.max_text)[1]  # first real token

        return {
            "images":      torch.stack(imgs, dim=0),        # (K, 3, H, W)
            "tokens":      torch.stack(tokens, dim=0),      # (K, L)
            "next_frame":  next_frame,                       # (3, H, W)
            "next_token":  next_tok_id,                     # scalar
            "next_text":   next_text,
        }


# ────────────────────────────────────────────────
# Collate
# ────────────────────────────────────────────────
def collate_fn(batch: list) -> dict:
    """
    Custom collate: stack tensors, keep next_img_feat as zeros
    (actual target features are computed in train loop via model.encode_image).
    """
    images      = torch.stack([b["images"]     for b in batch])   # (B,K,3,H,W)
    tokens      = torch.stack([b["tokens"]     for b in batch])   # (B,K,L)
    next_frames = torch.stack([b["next_frame"] for b in batch])   # (B,3,H,W)
    next_tokens = torch.tensor([b["next_token"] for b in batch], dtype=torch.long)  # (B,)
    next_texts  = [b["next_text"] for b in batch]

    return {
        "images":      images,
        "tokens":      tokens,
        "next_frames": next_frames,       # raw; model encodes during training
        "next_token":  next_tokens,
        "next_texts":  next_texts,
        # next_img_feat will be filled in training loop using model.encode_image
        "next_img_feat": torch.zeros(len(batch), 512),   # placeholder
    }
