"""
emotion_steering_gnn_train.py  —  v3
═══════════════════════════════════════════════════════════════════════════════
Self-contained pipeline.  No command-line arguments needed — edit the CONFIG
block below and run:

    python emotion_steering_gnn_train.py

What it does, in order:
  1. Downloads + extracts RAVDESS (if not already present)
  2. Extracts temporal acoustic features from every relevant WAV
  3. Trains EmotionConditionedGNN on (neutral_feat, emotion_label) → delta
  4. Saves the best checkpoint to GNN_CKPT_DIR
  5. Loads F5-TTS, generates neutral audio for TARGET_TEXT
  6. For each of happy / sad / angry, steers the neutral audio via the GNN
     delta and saves all four WAVs to ./Output/GNN/

Architecture fixes (vs original):
  F1  No supervision leakage — input is (neutral_feat, emotion_label), never diff
  F2  Adjacency identified via auxiliary emotion classification loss
  F3  Emotion-conditioned adjacency: A_e = softmax(MLP(emo_emb))
  F4  Temporal bin features [mean, std] × 10 bins, shape (5, 20)
  F5  DSP operations semantically aligned with feature definitions
  F6  Content-node preservation loss on rate + voice_quality nodes
  F7  EmotionPrototypes normalises out RAVDESS acted-speech magnitude bias;
      intensity scalar at inference controls steering strength
"""

import os
import sys
import random
import warnings
import zipfile
import requests
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG  — edit this block, then just run the file
# ═══════════════════════════════════════════════════════════════════════════════

# ── F5-TTS paths ──────────────────────────────────────────────────────────────
F5_ROOT    = Path("./F5-TTS")
CKPT_PATH  = F5_ROOT / "ckpts" / "F5TTS_v1_Base" / "model_1250000.safetensors"
VOCAB_FILE = F5_ROOT / "ckpts" / "F5TTS_v1_Base" / "vocab.txt"

# ── RAVDESS ───────────────────────────────────────────────────────────────────
RAVDESS_DIR     = "./EmoSteer/data_ravdess/ravdess"          # where to extract
RAVDESS_ZIP     = "./data/ravdess.zip"      # where to download the zip

# ── Text to generate ──────────────────────────────────────────────────────────
TARGET_TEXT     = "Hello, I am a voice model demonstrating emotional speech synthesis."

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS          = 80
BATCH_SIZE      = 16
LR              = 1e-3
W_SMOOTH        = 0.05     # temporal smoothness weight
W_CONTENT       = 0.10     # content-node preservation weight
W_AUX           = 0.30     # auxiliary classification weight

# ── Inference ─────────────────────────────────────────────────────────────────
# 0 = unchanged, 1 = full RAVDESS-scale steering; 0.5–0.7 suits TTS outputs
STEERING_INTENSITY = 0.65
NFE_STEP           = 32    # CFM solver steps (>=32 for stable trajectories)

# ── Output paths ──────────────────────────────────────────────────────────────
GNN_CKPT_DIR    = Path("./gnn_ckpts")
OUTPUT_DIR      = Path("./Output/GNN_Steer3")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Acoustic constants
# ═══════════════════════════════════════════════════════════════════════════════
EMOTION_MAP: Dict[str, str] = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}
RAVDESS_TEXT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}
STEER_EMOTIONS  = ["happy", "sad", "angry"]   # emotions to generate output for
EMOTIONS        = ["neutral"] + STEER_EMOTIONS
N_EMOTIONS      = len(EMOTIONS)
EMO2IDX         = {e: i for i, e in enumerate(EMOTIONS)}

NODE_NAMES      = ["pitch", "energy", "rate", "spectral", "voice_quality"]
N_NODES         = 5
T_BINS          = 10           # temporal bins per node
NODE_DIM        = T_BINS * 2   # [mean, std] per bin → 20 per node
EMO_EMB_DIM     = 16
TARGET_SR       = 16_000
CONTENT_NODES   = [2, 4]       # rate, voice_quality — content-sensitive


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  RAVDESS download
# ═══════════════════════════════════════════════════════════════════════════════
def download_ravdess() -> str:
    """
    Downloads and extracts RAVDESS speech-only files if not already present.
    Returns the path to the extracted root directory.
    Mirrors the EmoSteer download pattern exactly.
    """
    os.makedirs(os.path.dirname(RAVDESS_DIR), exist_ok=True)

    if not os.path.exists(RAVDESS_DIR):
        print("Downloading RAVDESS speech dataset …")
        url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(RAVDESS_ZIP, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        print(f"\r  {pct:.1f}%", end="", flush=True)
        print("\nDownload complete.")
    else:
        print(f"[RAVDESS] already exists: {RAVDESS_DIR}")

    extract_root = RAVDESS_DIR
    if not os.path.exists(extract_root):
        print("Extracting RAVDESS …")
        os.makedirs(extract_root, exist_ok=True)
        with zipfile.ZipFile(RAVDESS_ZIP, "r") as zf:
            zf.extractall(extract_root)
        print("Extraction complete.")
    else:
        print(f"[RAVDESS] Already extracted: {extract_root}")

    return extract_root

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Temporal Feature Extractor  →  (N_NODES, NODE_DIM)           [F4]
# ═══════════════════════════════════════════════════════════════════════════════
def _bin_stats(signal_1d: np.ndarray, n_bins: int = T_BINS) -> np.ndarray:
    """
    Split signal_1d into n_bins equal chunks.
    Return [mean, std] per chunk — shape (n_bins * 2,), always finite.
    """
    out   = np.zeros(n_bins * 2, dtype=np.float32)
    if len(signal_1d) == 0:
        return out
    chunk = max(1, len(signal_1d) // n_bins)
    for k in range(n_bins):
        seg = signal_1d[k * chunk : (k + 1) * chunk]
        seg = seg[np.isfinite(seg)]
        if len(seg) == 0:
            continue
        out[2 * k]     = float(seg.mean())
        out[2 * k + 1] = float(seg.std())
    return out


def extract_features(wav_path: str) -> torch.Tensor:
    """
    Load a WAV and return a (N_NODES, NODE_DIM) = (5, 20) float32 tensor.

    Node 0 – pitch         : 10-bin [mean F0, std F0] trajectory (Hz, voiced)
    Node 1 – energy        : 10-bin [mean RMS, std RMS] trajectory
    Node 2 – rate          : 10-bin [voiced-fraction, energy-variance] proxy
    Node 3 – spectral      : 10-bin [mean centroid, std centroid] (Hz)
    Node 4 – voice_quality : 10-bin [mean ZCR, std ZCR]

    Bin trajectories preserve contour shape (F0 arc, energy dynamics, etc.)
    which is the primary perceptual carrier of emotion.                   (F4)
    """
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    wav = waveform.squeeze(0).numpy()
    sr  = TARGET_SR

    frame_len = int(0.025 * sr)   # 25 ms
    hop_len   = int(0.010 * sr)   # 10 ms

    def make_frames(sig: np.ndarray) -> np.ndarray:
        nf  = max(1, (len(sig) - frame_len) // hop_len + 1)
        idx = np.arange(frame_len)[None, :] + hop_len * np.arange(nf)[:, None]
        idx = np.clip(idx, 0, len(sig) - 1)
        return sig[idx]   # (nf, frame_len)

    frms = make_frames(wav)
    win  = np.hanning(frame_len)

    # ── Node 0: F0 via autocorrelation ────────────────────────────────────
    f0_frames = []
    for fr in frms:
        w  = fr * win
        ac = np.correlate(w, w, mode="full")[frame_len - 1:]
        lo = max(1, int(sr / 500))
        hi = min(int(sr / 50), len(ac) - 1)
        if lo >= hi:
            f0_frames.append(0.0)
            continue
        pk = int(np.argmax(ac[lo:hi])) + lo
        f0_frames.append(float(sr / pk) if pk > 0 else 0.0)
    f0     = np.array(f0_frames, dtype=np.float32)
    voiced = f0 > 50
    f0_v   = np.where(voiced, f0, 0.0)
    node0  = _bin_stats(f0_v)

    # ── Node 1: RMS energy ────────────────────────────────────────────────
    rms   = np.sqrt((frms ** 2).mean(axis=1)).astype(np.float32)
    node1 = _bin_stats(rms)

    # ── Node 2: Rate proxy (voiced-fraction + energy-variance per bin) ────
    chunk = max(1, len(voiced) // T_BINS)
    node2 = np.zeros(T_BINS * 2, dtype=np.float32)
    for k in range(T_BINS):
        v_seg   = voiced[k * chunk : (k + 1) * chunk]
        rms_seg = rms[k * chunk : (k + 1) * chunk]
        node2[2 * k]     = float(v_seg.mean())  if len(v_seg)   > 0 else 0.0
        node2[2 * k + 1] = float(rms_seg.var()) if len(rms_seg) > 0 else 0.0

    # ── Node 3: Spectral centroid ─────────────────────────────────────────
    spec  = np.abs(np.fft.rfft(frms * win[None, :], axis=1))
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr).astype(np.float32)
    denom = spec.sum(axis=1) + 1e-8
    cent  = (spec * freqs[None, :]).sum(axis=1) / denom
    node3 = _bin_stats(cent.astype(np.float32))

    # ── Node 4: ZCR (voice quality proxy) ────────────────────────────────
    signs = np.sign(frms)
    zcr   = (np.diff(signs, axis=1) != 0).mean(axis=1).astype(np.float32)
    node4 = _bin_stats(zcr)

    feat = np.stack([node0, node1, node2, node3, node4], axis=0)  # (5, 20)
    return torch.tensor(feat, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Emotion Prototype Bank                                        [F7]
# ═══════════════════════════════════════════════════════════════════════════════
class EmotionPrototypes:
    """
    Fits per-emotion mean and std feature tensors across all RAVDESS samples.

    Two purposes:
      (a) Training — normalise targets into prototype-relative space so the
          model learns relative shifts, not RAVDESS-exaggerated magnitudes.
      (b) Inference — invert the relative delta back to absolute feature
          space so DSP knows how much to shift each acoustic quantity.
    """

    def __init__(self):
        self.protos: Dict[str, torch.Tensor] = {}   # emo → (N, D) mean
        self.scales: Dict[str, torch.Tensor] = {}   # emo → (N, D) std

    def fit(self, all_feats: Dict[str, List[torch.Tensor]]):
        for emo, feat_list in all_feats.items():
            stacked          = torch.stack(feat_list)   # (S, N, D)
            self.protos[emo] = stacked.mean(0)           # (N, D)
            self.scales[emo] = stacked.std(0).clamp(min=1e-6)

    def relative_target(
        self,
        target_feat:  torch.Tensor,   # (N, D)
        target_emo:   str,
        neutral_feat: torch.Tensor,   # (N, D)
    ) -> torch.Tensor:
        """
        How far is target_feat from neutral_feat in prototype-scale units?
        Strips RAVDESS absolute magnitude bias.                           (F7)
        """
        scale          = self.scales[target_emo]
        neutral_offset = (neutral_feat - self.protos["neutral"])  / scale
        target_offset  = (target_feat  - self.protos[target_emo]) / scale
        return target_offset - neutral_offset   # (N, D)

    def absolute_delta(
        self,
        rel_delta:    torch.Tensor,   # (N, D) — GNN output at inference
        neutral_feat: torch.Tensor,   # (N, D) — neutral F5-TTS features
        target_emo:   str,
    ) -> torch.Tensor:
        """
        Inverse of relative_target: convert predicted relative delta
        back to absolute feature space for DSP application.              (F7)
        """
        dev         = rel_delta.device
        scale       = self.scales[target_emo].to(dev)
        proto_neu   = self.protos["neutral"].to(dev)
        proto_tgt   = self.protos[target_emo].to(dev)
        neutral_off = (neutral_feat - proto_neu) / scale
        abs_delta   = (neutral_off + rel_delta) * scale + proto_tgt - neutral_feat
        return abs_delta


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Dataset
# ═══════════════════════════════════════════════════════════════════════════════
class RavdessDataset(Dataset):
    """
    Each item: (neutral_feat, rel_target_delta, emotion_idx)

    neutral_feat     : (N_NODES, NODE_DIM)  — features of a neutral utterance
    rel_target_delta : (N_NODES, NODE_DIM)  — prototype-normalised supervision
                       that the GNN must predict from emotion label alone  (F1)
    emotion_idx      : int — target emotion label

    The target audio's raw features are used ONLY to compute rel_target_delta
    as a supervision signal; they are never fed to the model.
    """

    def __init__(self, root: str, prototypes: EmotionPrototypes):
        super().__init__()
        self.prototypes = prototypes
        self.cache: Dict[str, torch.Tensor] = {}

        # ── Walk RAVDESS, group by actor and emotion ───────────────────────
        # File naming convention: 03-01-{emo}-{intensity}-{stmt}-{rep}-{actor}.wav
        actor_emo: Dict[str, Dict[str, List[str]]] = \
            defaultdict(lambda: defaultdict(list))

        for dp, _, fns in os.walk(root):
            for fname in sorted(fns):
                if not fname.endswith(".wav"):
                    continue
                parts = fname.replace(".wav", "").split("-")
                if len(parts) < 7:
                    continue
                modality, vocal, emo, intensity, stmt = (
                    parts[0], parts[1], parts[2], parts[3], parts[4]
                )
                if not (
                    modality  == "03"          # speech only
                    and vocal     == "01"      # speech channel
                    and intensity in ("01", "02")
                    and emo       in EMOTION_MAP
                    and stmt      in ("01", "02")
                ):
                    continue
                actor = parts[6]
                actor_emo[actor][EMOTION_MAP[emo]].append(
                    os.path.join(dp, fname)
                )

        if not actor_emo:
            raise RuntimeError(
                f"No valid RAVDESS WAVs found under '{root}'.\n"
                "Expected structure: Actor_01/03-01-01-01-01-01-01.wav …"
            )

        # ── Pre-extract all features (cached in RAM) ───────────────────────
        all_paths = set(
            p for ad in actor_emo.values()
              for ps in ad.values()
              for p  in ps
        )
        print(f"[Dataset] Extracting features for {len(all_paths)} files …")
        for p in all_paths:
            try:
                self.cache[p] = extract_features(p)
            except Exception as exc:
                warnings.warn(f"Skipping {p}: {exc}")

        # ── Fit prototypes across whole corpus ────────────────────────────
        all_feats: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for ad in actor_emo.values():
            for emo, paths in ad.items():
                for p in paths:
                    if p in self.cache:
                        all_feats[emo].append(self.cache[p])

        if not all_feats:
            raise RuntimeError("Feature cache is empty — check file paths.")

        self.prototypes.fit(all_feats)
        print(f"[Dataset] Prototypes fitted: {sorted(self.prototypes.protos.keys())}")

        # ── Warn if any emotion has very few samples ───────────────────────
        for emo, feats in all_feats.items():
            if len(feats) < 5:
                warnings.warn(
                    f"Emotion '{emo}' has only {len(feats)} samples — "
                    "steering direction may be unreliable."
                )

        # ── Build (neutral_path, target_path, emotion) triples ───────────
        # Pairs are within the same actor to minimise speaker-identity confounds.
        self.samples: List[Tuple[str, str, str]] = []
        for actor, emo_dict in actor_emo.items():
            if "neutral" not in emo_dict:
                continue
            for emo, tgt_paths in emo_dict.items():
                if emo == "neutral":
                    continue
                for tgt_path in tgt_paths:
                    if tgt_path not in self.cache:
                        continue
                    neu_path = random.choice(emo_dict["neutral"])
                    if neu_path not in self.cache:
                        continue
                    self.samples.append((neu_path, tgt_path, emo))

        if not self.samples:
            raise RuntimeError(
                "No valid (neutral, target) pairs could be built. "
                "Ensure multiple emotions are present under each Actor folder."
            )

        emo_counts = defaultdict(int)
        for _, _, e in self.samples:
            emo_counts[e] += 1
        print(f"[Dataset] {len(self.samples)} pairs built: " +
              ", ".join(f"{e}={emo_counts[e]}" for e in sorted(emo_counts)))

        # ── Keep a reference to all neutral paths (for inference) ─────────
        self._neutral_paths: List[str] = [
            p for ad in actor_emo.values()
              for p in ad.get("neutral", [])
              if p in self.cache
        ]

    def neutral_sample(self) -> str:
        """Return a random neutral WAV path (for F5-TTS conditioning)."""
        if not self._neutral_paths:
            raise RuntimeError("No neutral samples found in dataset.")
        return random.choice(self._neutral_paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        neu_path, tgt_path, emo = self.samples[idx]
        neu_feat = self.cache[neu_path]   # (N, D)
        tgt_feat = self.cache[tgt_path]   # (N, D) — used only for supervision label

        rel_tgt = self.prototypes.relative_target(tgt_feat, emo, neu_feat)

        return (
            neu_feat,
            rel_tgt,
            torch.tensor(EMO2IDX[emo], dtype=torch.long),
        )


def collate_fn(batch):
    neu = torch.stack([b[0] for b in batch])   # (B, N, D)
    tgt = torch.stack([b[1] for b in batch])   # (B, N, D)
    emo = torch.stack([b[2] for b in batch])   # (B,)
    return neu, tgt, emo


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Model                                                    [F1, F2, F3]
# ═══════════════════════════════════════════════════════════════════════════════
class EmotionConditionedGNN(nn.Module):
    """
    Maps (neutral_feat, emotion_label) → relative prosodic delta.

    Key design decisions
    ─────────────────────
    F1  Input is (neutral features, emotion label) — never (target − neutral).
        At inference, no target audio is required.

    F3  Each emotion gets its own adjacency matrix A_e generated by
        A_e = softmax( MLP( emotion_embedding ) ).reshape(N, N)
        so angry, sad, happy can have fundamentally different node couplings.

    F2  An auxiliary emotion classifier sitting on the aggregated node
        representations h_agg forces A_e to produce emotion-discriminative
        node outputs, giving the adjacency a unique optimum under the
        joint loss (style + smoothness + content + classification).
    """

    def __init__(
        self,
        n_emotions:  int = N_EMOTIONS,
        emo_emb_dim: int = EMO_EMB_DIM,
        n_nodes:     int = N_NODES,
        node_dim:    int = NODE_DIM,
        hidden_dim:  int = 32,
    ):
        super().__init__()
        self.n_nodes  = n_nodes
        self.node_dim = node_dim

        # Learned emotion embedding (one vector per emotion class)       (F1)
        self.emo_emb = nn.Embedding(n_emotions, emo_emb_dim)

        # Adjacency MLP — produces one (N×N) matrix per emotion          (F3)
        self.A_mlp = nn.Sequential(
            nn.Linear(emo_emb_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_nodes * n_nodes),
        )

        # Node encoder: concatenates node features with emotion embedding
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim + emo_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Output MLP: aggregated node hidden → per-node delta
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Auxiliary emotion classifier on flattened aggregated features  (F2)
        # Forces adjacency to encode emotion-discriminative structure.
        self.aux_cls = nn.Linear(hidden_dim * n_nodes, n_emotions)

    def adjacency(self, emo_emb: torch.Tensor) -> torch.Tensor:
        """
        emo_emb : (B, E)
        Returns row-softmax normalised A : (B, N, N)
        """
        return F.softmax(
            self.A_mlp(emo_emb).view(-1, self.n_nodes, self.n_nodes),
            dim=-1,
        )

    def forward(
        self,
        neutral_feat: torch.Tensor,   # (B, N, D)
        emo_idx:      torch.Tensor,   # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
          delta      (B, N, D)          — predicted relative prosodic delta
          aux_logits (B, n_emotions)    — for auxiliary classification loss
          A          (B, N, N)          — emotion-specific adjacency (for inspection)
        """
        B, N, _ = neutral_feat.shape

        e     = self.emo_emb(emo_idx)                           # (B, E)
        A     = self.adjacency(e)                               # (B, N, N)

        e_exp = e.unsqueeze(1).expand(-1, N, -1)                # (B, N, E)
        h     = self.node_enc(
            torch.cat([neutral_feat, e_exp], dim=-1)            # (B, N, D+E)
        )                                                        # (B, N, H)

        h_agg      = torch.bmm(A, h)                            # (B, N, H)
        delta      = self.out_mlp(h_agg)                        # (B, N, D)
        aux_logits = self.aux_cls(h_agg.view(B, -1))            # (B, n_emo)

        return delta, aux_logits, A


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Losses
# ═══════════════════════════════════════════════════════════════════════════════
def style_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    """MSE in prototype-normalised delta space — strips RAVDESS bias.   (F7)"""
    return F.mse_loss(pred_delta, target_delta)


def smoothness_loss(delta: torch.Tensor) -> torch.Tensor:
    """
    Penalise abrupt bin-to-bin jumps within each node's temporal trajectory.
    delta: (B, N, T_BINS*2)
    """
    B, N, _ = delta.shape
    d    = delta.view(B, N, T_BINS, 2)
    diff = d[:, :, 1:, :] - d[:, :, :-1, :]   # (B, N, T-1, 2)
    return diff.pow(2).mean()


def content_preservation_loss(delta: torch.Tensor) -> torch.Tensor:
    """
    Penalise large deltas on rate and voice_quality nodes.
    These are most correlated with phonetic intelligibility.             (F6)
    """
    return delta[:, CONTENT_NODES, :].pow(2).mean()


def aux_classification_loss(logits: torch.Tensor, emo_idx: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy on the auxiliary emotion classifier.
    Gives the adjacency matrix A_e a unique optimum.                    (F2)
    """
    return F.cross_entropy(logits, emo_idx)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Training
# ═══════════════════════════════════════════════════════════════════════════════
def run_epoch(
    model:   EmotionConditionedGNN,
    dl:      DataLoader,
    device:  torch.device,
    opt:     torch.optim.Optimizer,   # None → validation mode
    weights: Tuple[float, float, float, float],
) -> Dict[str, float]:
    training = opt is not None
    model.train(training)
    w_style, w_smooth, w_content, w_aux = weights
    metrics: Dict[str, List[float]] = defaultdict(list)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for neu, tgt_delta, emo_idx in dl:
            neu       = neu.to(device)
            tgt_delta = tgt_delta.to(device)
            emo_idx   = emo_idx.to(device)

            pred_delta, aux_logits, _ = model(neu, emo_idx)

            L_style   = style_loss(pred_delta, tgt_delta)
            L_smooth  = smoothness_loss(pred_delta)
            L_content = content_preservation_loss(pred_delta)
            L_aux     = aux_classification_loss(aux_logits, emo_idx)
            loss      = (
                w_style   * L_style
              + w_smooth  * L_smooth
              + w_content * L_content
              + w_aux     * L_aux
            )

            if training:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            metrics["loss"].append(loss.item())
            metrics["style"].append(L_style.item())
            metrics["aux_acc"].append(
                (aux_logits.argmax(-1) == emo_idx).float().mean().item()
            )

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def train_gnn(
    dataset:    RavdessDataset,
    prototypes: EmotionPrototypes,
    device:     torch.device,
) -> EmotionConditionedGNN:
    """
    Full training loop.
    Returns the best model (lowest val loss) loaded back from checkpoint.
    """
    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = EmotionConditionedGNN().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    weights   = (1.0, W_SMOOTH, W_CONTENT, W_AUX)
    ckpt_path = GNN_CKPT_DIR / "best_emotion_gnnsteer3.pt"
    GNN_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val  = float("inf")

    print(f"\n[Train] {n_train} train / {n_val} val samples  |  device={device}")

    for epoch in range(1, EPOCHS + 1):
        tr  = run_epoch(model, train_dl, device, opt,  weights)
        val = run_epoch(model, val_dl,   device, None, weights)
        sched.step()

        print(
            f"  Epoch {epoch:3d}/{EPOCHS}  "
            f"train={tr['loss']:.4f}  val={val['loss']:.4f}  "
            f"aux_acc={val['aux_acc']:.1%}  "
            f"lr={sched.get_last_lr()[0]:.2e}"
        )

        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save({
                "epoch":             epoch,
                "model_state":       model.state_dict(),
                "val_loss":          val["loss"],
                "prototypes_protos": {k: v.cpu() for k, v in prototypes.protos.items()},
                "prototypes_scales": {k: v.cpu() for k, v in prototypes.scales.items()},
            }, ckpt_path)
            print(f"  ✓ checkpoint saved  (val={val['loss']:.4f})")

    print(f"\n[Train] Complete.  Best val loss: {best_val:.4f}")

    # Reload best checkpoint
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_model = EmotionConditionedGNN().to(device)
    best_model.load_state_dict(ckpt["model_state"])
    best_model.eval()

    print_adjacency(best_model, device)
    return best_model


def print_adjacency(model: EmotionConditionedGNN, device: torch.device):
    """Print the learned adjacency matrix for each emotion."""
    model.eval()
    print("\n[GNN] Learned adjacency matrices A_e:")
    with torch.no_grad():
        for emo, idx in EMO2IDX.items():
            e = model.emo_emb(torch.tensor([idx], device=device))
            A = model.adjacency(e).squeeze(0).cpu().numpy()
            print(f"\n  [{emo.upper()}]")
            print("  " + " " * 16 + "".join(f"{n:>16s}" for n in NODE_NAMES))
            for i, row_name in enumerate(NODE_NAMES):
                row = "".join(f"{A[i, j]:>16.3f}" for j in range(N_NODES))
                print(f"  {row_name:16s}{row}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  DSP — apply GNN delta to a waveform                           [F5]
# ═══════════════════════════════════════════════════════════════════════════════
def apply_delta_to_audio(
    wav:       torch.Tensor,    # (1, T)
    abs_delta: torch.Tensor,    # (N_NODES, NODE_DIM) — absolute feature space
) -> torch.Tensor:
    """
    Translates the absolute feature delta into waveform-level DSP.

    Semantic alignment with extract_features                             (F5)
    ────────────────────────────────────────────────────────────────────
    Every DSP operation moves the same acoustic quantity its node tracks,
    in the same direction:
      Node 0 pitch     → soft FIR high-shelf (F0 proxy; true PSOLA needs lib)
      Node 1 energy    → RMS gain scaling
      Node 2 rate      → NOT applied (content-safe; training suppresses it)
      Node 3 spectral  → first-order FIR spectral tilt
      Node 4 zcr       → gentle low-pass smoothing (lowers ZCR)

    Summarise each node's trajectory to a single scalar by taking the
    mean of all per-bin means (every even index in the (N, 20) tensor).
    """
    wav_np          = wav.squeeze(0).numpy().copy()
    delta_np        = abs_delta.detach().cpu().numpy()    # (N, T*2)
    node_mean_delta = delta_np[:, 0::2].mean(axis=1)     # (N,)

    # ── Node 1: Energy (RMS gain) ─────────────────────────────────────────
    rms_cur = float(np.sqrt((wav_np ** 2).mean()) + 1e-8)
    rms_tgt = max(rms_cur + node_mean_delta[1], 1e-6)
    gain    = float(np.clip(rms_tgt / rms_cur, 0.5, 2.0))
    wav_np  = wav_np * gain

    # ── Node 3: Spectral tilt (FIR) ──────────────────────────────────────
    spec_shift = float(node_mean_delta[3])
    if abs(spec_shift) > 20.0:
        alpha  = float(np.clip(spec_shift / 3000.0, -0.25, 0.25))
        fir    = np.array([1.0 + alpha, -alpha * 0.7], dtype=np.float32)
        wav_np = np.convolve(wav_np, fir, mode="same")

    # ── Node 0: Pitch proxy (soft high-shelf FIR) ─────────────────────────
    pitch_shift = float(node_mean_delta[0])
    if abs(pitch_shift) > 5.0:
        p_alpha = float(np.clip(pitch_shift / 400.0, -0.15, 0.15))
        p_fir   = np.array([1.0 + p_alpha, -p_alpha * 0.5], dtype=np.float32)
        wav_np  = np.convolve(wav_np, p_fir, mode="same")

    # ── Node 4: Voice quality (low-pass damp when ZCR should decrease) ────
    if float(node_mean_delta[4]) < -0.01:
        wav_np = np.convolve(wav_np,
                             np.array([0.5, 0.5], dtype=np.float32),
                             mode="same")

    # ── Node 2: Rate — intentionally skipped                          (F6)

    # ── Final normalisation ───────────────────────────────────────────────
    peak = np.abs(wav_np).max()
    if peak > 1.0:
        wav_np /= peak

    return torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  F5-TTS generation + steering
# ═══════════════════════════════════════════════════════════════════════════════
def load_f5tts():
    """Load F5-TTS DiT model and Vocos vocoder."""
    sys.path.insert(0, str(F5_ROOT / "src"))
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_model, load_vocoder

    model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2,
        text_dim=512, text_mask_padding=True, conv_layers=4,
    )
    print("[F5-TTS] Loading model …")
    f5_model = load_model(
        DiT, model_cfg, str(CKPT_PATH),
        mel_spec_type="vocos",
        vocab_file=str(VOCAB_FILE),
        device=DEVICE,
    )
    f5_model.eval()

    print("[F5-TTS] Loading vocoder …")
    vocoder = load_vocoder("vocos", is_local=False, device=DEVICE)
    return f5_model, vocoder


def generate_neutral(
    f5_model,
    vocoder,
    ref_audio_path: str,
    ref_text:       str,
) -> Tuple[np.ndarray, int]:
    """
    Generate speech for TARGET_TEXT using F5-TTS, cloning the voice from
    ref_audio_path.

    Follows EmoSteer exactly: call infer_process with ref_audio, ref_text,
    and gen_text=TARGET_TEXT, then save the full output with .squeeze().
    No trimming, no preprocessing wrapper — infer_process handles everything
    internally and its output is the correct file to save and use downstream.

    The returned array includes the reference audio prepended (that is normal
    F5-TTS behaviour for zero-shot voice cloning) and is what you hear when
    you play neutral.wav — reference voice clone followed by TARGET_TEXT.
    """
    from f5_tts.infer.utils_infer import infer_process

    with torch.inference_mode():
        audio_out, sr, _ = infer_process(
            ref_audio  = ref_audio_path,   # raw path, exactly like EmoSteer
            ref_text   = ref_text,          # RAVDESS sentence for that clip
            gen_text   = TARGET_TEXT,       # what we actually want generated
            model_obj  = f5_model,
            vocoder    = vocoder,
            device     = DEVICE,
            nfe_step   = NFE_STEP,
        )

    # .squeeze() with no arg is safe here: infer_process returns (T,) or (1,T)
    # and TARGET_TEXT is long enough that T >> 1, so no silent dim-collapse risk.
    wav_np = audio_out.squeeze()
    print(f"  [F5-TTS] Output: {len(wav_np)/sr:.2f}s at {sr} Hz  ({len(wav_np)} samples)")
    return wav_np, sr
def generate_steered_outputs(
    gnn_model:    EmotionConditionedGNN,
    prototypes:   EmotionPrototypes,
    neutral_wav:  np.ndarray,
    neutral_sr:   int,
    neutral_path: str,       # path to the RAVDESS neutral file used as F5-TTS ref
) -> None:
    """
    For each steer emotion (happy, sad, angry):
      1. Extract features from the neutral F5-TTS output
      2. Run the GNN with that emotion label → relative delta
      3. Convert to absolute delta via prototypes
      4. Apply DSP to neutral_wav → steered_wav
      5. Save to ./Output/GNN/{emotion}.wav
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Features of the neutral F5-TTS output (not the RAVDESS ref)
    # We save it to a temp file so extract_features can load it
    tmp_path = str(OUTPUT_DIR / "_neutral_tmp.wav")
    sf.write(tmp_path, neutral_wav, neutral_sr)
    neu_feat = extract_features(tmp_path)   # (N, D)

    # Keep everything on CPU for inference — prototypes live on CPU
    gnn_model.eval().cpu()

    gnn_model.eval()
    for emo in STEER_EMOTIONS:
        # Both neu_feat and emo_idx on CPU; gnn_model already moved above
        emo_idx = torch.tensor([EMO2IDX[emo]])   # CPU

        with torch.no_grad():
            pred_delta, _, _ = gnn_model(
                neu_feat.unsqueeze(0),   # (1, N, D) — CPU
                emo_idx,                 # CPU
            )
        # Scale intensity: 0 = no change, 1 = full RAVDESS-scale steering
        pred_delta = pred_delta.squeeze(0) * STEERING_INTENSITY   # (N, D) CPU

        # Invert prototype normalisation → absolute feature delta for DSP
        # Both pred_delta and neu_feat are on CPU; absolute_delta handles device
        abs_delta = prototypes.absolute_delta(pred_delta, neu_feat, emo)

        wav_tensor = torch.tensor(neutral_wav, dtype=torch.float32).unsqueeze(0)
        steered    = apply_delta_to_audio(wav_tensor, abs_delta)

        out_path = OUTPUT_DIR / f"{emo}.wav"
        sf.write(str(out_path), steered.squeeze(0).numpy(), neutral_sr)
        print(f"  [Output] Saved steered ({emo}) → {out_path}")

    # Clean up temp file
    try:
        os.remove(tmp_path)
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device(DEVICE)
    print(f"[Main] Device: {device}")
    print(f"[Main] Target text: \"{TARGET_TEXT}\"")
    print(f"[Main] Steering intensity: {STEERING_INTENSITY}")

    # ── Step 1: Download RAVDESS ───────────────────────────────────────────
    print("\n── Step 1: RAVDESS dataset ──────────────────────────────────────")
    ravdess_root = download_ravdess()

    # ── Step 2: Build dataset + extract features ───────────────────────────
    print("\n── Step 2: Feature extraction ───────────────────────────────────")
    prototypes = EmotionPrototypes()
    dataset    = RavdessDataset(ravdess_root, prototypes)

    # ── Step 3: Train GNN ─────────────────────────────────────────────────
    print("\n── Step 3: Training EmotionConditionedGNN ───────────────────────")
    gnn_model = train_gnn(dataset, prototypes, device)

    # ── Step 4: Load F5-TTS ───────────────────────────────────────────────
    print("\n── Step 4: Loading F5-TTS ───────────────────────────────────────")
    f5_model, vocoder = load_f5tts()

    # ── Step 5: Generate neutral audio ────────────────────────────────────
    print("\n── Step 5: Generating neutral audio ─────────────────────────────")
    # Use a RAVDESS neutral sample as the voice reference — avoids prosody
    # bleed from expressive speakers (mirrors EmoSteer FIX 6)
    ref_audio_path = dataset.neutral_sample()
    # Use the known text for that RAVDESS utterance
    # (neutral samples use statement "01" or "02"; we pick "01" text)
    ref_text = "Kids are talking by the door"

    print(f"  Reference audio: {ref_audio_path}")
    neutral_wav, neutral_sr = generate_neutral(
        f5_model, vocoder, ref_audio_path, ref_text
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    neutral_out = OUTPUT_DIR / "neutral.wav"
    sf.write(str(neutral_out), neutral_wav, neutral_sr)
    print(f"  [Output] Saved neutral → {neutral_out}")

    # ── Step 6: Steer and generate all emotions ───────────────────────────
    print("\n── Step 6: Generating steered outputs ───────────────────────────")
    generate_steered_outputs(
        gnn_model, prototypes, neutral_wav, neutral_sr, ref_audio_path
    )

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n══ All outputs saved ════════════════════════════════════════════")
    print(f"  {OUTPUT_DIR / 'neutral.wav'}")
    for emo in STEER_EMOTIONS:
        print(f"  {OUTPUT_DIR / (emo + '.wav')}")


if __name__ == "__main__":
    main()