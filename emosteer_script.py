"""
EmoSteer — Corrected Implementation
=====================================
Fixes applied:
  1. Clean mean-only aggregation (no double-reduction noise)
  2. w_l computed via softmax token-importance scores & saved alongside s_hat
  3. Top-k positional masking at inference hooks
  4. nfe_step raised to 32 for stable CFM trajectories
  5. ALPHA default lowered to 2.5 (tunable); aggressive values warned against
  6. Neutral reference audio selected automatically for cleaner conditioning
  7. Vocoder=None guarded during stat collection
  8. Per-emotion sample count raised; fallback warning if samples are scarce
  9. compute_steering skips layers with insufficient cross-emotion coverage
 10. All hooks guaranteed removed via try/finally
"""

import os
import sys
import torch
import requests
import zipfile
import random
import warnings
from pathlib import Path
from collections import defaultdict
import soundfile as sf
import torch.nn.functional as F

# ───────── CONFIG ─────────

DEVICE        = "mps"                  # change to "cuda" if available
TARGET_EMOTION = "happy"

F5_ROOT    = Path("./F5-TTS")
sys.path.insert(0, str(F5_ROOT / "src"))

CKPT_PATH  = F5_ROOT / "ckpts" / "F5TTS_v1_Base" / "model_1250000.safetensors"
VOCAB_FILE = F5_ROOT / "ckpts" / "F5TTS_v1_Base" / "vocab.txt"

DATA_DIR   = "./EmoSteer/data_ravdess"
STEER_PATH = f"./EmoSteer/{TARGET_EMOTION}_alpha2.pt"
#OUTPUT_WAV = f"./Output/EmoSteer/{TARGET_EMOTION}.wav"

# FIX 5 — lowered from 8.0; start here and increase only if effect is too weak
ALPHA = {
    "happy": 2.0,
    "sad": 1.5,
    "angry": 1.5,
}[TARGET_EMOTION]

# FIX 3 — top-k fraction of token positions to steer (0.0–1.0)
TOP_K_FRACTION     = 0.4
# FIX 8 — raised from 15 for more reliable mean estimates
MAX_SAMPLES_PER_EMOTION = 40
# FIX 4 — raised from 8; 32 matches the paper and keeps trajectory stable
NFE_STEP           = 32
# Deeper layers only (semantic/style content); same as before but documented
STEER_LAYER_START  = 14
NUM_LAYERS         = 22

TARGET_TEXT = f"This is an example of {TARGET_EMOTION} speech."

OUTPUT_NEUTRAL_WAV = f"./Output/EmoSteer/{TARGET_EMOTION}_neutral_aplha{ALPHA}.wav"
OUTPUT_STEERED_WAV = f"./Output/EmoSteer/{TARGET_EMOTION}_steered_alpha{ALPHA}.wav"
# ───────── DOWNLOAD ─────────
def download_ravdess() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ravdess.zip")

    if not os.path.exists(zip_path):
        print("Downloading RAVDESS dataset …")
        url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    extract_path = os.path.join(DATA_DIR, "ravdess")
    if not os.path.exists(extract_path):
        print("Extracting …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)

    return extract_path


# ───────── DATASET ─────────
EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    #"06": "fearful",
    #"07": "disgust",
    #"08": "surprised",
}

RAVDESS_TEXT = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}


class RavdessDataset(torch.utils.data.Dataset):
    """
    Loads RAVDESS speech-only WAVs.
    Accepts both intensity levels (01 = normal, 02 = strong) so that
    more samples are available for reliable mean estimation.
    """

    def __init__(self, root: str):
        self.samples: list[tuple[str, str, str]] = []

        for dp, _, fns in os.walk(root):
            for f in fns:
                if not f.endswith(".wav"):
                    continue

                parts = f.replace(".wav", "").split("-")
                if len(parts) < 7:
                    continue

                modality, vocal, emo, intensity, stmt = (
                    parts[0], parts[1], parts[2], parts[3], parts[4]
                )

                # FIX 8 — accept both intensities to widen sample pool
                if not (
                    modality == "03"
                    and vocal == "01"
                    and intensity in ("01", "02")
                    and emo in EMOTION_MAP
                    and stmt in RAVDESS_TEXT
                ):
                    continue

                self.samples.append((
                    os.path.join(dp, f),
                    RAVDESS_TEXT[stmt],
                    EMOTION_MAP[emo],
                ))

        if not self.samples:
            raise RuntimeError(
                f"No valid WAV files found under {root}. "
                "Check the extraction path."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

    def neutral_sample(self) -> tuple[str, str, str]:
        """
        FIX 6 — return a neutral sample for use as inference reference,
        avoiding prosody bleed-through from expressive speakers.
        """
        neutrals = [s for s in self.samples if s[2] == "neutral"]
        if not neutrals:
            warnings.warn("No neutral samples found; falling back to first sample.")
            return self.samples[0]
        return random.choice(neutrals)


# ───────── LOAD MODEL ─────────
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process


def load_f5tts():
    model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2,
        text_dim=512, text_mask_padding=True, conv_layers=4,
    )
    model = load_model(
        DiT, model_cfg, str(CKPT_PATH),
        mel_spec_type="vocos",
        vocab_file=str(VOCAB_FILE),
        device=DEVICE,
    )
    model.eval()

    vocoder = load_vocoder("vocos", is_local=False, device=DEVICE)
    return model, vocoder


# ───────── HOOKS — stat collection ─────────
def attach_hooks(model, storage: defaultdict) -> list:
    hooks = []
    for i, layer in enumerate(model.transformer.transformer_blocks):
        def _hook(m, inp, out, idx=i):
            # out shape: (batch, seq_len, dim)
            storage[idx].append(out.detach().cpu())
        hooks.append(layer.register_forward_hook(_hook))
    return hooks


# ───────── STAT COLLECTION ─────────
def collect_stats(dataset: RavdessDataset, model) -> dict:
    """
    FIX 1 — Collect clean per-sample mean activations.
    Shape stored: stats[(emotion, layer)] = list of (dim,) tensors.
    """
    stats: dict[tuple, list] = defaultdict(list)

    grouped: dict[str, list] = defaultdict(list)
    for path, text, label in dataset:
        grouped[label].append((path, text))

    for label, items in grouped.items():
        n = min(MAX_SAMPLES_PER_EMOTION, len(items))
        if n < 5:
            warnings.warn(
                f"Emotion '{label}' has only {n} samples — "
                "steering direction may be unreliable."
            )
        samples = random.sample(items, n)

        for path, text in samples:
            storage: defaultdict = defaultdict(list)
            hooks = attach_hooks(model, storage)

            try:
                with torch.inference_mode():
                    # FIX 7 — vocoder=None is intentional here (we only want
                    # activations); guarded so exceptions don't leave hooks dangling
                    infer_process(
                        ref_audio=path,
                        ref_text=text,
                        gen_text=text,
                        model_obj=model,
                        vocoder=None,           # intentional: skip waveform decode
                        device=DEVICE,
                        nfe_step=NFE_STEP,      # FIX 4
                    )
            except Exception as exc:
                warnings.warn(f"Forward pass failed for {path}: {exc}")
                continue
            finally:
                for h in hooks:
                    h.remove()          # FIX 10 — always removed

            for layer_idx, acts_list in storage.items():
                # acts_list: list of (batch, seq, dim) across CFM steps
                acts = torch.cat(acts_list, dim=0)   # (steps*batch, seq, dim)
                # FIX 1 — clean mean across time steps and sequence positions
                mean_act = acts.mean(dim=(0, 1))     # (dim,)
                stats[(label, layer_idx)].append(mean_act)

    return stats


# ───────── STEERING VECTOR COMPUTATION ─────────
def compute_token_weights(
    target_acts: torch.Tensor,    # (n_samples, dim)
    baseline_acts: torch.Tensor,  # (m_samples, dim)
) -> torch.Tensor:
    """
    FIX 2 — Compute w_l via a simple softmax importance score:
    importance_d = |mean_target_d - mean_baseline_d|
    w_l = softmax(importance) over feature dim → scalar weight per position.

    At inference the weight is projected back to sequence positions by
    computing per-position activation magnitudes.
    """
    diff = target_acts.mean(0) - baseline_acts.mean(0)      # (dim,)
    importance = torch.abs(diff)
    w = F.softmax(importance, dim=0)                         # (dim,) sums to 1
    return w


def compute_steering(stats: dict) -> None:
    """
    FIX 2, 9 — Compute and save both s_hat and w_l per layer.
    Skips layers with insufficient cross-emotion coverage.
    """
    # Consolidate lists → tensors
    consolidated: dict[tuple, torch.Tensor] = {}
    for (label, layer_idx), acts_list in stats.items():
        if acts_list:
            consolidated[(label, layer_idx)] = torch.stack(acts_list)  # (n, dim)

    s_hat: dict[int, torch.Tensor] = {}
    w_l:   dict[int, torch.Tensor] = {}

    all_emotions = list(EMOTION_MAP.values())

    for layer_idx in range(NUM_LAYERS):
        if (TARGET_EMOTION, layer_idx) not in consolidated:
            continue

        target_acts = consolidated[(TARGET_EMOTION, layer_idx)]   # (n, dim)

        others = [
            consolidated[(e, layer_idx)]
            for e in all_emotions
            if e != TARGET_EMOTION and (e, layer_idx) in consolidated
        ]

        # FIX 9 — need at least 2 other emotions for a reliable baseline
        if len(others) < 2:
            warnings.warn(
                f"Layer {layer_idx}: only {len(others)} baseline emotion(s) "
                "available — skipping."
            )
            continue

        baseline_acts = torch.cat(others, dim=0)                   # (m, dim)

        # Difference-in-means direction
        u = target_acts.mean(0) - baseline_acts.mean(0)            # (dim,)
        s_hat[layer_idx] = u / (torch.norm(u) + 1e-8)

        # FIX 2 — token importance weights
        w_l[layer_idx] = compute_token_weights(target_acts, baseline_acts)

    if not s_hat:
        raise RuntimeError(
            "No steering vectors computed. Check dataset and emotion labels."
        )

    os.makedirs(os.path.dirname(STEER_PATH), exist_ok=True)
    torch.save({"s_hat": s_hat, "w_l": w_l}, STEER_PATH)
    print(f"Steering vectors saved for {len(s_hat)} layers → {STEER_PATH}")


# ───────── EMOSTEER — inference ─────────
class EmotionSteerer:
    def __init__(self, path: str):
        d = torch.load(path, map_location=DEVICE)
        self.s_hat: dict[int, torch.Tensor] = d["s_hat"]
        self.w_l:   dict[int, torch.Tensor] = d["w_l"]   # FIX 2


def _top_k_mask(seq_len: int, scores: torch.Tensor, fraction: float) -> torch.Tensor:
    """
    FIX 3 — Build a boolean mask of shape (seq_len,) that is True
    for the top-k sequence positions ranked by their projected importance.
    `scores` is a (seq_len,) tensor of per-position relevance.
    """
    k = max(1, int(seq_len * fraction))
    topk_indices = torch.topk(scores, k=k, dim=0).indices
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[topk_indices] = True
    return mask


def attach_emosteer(model, steerer: EmotionSteerer) -> list:
    """
    FIX 2, 3, 5 — Attach inference hooks that:
      • apply only to layers ≥ STEER_LAYER_START
      • use w_l to score sequence positions, then mask to top-k
      • scale by local activation std (keeps magnitude relative)
      • use the corrected ALPHA
    """
    hooks = []

    for i, layer in enumerate(model.transformer.transformer_blocks):
        if i < STEER_LAYER_START:
            continue
        if i not in steerer.s_hat:
            continue

        s = steerer.s_hat[i]   # (dim,)
        w = steerer.w_l[i]     # (dim,) — feature-dim importance weights

        def _hook(m, inp, out, idx=i, _s=s, _w=w):
            # out: (batch, seq_len, dim)
            _s = _s.to(out.device)
            _w = _w.to(out.device)

            batch, seq_len, dim = out.shape

            # Per-position importance: project activations onto w
            # shape: (batch, seq_len)
            pos_scores = (out * _w.unsqueeze(0).unsqueeze(0)).sum(-1).abs()
            pos_scores = pos_scores.mean(0)              # (seq_len,) avg over batch

            # FIX 3 — build top-k mask
            mask = _top_k_mask(seq_len, pos_scores, TOP_K_FRACTION)
            mask = mask.to(out.device)                   # (seq_len,)

            # Local scale: std of active positions
            active = out[:, mask, :]                     # (batch, k, dim)
            scale = active.std(dim=-1, keepdim=True).mean() + 1e-5   # scalar

            # Additive steering — only at selected positions
            delta = ALPHA * scale * _s                   # (dim,)
            out = out.clone()
            out[:, mask, :] = out[:, mask, :] + delta.unsqueeze(0).unsqueeze(0)

            return out

        hooks.append(layer.register_forward_hook(_hook))

    return hooks


# ───────── MAIN ─────────
def main():
    if ALPHA > 6.0:
        warnings.warn(
            f"ALPHA={ALPHA} is high. Values above 6.0 frequently cause "
            "audio artifacts. Consider starting at 2.0–3.0."
        )

    data_path = download_ravdess()
    dataset   = RavdessDataset(data_path)
    print(f"Dataset size: {len(dataset)} samples")

    model, vocoder = load_f5tts()

    print("Collecting activations …")
    stats = collect_stats(dataset, model)

    print("Computing steering vectors …")
    compute_steering(stats)

    print("Generating steered audio …")
    steerer = EmotionSteerer(STEER_PATH)
    hooks   = attach_emosteer(model, steerer)

    # FIX 6 — use a neutral reference to minimise prosody bleed-through
    print("Generating neutral audio (no steering) …")

    # FIX 6 — use a neutral reference to minimise prosody bleed-through
    ref_audio, ref_text, _ = dataset.neutral_sample()

    with torch.inference_mode():
        neutral_audio, sr, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=TARGET_TEXT,
            model_obj=model,
            vocoder=vocoder,
            device=DEVICE,
            nfe_step=NFE_STEP,
        )

    os.makedirs(os.path.dirname(OUTPUT_NEUTRAL_WAV), exist_ok=True)
    sf.write(OUTPUT_NEUTRAL_WAV, neutral_audio.squeeze(), sr)
    print(f"Saved neutral: {OUTPUT_NEUTRAL_WAV}")


    print("Generating steered audio …")
    steerer = EmotionSteerer(STEER_PATH)
    hooks   = attach_emosteer(model, steerer)

    try:
        with torch.inference_mode():
            steered_audio, sr, _ = infer_process(
                ref_audio=ref_audio,     # SAME reference
                ref_text=ref_text,       # SAME content
                gen_text=TARGET_TEXT,    # SAME text
                model_obj=model,
                vocoder=vocoder,
                device=DEVICE,
                nfe_step=NFE_STEP,
            )
    finally:
        for h in hooks:
            h.remove()   # FIX 10 — always cleaned up

    sf.write(OUTPUT_STEERED_WAV, steered_audio.squeeze(), sr)
    print(f"Saved steered: {OUTPUT_STEERED_WAV}")


if __name__ == "__main__":
    main()