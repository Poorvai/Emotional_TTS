"""
=============================================================
  Emotional TTS Evaluation Metrics
=============================================================
  Covers:
    1. Word Error Rate (WER)
    2. Mel-Cepstral Distortion (MCD)
    3. Emotion Similarity Score (E-SIM)
    4. Speaker Similarity (S-SIM)
    5. Prosody / Duration Metrics
    6. Mean Opinion Score logger (MOS / N-MOS / EI-MOS / EE-MOS)
    7. Emotional Appropriateness logger
    8. Frame-wise Emotion Similarity (research-grade)

  Install dependencies:
    pip install jiwer librosa numpy scipy speechbrain
    pip install transformers torch torchaudio
    pip install soundfile pandas tabulate
=============================================================
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from tabulate import tabulate


# ─────────────────────────────────────────────
# 1. WORD ERROR RATE (WER)
# ─────────────────────────────────────────────
def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between reference text and ASR hypothesis.

    Args:
        reference  : Ground-truth transcript (str)
        hypothesis : ASR-decoded transcript of synthesised audio (str)

    Returns:
        WER as a float (0.0 = perfect, 1.0 = total mismatch)

    Usage:
        score = compute_wer("hello world", "hello word")
        # 0.5 → one word wrong out of two
    """
    try:
        from jiwer import wer
    except ImportError:
        raise ImportError("Run: pip install jiwer")

    score = wer(reference.lower().strip(), hypothesis.lower().strip())
    print(f"[WER]  Reference : {reference}")
    print(f"[WER]  Hypothesis: {hypothesis}")
    print(f"[WER]  Score     : {score:.4f}  ({score*100:.2f}%)\n")
    return score


# ─────────────────────────────────────────────
# 2. MEL-CEPSTRAL DISTORTION (MCD)
# ─────────────────────────────────────────────
def compute_mcd(ref_audio_path: str, syn_audio_path: str,
                sr: int = 22050, n_mfcc: int = 13) -> float:
    """
    Mel-Cepstral Distortion between reference and synthesised audio.
    Lower is better.

    Args:
        ref_audio_path : Path to reference (natural) .wav file
        syn_audio_path : Path to synthesised .wav file
        sr             : Sample rate to resample to (default 22050)
        n_mfcc         : Number of MFCC coefficients (default 13)

    Returns:
        MCD value (float). Typical range: 4–10 dB

    Usage:
        mcd = compute_mcd("natural.wav", "synthesised.wav")
    """
    ref, _ = librosa.load(ref_audio_path, sr=sr)
    syn, _ = librosa.load(syn_audio_path, sr=sr)

    # Extract MFCCs (skip coefficient 0 — energy term)
    ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc)[1:]
    syn_mfcc = librosa.feature.mfcc(y=syn, sr=sr, n_mfcc=n_mfcc)[1:]

    # Align lengths
    min_len = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_len]
    syn_mfcc = syn_mfcc[:, :min_len]

    # MCD formula: (10 / ln(10)) * sqrt(2 * sum((c_r - c_s)^2))
    diff = ref_mfcc - syn_mfcc
    mcd = (10.0 / np.log(10)) * np.mean(
        np.sqrt(2 * np.sum(diff ** 2, axis=0))
    )
    print(f"[MCD]  Reference : {ref_audio_path}")
    print(f"[MCD]  Synthesis : {syn_audio_path}")
    print(f"[MCD]  Score     : {mcd:.4f} dB  (lower is better)\n")
    return mcd


# ─────────────────────────────────────────────
# 3. EMOTION SIMILARITY SCORE (E-SIM)
# ─────────────────────────────────────────────
def compute_esim(ref_audio_path: str, syn_audio_path: str) -> float:
    """
    Emotion Similarity Score using a pre-trained emotion classifier.
    Compares the emotion embedding of synthesised vs. reference audio.
    Cosine similarity: 1.0 = identical emotion profile.

    Uses: facebook/wav2vec2-base (as a proxy encoder) + softmax logits
    For production, swap with emotion2vec or SER model fine-tuned on
    datasets like IEMOCAP / RAVDESS.

    Args:
        ref_audio_path : Path to reference .wav (natural, labelled emotion)
        syn_audio_path : Path to synthesised .wav

    Returns:
        Cosine similarity score (float, -1 to 1; higher is better)

    Usage:
        score = compute_esim("angry_ref.wav", "angry_syn.wav")
    """
    try:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
    except ImportError:
        raise ImportError("Run: pip install transformers torch torchaudio")

    print("[E-SIM] Loading wav2vec2 encoder …")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()

    def get_embedding(path):
        audio, sr = librosa.load(path, sr=16000)
        inputs = processor(audio, sampling_rate=16000,
                           return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean-pool over time
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    emb_ref = get_embedding(ref_audio_path)
    emb_syn = get_embedding(syn_audio_path)

    # Cosine similarity
    cos_sim = np.dot(emb_ref, emb_syn) / (
        np.linalg.norm(emb_ref) * np.linalg.norm(emb_syn)
    )
    print(f"[E-SIM] Reference : {ref_audio_path}")
    print(f"[E-SIM] Synthesis : {syn_audio_path}")
    print(f"[E-SIM] Score     : {cos_sim:.4f}  (higher is better)\n")
    return float(cos_sim)


# ─────────────────────────────────────────────
# 4. SPEAKER SIMILARITY (S-SIM)
# ─────────────────────────────────────────────
def compute_ssim(ref_audio_path: str, syn_audio_path: str) -> float:
    """
    Speaker Similarity using ECAPA-TDNN speaker embeddings from SpeechBrain.
    Cosine similarity: 1.0 = same speaker, 0.0 = unrelated.

    Args:
        ref_audio_path : Path to reference .wav (natural speaker)
        syn_audio_path : Path to synthesised .wav

    Returns:
        Cosine similarity score (float, 0 to 1; higher is better)

    Usage:
        score = compute_ssim("speaker_ref.wav", "tts_output.wav")
    """
    try:
        import torch
        import torchaudio
        from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        raise ImportError("Run: pip install speechbrain torchaudio torch")

    print("[S-SIM] Loading ECAPA-TDNN speaker encoder …")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )

    def get_spk_embedding(path):
        signal, fs = torchaudio.load(path)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, fs, 16000)
        with torch.no_grad():
            emb = classifier.encode_batch(signal)
        return emb.squeeze().numpy()

    emb_ref = get_spk_embedding(ref_audio_path)
    emb_syn = get_spk_embedding(syn_audio_path)

    cos_sim = np.dot(emb_ref, emb_syn) / (
        np.linalg.norm(emb_ref) * np.linalg.norm(emb_syn)
    )
    print(f"[S-SIM] Reference : {ref_audio_path}")
    print(f"[S-SIM] Synthesis : {syn_audio_path}")
    print(f"[S-SIM] Score     : {cos_sim:.4f}  (higher is better)\n")
    return float(cos_sim)


# ─────────────────────────────────────────────
# 5. PROSODY / DURATION METRICS
# ─────────────────────────────────────────────
def compute_prosody_metrics(ref_audio_path: str, syn_audio_path: str,
                            sr: int = 22050) -> dict:
    """
    Compare prosody features between reference and synthesised audio:
      - Duration ratio
      - Pitch (F0) mean & std deviation error
      - Energy (RMS) mean error

    Args:
        ref_audio_path : Path to reference .wav
        syn_audio_path : Path to synthesised .wav
        sr             : Sample rate

    Returns:
        dict with keys: duration_ratio, pitch_mean_error,
                        pitch_std_error, energy_mean_error

    Usage:
        metrics = compute_prosody_metrics("ref.wav", "syn.wav")
    """
    ref, _ = librosa.load(ref_audio_path, sr=sr)
    syn, _ = librosa.load(syn_audio_path, sr=sr)

    # Duration
    dur_ref = len(ref) / sr
    dur_syn = len(syn) / sr
    duration_ratio = dur_syn / dur_ref  # 1.0 = perfect match

    # Pitch (F0) via piptrack
    def get_pitch_stats(y):
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[magnitudes > np.median(magnitudes)]
        pitch_vals = pitch_vals[pitch_vals > 0]
        if len(pitch_vals) == 0:
            return 0.0, 0.0
        return float(np.mean(pitch_vals)), float(np.std(pitch_vals))

    ref_f0_mean, ref_f0_std = get_pitch_stats(ref)
    syn_f0_mean, syn_f0_std = get_pitch_stats(syn)

    pitch_mean_error = abs(ref_f0_mean - syn_f0_mean)
    pitch_std_error = abs(ref_f0_std - syn_f0_std)

    # Energy (RMS)
    ref_rms = float(np.mean(librosa.feature.rms(y=ref)))
    syn_rms = float(np.mean(librosa.feature.rms(y=syn)))
    energy_mean_error = abs(ref_rms - syn_rms)

    results = {
        "duration_ratio"   : round(duration_ratio, 4),
        "pitch_mean_error" : round(pitch_mean_error, 2),
        "pitch_std_error"  : round(pitch_std_error, 2),
        "energy_mean_error": round(energy_mean_error, 6),
    }
    print("[PROSODY] Results:")
    for k, v in results.items():
        print(f"          {k}: {v}")
    print()
    return results


# ─────────────────────────────────────────────
# 6. MEAN OPINION SCORE LOGGER (MOS / N-MOS / EI-MOS / EE-MOS)
# ─────────────────────────────────────────────
def collect_mos_ratings(samples: list[dict],
                        output_csv: str = "mos_ratings.csv") -> pd.DataFrame:
    """
    Interactive CLI tool to collect MOS-style ratings from a human evaluator.
    Ratings collected:
      - N-MOS  : Naturalness (1–5)
      - EI-MOS : Emotion Intensity (1–5)
      - EE-MOS : Emotion Expressiveness (1–5)

    Args:
        samples    : List of dicts with keys 'id', 'audio_path', 'emotion'
        output_csv : File to save results

    Returns:
        pandas DataFrame with all ratings

    Usage:
        samples = [
            {"id": "s1", "audio_path": "out1.wav", "emotion": "angry"},
            {"id": "s2", "audio_path": "out2.wav", "emotion": "happy"},
        ]
        df = collect_mos_ratings(samples)
    """
    print("\n" + "="*55)
    print("  MOS Rating Session")
    print("  Rate each sample on a scale of 1–5")
    print("  1 = Bad  |  3 = Fair  |  5 = Excellent")
    print("="*55 + "\n")

    records = []
    for s in samples:
        print(f"Sample ID   : {s['id']}")
        print(f"Audio       : {s['audio_path']}")
        print(f"Emotion tag : {s['emotion']}")

        def get_rating(prompt):
            while True:
                try:
                    val = int(input(f"  {prompt} (1-5): "))
                    if 1 <= val <= 5:
                        return val
                    print("  Please enter a number between 1 and 5.")
                except ValueError:
                    print("  Invalid input.")

        n_mos  = get_rating("N-MOS  (Naturalness)")
        ei_mos = get_rating("EI-MOS (Emotion Intensity)")
        ee_mos = get_rating("EE-MOS (Emotion Expressiveness)")

        records.append({
            "id"           : s["id"],
            "audio_path"   : s["audio_path"],
            "emotion"      : s["emotion"],
            "N-MOS"        : n_mos,
            "EI-MOS"       : ei_mos,
            "EE-MOS"       : ee_mos,
            "avg_MOS"      : round((n_mos + ei_mos + ee_mos) / 3, 2),
        })
        print()

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n[MOS] Ratings saved to: {output_csv}")
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    return df


# ─────────────────────────────────────────────
# 7. EMOTIONAL APPROPRIATENESS LOGGER
# ─────────────────────────────────────────────
def collect_emotional_appropriateness(samples: list[dict],
                                      output_json: str = "ea_ratings.json") -> list:
    """
    Collect Emotional Appropriateness ratings from human evaluators.
    For each sample the rater answers:
      Q1. Does the speech convey the intended emotion?    (Yes / No / Partial)
      Q2. How appropriate is the emotion on a 1–5 scale?
      Q3. Free-text comment (optional)

    Args:
        samples     : List of dicts with keys 'id', 'audio_path', 'emotion'
        output_json : File to save results

    Returns:
        List of rating dicts

    Usage:
        results = collect_emotional_appropriateness(samples)
    """
    print("\n" + "="*55)
    print("  Emotional Appropriateness Rating")
    print("="*55 + "\n")

    results = []
    for s in samples:
        print(f"Sample ID   : {s['id']}")
        print(f"Audio       : {s['audio_path']}")
        print(f"Intended    : {s['emotion']}")

        q1 = ""
        while q1 not in ["yes", "no", "partial"]:
            q1 = input("  Q1. Does speech convey intended emotion? "
                       "(yes/no/partial): ").strip().lower()

        while True:
            try:
                q2 = int(input("  Q2. Emotional appropriateness (1-5): "))
                if 1 <= q2 <= 5:
                    break
            except ValueError:
                pass
            print("  Please enter 1–5.")

        q3 = input("  Q3. Comment (optional, press Enter to skip): ").strip()

        results.append({
            "id"                   : s["id"],
            "audio_path"           : s["audio_path"],
            "intended_emotion"     : s["emotion"],
            "emotion_conveyed"     : q1,
            "appropriateness_score": q2,
            "comment"              : q3,
        })
        print()

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EA] Ratings saved to: {output_json}\n")
    return results


# ─────────────────────────────────────────────
# 8. FRAME-WISE EMOTION SIMILARITY (Research-grade)
# ─────────────────────────────────────────────
def compute_framewise_emotion_similarity(ref_audio_path: str,
                                         syn_audio_path: str,
                                         chunk_duration: float = 0.5,
                                         sr: int = 16000) -> dict:
    """
    Frame-wise (chunk-wise) emotion similarity between reference and synthesis.
    Splits audio into fixed-duration chunks, extracts embeddings per chunk,
    and computes average cosine similarity.

    Args:
        ref_audio_path  : Path to reference .wav
        syn_audio_path  : Path to synthesised .wav
        chunk_duration  : Chunk size in seconds (default 0.5)
        sr              : Sample rate

    Returns:
        dict with keys: mean_similarity, std_similarity, per_chunk_scores

    Usage:
        result = compute_framewise_emotion_similarity("ref.wav", "syn.wav")
    """
    try:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
    except ImportError:
        raise ImportError("Run: pip install transformers torch")

    print("[FW-ESIM] Loading encoder for frame-wise analysis …")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.eval()

    ref, _ = librosa.load(ref_audio_path, sr=sr)
    syn, _ = librosa.load(syn_audio_path, sr=sr)

    chunk_size = int(chunk_duration * sr)
    min_samples = min(len(ref), len(syn))
    n_chunks = min_samples // chunk_size

    def embed_chunk(chunk):
        import torch
        inputs = processor(chunk.astype(np.float32),
                           sampling_rate=sr,
                           return_tensors="pt",
                           padding=True)
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state.mean(dim=1).squeeze().numpy()

    similarities = []
    for i in range(n_chunks):
        start = i * chunk_size
        end   = start + chunk_size
        r_chunk = ref[start:end]
        s_chunk = syn[start:end]

        e_ref = embed_chunk(r_chunk)
        e_syn = embed_chunk(s_chunk)

        sim = np.dot(e_ref, e_syn) / (
            np.linalg.norm(e_ref) * np.linalg.norm(e_syn)
        )
        similarities.append(float(sim))

    result = {
        "mean_similarity"    : round(float(np.mean(similarities)), 4),
        "std_similarity"     : round(float(np.std(similarities)), 4),
        "per_chunk_scores"   : [round(s, 4) for s in similarities],
        "n_chunks_evaluated" : n_chunks,
    }
    print(f"[FW-ESIM] Mean Emotion Similarity : {result['mean_similarity']}")
    print(f"[FW-ESIM] Std Dev                 : {result['std_similarity']}")
    print(f"[FW-ESIM] Chunks evaluated        : {n_chunks}\n")
    return result


# ─────────────────────────────────────────────
# MASTER EVALUATION RUNNER
# ─────────────────────────────────────────────
def run_full_evaluation(ref_audio: str,
                        syn_audio: str,
                        ref_text: str,
                        hyp_text: str,
                        emotion: str = "unknown",
                        sample_id: str = "sample_01",
                        run_subjective: bool = False,
                        output_dir: str = "./eval_results") -> dict:
    """
    Run ALL objective metrics in one call and optionally trigger
    subjective rating collection.

    Args:
        ref_audio       : Path to reference (natural) .wav
        syn_audio       : Path to synthesised .wav
        ref_text        : Ground-truth transcript
        hyp_text        : ASR output from synthesised audio
        emotion         : Emotion label (e.g. "happy", "angry")
        sample_id       : Unique ID for this sample
        run_subjective  : If True, prompts for human MOS & EA ratings
        output_dir      : Directory to save JSON/CSV results

    Returns:
        dict with all metric scores
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*55)
    print(f"  FULL EVALUATION  |  Sample: {sample_id}")
    print("="*55 + "\n")

    scores = {"sample_id": sample_id, "emotion": emotion}

    # Objective metrics
    scores["WER"]     = compute_wer(ref_text, hyp_text)
    scores["MCD"]     = compute_mcd(ref_audio, syn_audio)
    scores["E-SIM"]   = compute_esim(ref_audio, syn_audio)
    scores["S-SIM"]   = compute_ssim(ref_audio, syn_audio)
    scores["prosody"] = compute_prosody_metrics(ref_audio, syn_audio)
    scores["fw_esim"] = compute_framewise_emotion_similarity(ref_audio, syn_audio)

    # Subjective (optional)
    if run_subjective:
        sample_list = [{"id": sample_id,
                        "audio_path": syn_audio,
                        "emotion": emotion}]
        mos_df = collect_mos_ratings(
            sample_list,
            output_csv=os.path.join(output_dir, f"{sample_id}_mos.csv")
        )
        scores["MOS"] = mos_df.to_dict(orient="records")[0]
        ea = collect_emotional_appropriateness(
            sample_list,
            output_json=os.path.join(output_dir, f"{sample_id}_ea.json")
        )
        scores["EA"] = ea[0]

    # Save full results
    result_path = os.path.join(output_dir, f"{sample_id}_results.json")
    with open(result_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n[DONE] All results saved to: {result_path}")

    # Pretty summary table
    summary = [
        ["WER",            scores["WER"],                    "lower is better"],
        ["MCD (dB)",       scores["MCD"],                    "lower is better"],
        ["E-SIM",          scores["E-SIM"],                  "higher is better"],
        ["S-SIM",          scores["S-SIM"],                  "higher is better"],
        ["Duration Ratio", scores["prosody"]["duration_ratio"], "ideal = 1.0"],
        ["Pitch Δ Mean",   scores["prosody"]["pitch_mean_error"], "lower is better"],
        ["FW-E-SIM Mean",  scores["fw_esim"]["mean_similarity"],  "higher is better"],
    ]
    print("\n" + tabulate(summary,
                          headers=["Metric", "Score", "Note"],
                          tablefmt="rounded_outline"))
    return scores


# ─────────────────────────────────────────────
# QUICK DEMO / USAGE EXAMPLE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    """
    Example usage — replace paths with your actual files.

    For ASR (needed for WER), you can use:
        pip install openai-whisper
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe("synthesised.wav")
        hyp_text = result["text"]
    """

    # ── Demo: individual metrics ──────────────
    print("Demo: WER")
    compute_wer(
        reference="she was happy to see her friend",
        hypothesis="she was happy to see her friend"
    )

    # ── Demo: full pipeline ───────────────────
    # Uncomment and fill in real paths:
    #
    # run_full_evaluation(
    #     ref_audio      = "data/happy_reference.wav",
    #     syn_audio      = "data/happy_synthesised.wav",
    #     ref_text       = "She was so happy to meet everyone today.",
    #     hyp_text       = "She was so happy to meet everyone today.",  # from ASR
    #     emotion        = "happy",
    #     sample_id      = "happy_001",
    #     run_subjective = False,         # set True for human rating prompts
    #     output_dir     = "./eval_results"
    # )