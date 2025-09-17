"""
audio_processing.py - Audio processing pipeline for Empathetic AI Therapist

This module handles:
- Receiving uploaded audio bytes and converting them to a single-channel 16 kHz WAV.
- Extracting simple audio features (duration, RMS, tempo, pitch mean).
- Running Speech-to-Text to obtain a transcript.
- Running a text-based emotion classifier (via Vertex if available; heuristic fallback otherwise).
- Fusing audio-based and text-based emotion signals into a single final emotion and confidence.
- Detecting safety/crisis language in transcripts.
- Returning a structured JSON result describing transcript, emotion, intensity, safety flags and meta details.

Important notes:
- The implementation intentionally falls back gracefully in multiple places (ffmpeg -> librosa, Vertex -> heuristics).
- No external features or logic were changed — only comments and docstrings were added for clarity.
"""

import os
import re
import logging
import uuid
import subprocess
import numpy as np
import librosa
import scipy.signal
import json
from datetime import datetime

# Suppress a known, harmless warning from librosa (improves log cleanliness)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

# Bring in the GCP helpers and safety utils from your project
from .gcp_clients import transcribe_wav_bytes, vertex_generate
from .utils import detect_safety

# Some scipy versions may not expose hann on scipy.signal; ensure availability
if not hasattr(scipy.signal, "hann"):
    from scipy.signal import windows
    scipy.signal.hann = windows.hann

# Module logger for debugging and error reporting
_logger = logging.getLogger(__name__)

# Temporary working directory for uploaded files and conversions
TMP_DIR = "/tmp/empathetic_ai"
os.makedirs(TMP_DIR, exist_ok=True)


def save_upload_and_convert_to_wav(file_bytes: bytes, filename_hint="upload"):
    """
    Save uploaded bytes to a temp file and convert them to a single-channel 16k WAV.

    Steps:
    1. Write uploaded bytes to a temp file (unique name using UUID).
    2. Try converting the file to 16 kHz mono WAV using ffmpeg (preferred).
    3. If ffmpeg fails, fall back to loading with librosa and writing with soundfile.
    4. Return the path to the converted WAV file.

    Returns:
        path to the converted WAV file (string).
    Raises:
        Re-raises the final exception if both conversion methods fail.
    """
    uid = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR, f"{uid}_{filename_hint}")
    with open(input_path, "wb") as fh:
        fh.write(file_bytes)

    out_wav = os.path.join(TMP_DIR, f"{uid}.wav")
    # ffmpeg command: force sample rate 16000, mono (1 channel), drop video (`-vn`)
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-vn", out_wav]

    try:
        # Run ffmpeg quietly (suppress stdout/stderr to keep logs clean)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_wav
    except Exception as e:
        # If ffmpeg fails (not installed, unsupported format, etc.), try librosa fallback
        _logger.warning("ffmpeg conversion failed: %s — trying librosa fallback", e)
        try:
            import soundfile as sf
            # librosa.load handles many input formats and resamples to target sr
            y, sr = librosa.load(input_path, sr=16000, mono=True)
            # Write out a PCM_16 WAV using soundfile
            sf.write(out_wav, y, 16000, subtype="PCM_16")
            return out_wav
        except Exception as e2:
            # If fallback also fails, log and re-raise so caller knows conversion failed
            _logger.error("Fallback conversion failed: %s", e2)
            raise


def extract_audio_features(wav_path: str, target_sr=16000):
    """
    Load the WAV file and compute basic audio features.

    Extracted features:
    - duration_s: length in seconds
    - rms: mean root-mean-square energy (approximate loudness)
    - tempo: estimated tempo from onset envelope (beats per minute)
    - pitch_mean: mean fundamental frequency estimate using librosa.yin
    - sr: sample rate used

    Returns:
        dict with feature keys (duration_s, rms, tempo, pitch_mean, sr)
    """
    # Load the audio at the desired sample rate (mono)
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # RMS: may fail on some edge cases, so handle exceptions
    try:
        rms = float(np.mean(librosa.feature.rms(y=y)))
    except Exception:
        rms = 0.0

    # Pitch estimation (YIN): fmin/fmax are chosen to cover typical human voice range
    try:
        pitch = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        pitch_mean = float(np.nanmean(pitch))
        if np.isnan(pitch_mean):
            pitch_mean = 0.0
    except Exception:
        pitch_mean = 0.0

    # Tempo estimation using onset envelope and beat tracker; fallback gracefully
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo = float(tempo)
    except Exception:
        tempo = 0.0

    return {
        "duration_s": duration,
        "rms": rms,
        "tempo": tempo,
        "pitch_mean": pitch_mean,
        "sr": sr,
    }


def classify_emotion_text(transcript: str):
    """
    Classify emotion from text.

    Behavior:
    - If transcript is empty or whitespace, returns a neutral default distribution.
    - Otherwise, it attempts to call `vertex_generate` with a strict instruction prompt
      asking for a JSON object containing emotion probabilities and labels.
    - If Vertex is unavailable or returns nothing parsable, a heuristic fallback runs:
      it builds a reasonable probability distribution (gives more mass to 'sad' when sadness keywords present).
    - The function returns a dict with keys:
        final_emotion, context_emotion, confidence, text_probs

    Note:
    - The function expects the Vertex response to contain a JSON object in the text.
    - The parsing includes some naive repairs (single -> double quotes, remove trailing commas).
    """
    # Empty-check: return a deterministic neutral result for empty transcripts
    if not transcript or not transcript.strip():
        return {
            "final_emotion": "neutral",
            "context_emotion": "neutral",
            "confidence": 0.0,
            "text_probs": {
                "neutral": 1.0, "happy": 0.0, "sad": 0.0, "angry": 0.0,
                "fear": 0.0, "disgust": 0.0, "surprise": 0.0
            }
        }

    # Build a strict instruction prompt for Vertex (expects only a JSON object)
    prompt = (
        "You are an expert clinical emotion classifier for therapeutic contexts. "
        "RETURN ONLY a JSON object with these EXACT keys: "
        "\"final_emotion\" (one of [neutral,happy,sad,angry,fear,disgust,surprise]), "
        "\"context_emotion\" (short phrase describing emotional context), "
        "\"confidence\" (0.0-1.0), "
        "\"text_probs\" (object with all 7 emotion labels as keys, values summing to 1.0). "
        "Analyze the user's text with clinical precision:\n\n"
        f"\"{transcript}\"\n\n"
        "Return ONLY the JSON object."
    )
    raw = vertex_generate(prompt, max_output_chars=1024)

    # If no Vertex output, use simple heuristics based on keywords
    if not raw:
        lower = transcript.lower()
        text_probs = {"neutral": 0.6, "happy": 0.05, "sad": 0.15, "angry": 0.05, "fear": 0.1, "disgust": 0.02, "surprise": 0.03}
        sadness_keywords = ["sad", "depress", "down", "upset", "cry", "stressed", "stress", "overwhelm", "overwhelmed", "hopeless"]
        if any(w in lower for w in sadness_keywords):
            # Shift mass towards sadness/fear when sadness-related tokens present
            text_probs = {"sad": 0.45, "fear": 0.35, "neutral": 0.1, "happy": 0.03, "angry": 0.03, "disgust": 0.02, "surprise": 0.02}
        total = sum(text_probs.values())
        text_probs = {k: v / total for k, v in text_probs.items()}
        confidence = max(text_probs.values())
        context = "Feeling overwhelmed" if text_probs["sad"] > 0.3 else "Anxious" if text_probs["fear"] > 0.3 else "Neutral"
        return {"final_emotion": max(text_probs, key=text_probs.get), "context_emotion": context, "confidence": confidence, "text_probs": text_probs}

    # If Vertex returned something, try to extract the JSON portion and parse it
    import re, json
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        # If no JSON-like substring, fall back to neutral output
        return {"final_emotion": "neutral", "context_emotion": "neutral", "confidence": 0.0, "text_probs": {"neutral": 1.0}}
    jt = m.group(0)
    try:
        parsed = json.loads(jt)
    except Exception:
        # Try some naive string repairs and parse again
        jt_fixed = jt.replace("'", "\"")
        jt_fixed = re.sub(r",\s*([}```])", r"\1", jt_fixed)
        try:
            parsed = json.loads(jt_fixed)
        except Exception:
            # If parsing still fails, return neutral fallback
            return {"final_emotion": "neutral", "context_emotion": "neutral", "confidence": 0.0, "text_probs": {"neutral": 1.0}}

    # Ensure all expected labels exist and normalize probabilities
    labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
    tprobs = {}
    for l in labels:
        val = parsed.get("text_probs", {}).get(l) if isinstance(parsed.get("text_probs", {}), dict) else None
        try:
            tprobs[l] = float(val) if val is not None else 0.0
        except Exception:
            tprobs[l] = 0.0
    total = sum(tprobs.values()) or 1.0
    tprobs = {k: v / total for k, v in tprobs.items()}
    confidence = max(tprobs.values())
    return {"final_emotion": parsed.get("final_emotion", "neutral"), "context_emotion": parsed.get("context_emotion", "neutral"), "confidence": confidence, "text_probs": tprobs}


def derive_intensity(features):
    """
    Derive a coarse energy/intensity level from audio features.

    Heuristic mapping:
    - Very low RMS and low tempo -> "low-energy"
    - High RMS or very fast tempo -> "high-energy"
    - Otherwise -> "medium-energy"
    """
    rms = features.get("rms", 0.0)
    tempo = features.get("tempo", 0.0)
    if rms < 0.007 and tempo < 100:
        return "low-energy"
    elif rms > 0.015 or tempo > 130:
        return "high-energy"
    else:
        return "medium-energy"


def fuse_emotions(asr_conf, text_probs, audio_feats):
    """
    Fuse text-based probabilities and audio-based heuristic probabilities.

    Inputs:
    - asr_conf: confidence score from ASR (0.0-1.0)
    - text_probs: dict of emotion probabilities from text classifier
    - audio_feats: extracted audio features (rms, tempo, pitch_mean, ...)

    Fusion strategy (heuristic):
    - Build a pseudo `audio_probs` distribution based on energy, tempo, pitch.
    - Normalize audio_probs.
    - If text_probs contains a reasonably confident signal (>0.1 mass on any label),
      assign a high weight to text (wt_text between 0.6 and 0.95, biased by asr_conf).
    - Otherwise, rely fully on audio (wt_text = 0.0).
    - Compute fused_raw = wt_text * text_probs + wt_audio * audio_probs
    - Normalize fused_raw -> fused_norm
    - final_emotion is the argmax of fused_norm and confidence is its probability.

    Returns:
        final_emotion (str),
        confidence (float),
        audio_probs (dict),
        fused_raw (dict before normalization),
        fused_norm (dict after normalization),
        wt_text (float),
        wt_audio (float)
    """
    labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

    # Extract audio cues; defaults provided to avoid exceptions
    energy = audio_feats.get("rms", 0.0)
    tempo = audio_feats.get("tempo", 0.0)
    pitch = audio_feats.get("pitch_mean", 180.0)

    # Heuristic audio->emotion mapping
    audio_probs = {
        "neutral": max(0.05, 1.0 - energy * 0.8 - tempo / 200.0),
        "happy": max(0.0, energy * 1.2 * (1.0 + (pitch - 180.0) / 360.0)),
        "sad": 0.25 if energy < 0.01 else 0.08,
        "angry": max(0.0, tempo / 150.0 * (1.0 + (pitch - 150.0) / 300.0)),
        "fear": 0.08 + min(0.2, energy * 0.5),
        "disgust": 0.04,
        "surprise": max(0.0, tempo / 120.0 * (1.0 + (pitch - 200.0) / 400.0))
    }

    # Normalize audio_probs
    s = sum(audio_probs.values()) or 1.0
    audio_probs = {k: v / s for k, v in audio_probs.items()}

    # Decide weights: prefer text when there is a meaningful text signal
    has_text = bool(text_probs and any(v > 0.1 for v in text_probs.values()))
    if has_text:
        # asr_conf or default 0.8 biases weight towards text if ASR is confident
        wt_text = max(0.6, min(0.95, float(asr_conf or 0.8)))
        wt_audio = 1.0 - wt_text
    else:
        wt_text = 0.0
        wt_audio = 1.0

    # Combine distributions
    fused_raw = {}
    for l in labels:
        tp = float(text_probs.get(l, 0.0)) if isinstance(text_probs, dict) else 0.0
        fused_raw[l] = wt_text * tp + wt_audio * audio_probs.get(l, 0.0)

    # Normalize fused_raw -> fused_norm
    total = sum(fused_raw.values()) or 1.0
    fused_norm = {k: v / total for k, v in fused_raw.items()}

    # Final decision
    final_emotion = max(fused_norm, key=fused_norm.get)
    confidence = fused_norm[final_emotion]
    return final_emotion, confidence, audio_probs, fused_raw, fused_norm, wt_text, wt_audio


def process_audio_file_bytes(file_bytes: bytes, filename_hint="upload", sample_rate=16000, language_code="en-US"):
    """
    High-level audio processing pipeline.

    Steps performed:
    1. Convert uploaded bytes to WAV (save_upload_and_convert_to_wav).
    2. Extract audio features (extract_audio_features).
    3. Read WAV bytes and run Speech-to-Text (transcribe_wav_bytes).
    4. Detect safety-related words/phrases in transcript (detect_safety).
    5. Classify emotion from transcript (classify_emotion_text).
    6. Fuse audio + text emotion signals into final_emotion (fuse_emotions).
    7. Derive intensity (derive_intensity).
    8. Build meta and result JSON and return it.

    On any exception during processing, returns an error-fallback JSON that describes the failure
    in a predictable neutral way (so downstream systems can still operate).
    """
    try:
        # 1) Save and convert upload to a standardized WAV
        wav_path = save_upload_and_convert_to_wav(file_bytes, filename_hint=filename_hint)

        # 2) Extract audio features (duration, rms, tempo, pitch_mean)
        feats = extract_audio_features(wav_path)

        # 3) Read raw WAV bytes to pass to the Speech-to-Text helper
        with open(wav_path, "rb") as fh:
            wav_bytes = fh.read()

        # 4) Transcribe using Google Speech-to-Text helper (returns transcript, asr_confidence)
        transcript, asr_conf = transcribe_wav_bytes(wav_bytes, sample_rate=sample_rate, language_code=language_code)
        if transcript is None:
            transcript = ""

        # 5) Safety detection (returns bool flag and matched keyword if any)
        safety_flag, safety_keyword = detect_safety(transcript)

        # 6) Classify emotion from the text (Vertex preferred, heuristics fallback)
        text_emotion = classify_emotion_text(transcript)

        # 7) Fuse text emotion with audio-derived probabilities to get final emotion
        final_emotion, confidence, audio_probs, fused_raw, fused_norm, wt_text, wt_audio = fuse_emotions(
            asr_conf, text_emotion.get("text_probs", {}), feats
        )

        # 8) Derive intensity (low/medium/high energy)
        intensity = derive_intensity(feats)

        # 9) Build metadata useful for debugging or analytics
        meta = {
            "reason": "weighted_fusion",
            "asr_conf": float(asr_conf or 0.0),
            "wt_text": float(wt_text),
            "wt_audio": float(wt_audio),
            "audio_probs": audio_probs,
            "text_probs": text_emotion.get("text_probs", {}),
            "fused_raw": fused_raw,
            "fused_norm": fused_norm,
            "features": feats
        }

        # 10) Final structured result returned to caller
        result_json = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "transcript": transcript,
            "final_emotion": final_emotion,
            "intensity": intensity,
            "context_emotion": text_emotion.get("context_emotion", "neutral"),
            "confidence": float(confidence),
            "meta": meta,
            "audio_features": feats,
            "safety_flag": safety_flag,
            "safety_keyword": safety_keyword
        }
        return result_json

    except Exception as e:
        # Log full stack trace to help diagnose root cause
        _logger.error("Audio processing pipeline failed: %s", e, exc_info=True)
        # Return a deterministic neutral fallback to keep downstream code robust
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "transcript": "",
            "final_emotion": "neutral",
            "intensity": "medium-energy",
            "context_emotion": "Audio processing error",
            "confidence": 0.0,
            "meta": {
                "reason": "error_fallback",
                "asr_conf": 0.0,
                "wt_text": 0.0,
                "wt_audio": 1.0,
                "audio_probs": {"neutral": 1.0},
                "text_probs": {"neutral": 1.0},
                "fused_raw": {"neutral": 1.0},
                "fused_norm": {"neutral": 1.0},
                "features": {"duration_s": 0.0, "rms": 0.0, "tempo": 0.0, "pitch_mean": 0.0, "sr": 16000}
            },
            "audio_features": {"duration_s": 0.0, "rms": 0.0, "tempo": 0.0, "pitch_mean": 0.0, "sr": 16000},
            "safety_flag": False,
            "safety_keyword": None
        }
