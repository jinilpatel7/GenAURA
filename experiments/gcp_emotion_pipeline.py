import os
import time
import json
import re
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import scipy.signal
import logging
from datetime import datetime
from google.cloud import speech
import vertexai
from vertexai.generative_models import GenerativeModel

# --- Fix for librosa expecting scipy.signal.hann (removed in SciPy >=1.8)
if not hasattr(scipy.signal, "hann"):
    from scipy.signal import windows
    scipy.signal.hann = windows.hann

# --- Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# --- GCP / Vertex Config ---
PROJECT_ID = "mind-sail-471005"
LOCATION = "us-central1"

# Init Vertex AI (will use GOOGLE_APPLICATION_CREDENTIALS env var)
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except Exception as e:
    logging.warning("vertexai.init() raised an exception (maybe credentials or env). Continuing — ensure env vars are set. Error: %s", e)

# ----------------------------
# Audio Recording
# ----------------------------
def record_audio(filename="demo.wav", duration=6, sr=16000):
    """
    Record audio from default microphone and save as 16-bit PCM WAV.
    """
    logging.info("Recording %ds @ %dHz...", duration, sr)
    data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    # Flatten to mono 1D array
    data_mono = np.squeeze(data)
    # Save as PCM_16 so GCP LINEAR16 encoding matches file contents
    sf.write(filename, data_mono, sr, subtype="PCM_16")
    logging.info("Saved: %s", filename)
    return filename

# ----------------------------
# Audio Feature Extraction
# ----------------------------
def extract_audio_features(wav_path: str, target_sr=16000):
    """
    Load audio (resample to target_sr) and compute a few features.
    """
    # Use a fixed SR to standardize behaviour
    try:
        y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    except Exception as e:
        logging.error("librosa.load failed: %s", e)
        raise

    duration = float(librosa.get_duration(y=y, sr=sr))
    # RMS: mean energy
    try:
        rms = float(np.mean(librosa.feature.rms(y=y)))
    except Exception:
        rms = 0.0

    # Pitch (YIN) — guard against short or silent audio
    try:
        pitch = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        pitch_mean = float(np.nanmean(pitch))
        if np.isnan(pitch_mean):
            pitch_mean = 0.0
    except Exception:
        pitch_mean = 0.0

    # Tempo estimate
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

# ----------------------------
# Speech-to-Text (ASR) via GCP
# ----------------------------
def transcribe_audio_gcp(wav_path: str):
    """
    Transcribe WAV using synchronous Google Cloud Speech-to-Text.
    Returns (transcript: str or None, confidence: float between 0 and 1)
    """
    client = speech.SpeechClient()

    with open(wav_path, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    # Since we saved PCM_16, LINEAR16 is appropriate. If you change saving,
    # consider using ENCODING_UNSPECIFIED for autodetection.
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    try:
        response = client.recognize(config=config, audio=audio)
    except Exception as e:
        logging.error("GCP speech.recognize failed: %s", e)
        return None, 0.0

    if not response.results:
        logging.warning("ASR returned no results.")
        return None, 0.0

    # Take first alternative of first result (synchronous recognize returns this)
    result = response.results[0].alternatives[0]
    transcript = result.transcript
    # Some responses may not include confidence; default to 0.0
    confidence = float(getattr(result, "confidence", 0.0))
    return transcript, confidence

# ----------------------------
# Emotion Classification (Vertex) with robust JSON extraction
# ----------------------------
def classify_emotion_with_vertex(transcript: str, model_name="gemini-2.5-flash", debug_save="vertex_raw_response.txt"):
    """
    Send the transcript to Vertex's generative model and parse a JSON response.
    Returns a dict with keys: final_emotion, context_emotion, confidence, text_probs (dict)
    If parsing fails, returns a safe fallback (text_probs empty).
    """

    # If transcript is None or empty, early return fallback
    if not transcript:
        return {
            "final_emotion": "neutral",
            "context_emotion": "neutral",
            "confidence": 0.0,
            "text_probs": {},
        }

    # Make a stricter prompt that requests only JSON. We still defensively parse.
    prompt = (
        "You are an expert emotion classifier. RETURN ONLY a single JSON object and nothing else.\n"
        "The JSON must contain these keys:\n"
        '  "final_emotion": one of [neutral,happy,sad,angry,fear,disgust,surprise],\n'
        '  "context_emotion": string (short),\n'
        '  "confidence": float between 0 and 1,\n'
        '  "text_probs": { "neutral": float, "happy": float, "sad": float, "angry": float, "fear": float, "disgust": float, "surprise": float }\n\n'
        "Respond ONLY with the JSON object. Here is the text to analyze:\n\n"
        + json.dumps(transcript)
    )

    try:
        model = GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        raw = getattr(resp, "text", str(resp))
    except Exception as e:
        logging.error("Vertex model.generate_content failed: %s", e)
        return {
            "final_emotion": "neutral",
            "context_emotion": "neutral",
            "confidence": 0.0,
            "text_probs": {},
        }

    # Save raw response for debugging
    try:
        with open(debug_save, "w", encoding="utf-8") as fh:
            fh.write(raw)
    except Exception:
        logging.debug("Could not write Vertex raw response to %s", debug_save)

    # Try to extract a JSON object from the raw text.
    # Simple greedy approach: first '{' to last '}' in the text.
    # This works in the common case where the model prints JSON somewhere in its output.
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        logging.warning("Could not find JSON object in Vertex response. Raw response saved to %s", debug_save)
        return {
            "final_emotion": "neutral",
            "context_emotion": "neutral",
            "confidence": 0.0,
            "text_probs": {},
        }

    json_text = m.group(0)
    try:
        parsed = json.loads(json_text)
    except Exception as e:
        logging.warning("json.loads failed on extracted JSON. Error: %s", e)
        # Try a relaxed replacement of trailing commas -> none, or single quotes -> double quotes
        json_text_fixed = json_text.replace("'", "\"")
        json_text_fixed = re.sub(r",\s*([}\]])", r"\1", json_text_fixed)
        try:
            parsed = json.loads(json_text_fixed)
        except Exception as e2:
            logging.error("Relaxed json.loads also failed: %s", e2)
            return {
                "final_emotion": "neutral",
                "context_emotion": "neutral",
                "confidence": 0.0,
                "text_probs": {},
            }

    # Ensure required fields exist and text_probs is a dict
    parsed.setdefault("final_emotion", "neutral")
    parsed.setdefault("context_emotion", "neutral")
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("text_probs", {})

    # Normalize numeric types
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
    except Exception:
        parsed["confidence"] = 0.0

    # Ensure text_probs values are floats and contain the expected labels (or at least empty)
    expected_labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
    tprobs = {}
    for lab in expected_labels:
        val = parsed["text_probs"].get(lab) if isinstance(parsed["text_probs"], dict) else None
        try:
            tprobs[lab] = float(val) if val is not None else 0.0
        except Exception:
            tprobs[lab] = 0.0
    parsed["text_probs"] = tprobs

    return parsed

# ----------------------------
# Intensity Derivation (Audio Features)
# ----------------------------
def derive_intensity(features):
    rms = features.get("rms", 0.0)
    tempo = features.get("tempo", 0.0)

    if rms < 0.007 and tempo < 100:
        return "low-energy"
    elif rms > 0.015 or tempo > 130:
        return "high-energy"
    else:
        return "medium-energy"

# ----------------------------
# Weighted Fusion of Audio + Text
# ----------------------------
def fuse_emotions(asr_conf, text_probs, audio_feats):
    """
    Returns:
      final_emotion (str),
      confidence (float normalized 0..1),
      audio_probs (dict normalized),
      fused_raw (dict raw weighted scores),
      fused_norm (dict normalized probabilities),
      wt_text, wt_audio
    """
    labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

    # Approximate audio_probs using features
    energy = audio_feats.get("rms", 0.0)
    tempo = audio_feats.get("tempo", 0.0)

    audio_probs = {
        "neutral": max(0.1, 1 - energy - tempo / 200.0),
        "happy": max(0.0, energy * 2),
        "sad": 0.3 if energy < 0.01 else 0.1,
        "angry": max(0.0, tempo / 200.0),
        "fear": 0.1,
        "disgust": 0.1,
        "surprise": max(0.0, tempo / 150.0),
    }
    s = sum(audio_probs.values()) or 1.0
    audio_probs = {k: v / s for k, v in audio_probs.items()}

    # If text_probs is empty, fall back to audio-only (wt_audio = 1.0)
    has_text = bool(text_probs and any((v for v in text_probs.values())))
    if has_text:
        wt_text = float(asr_conf)
        wt_audio = 1.0 - wt_text
    else:
        wt_text = 0.0
        wt_audio = 1.0

    fused_raw = {}
    for l in labels:
        tp = float(text_probs.get(l, 0.0)) if isinstance(text_probs, dict) else 0.0
        fused_raw[l] = wt_text * tp + wt_audio * audio_probs.get(l, 0.0)

    total = sum(fused_raw.values()) or 1.0
    fused_norm = {k: v / total for k, v in fused_raw.items()}

    final_emotion = max(fused_norm, key=fused_norm.get)
    confidence = fused_norm[final_emotion]

    return final_emotion, confidence, audio_probs, fused_raw, fused_norm, wt_text, wt_audio

# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    wav = record_audio()
    feats = extract_audio_features(wav)
    logging.info("Audio Features: %s", feats)

    transcript, asr_conf = transcribe_audio_gcp(wav)
    logging.info("Transcript: %s | ASR_conf: %s", transcript, asr_conf)

    text_emotion = classify_emotion_with_vertex(transcript)

    final_emotion, confidence, audio_probs, fused_raw, fused_norm, wt_text, wt_audio = fuse_emotions(
        asr_conf, text_emotion.get("text_probs", {}), feats
    )

    intensity = derive_intensity(feats)

    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "transcript": transcript,
        "final_emotion": final_emotion,
        "intensity": intensity,
        "context_emotion": text_emotion.get("context_emotion"),
        "confidence": confidence,
        "meta": {
            "reason": "weighted_fusion",
            "asr_conf": asr_conf,
            "wt_text": wt_text,
            "wt_audio": wt_audio,
            "audio_probs": audio_probs,
            "text_probs": text_emotion.get("text_probs", {}),
            "fused_raw": fused_raw,
            "fused_norm": fused_norm,
            "features": feats,
        },
        "audio_features": feats,
        "safety_flag": False,
        "safety_keyword": None,
    }

    print("\n=== RESULT JSON ===")
    print(json.dumps(result, indent=2))
    print("===================")

if __name__ == "__main__":
    main()
