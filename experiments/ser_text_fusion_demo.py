# setx GOOGLE_APPLICATION_CREDENTIALS "C:\keys\genaura77-sa-key.json" ; setx GCP_PROJECT "genaura77"
# cd C:\Users\jinil\Desktop\experiments


import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-2025-08-25-git-1b62f9d3ae-full_build\bin"

import time, json, warnings
import numpy as np
import sounddevice as sd, soundfile as sf
import librosa
from google.cloud import speech_v1p1beta1 as speech
from transformers import pipeline

# Try imageio-ffmpeg to help with ffmpeg path inside venv (optional)
try:
    import imageio_ffmpeg
    ffexe = imageio_ffmpeg.get_ffmpeg_exe()
    if ffexe:
        os.environ["PATH"] += os.pathsep + os.path.dirname(ffexe)
except Exception:
    pass

# --- Compatibility patch for scipy hann used by librosa (if needed) ---
try:
    import scipy.signal
    if not hasattr(scipy.signal, "hann"):
        import numpy as _np
        scipy.signal.hann = lambda M: _np.hanning(M)
except Exception:
    pass

# ---------- CONFIG ----------
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
LOCAL_FILE = "demo.wav"

# Hugging Face models (changeable)
AUDIO_SER_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Canonical labels we want to output
CANONICAL_LABELS = [
    "neutral","happy","sad","angry","fear","surprise","disgust",
    "energetic","calm","hurried"
]

# Label mapping from model-specific labels -> canonical labels
# Adjust if your models use different label names
AUDIO_LABEL_MAP = {
    "calm": "calm",
    "angry": "angry",
    "disgust": "disgust",
    "fearful": "fear",
    "sad": "sad",
    "neutral": "neutral",
    "surprised": "surprise",
    "happy": "happy",
    # fallback acoustic-derived labels (if model uses these)
    "energetic": "energetic",
    "excited": "energetic"
}

TEXT_LABEL_MAP = {
    "joy": "happy",
    "happy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "neutral": "neutral",
    # some text models use slightly different names - map them if needed
    "love": "happy"
}

# Fusion hyperparams
MIN_TEXT_DOMINANCE_CONF = 0.85   # if text prob for top label >= this AND ASR_conf high, prefer text
ASR_CONF_THRESHOLD = 0.5         # minimum ASR confidence to trust text strongly
AROUSAL_RMS_THRESHOLD = 0.01     # heuristic thresholds for fallback
AROUSAL_TEMPO_THRESHOLD = 120.0

# Safety keywords (very small triage): if detected, set 'safety_risk' flag
SAFETY_KEYWORDS = [
    "kill myself", "i will kill myself", "i'm going to kill myself", "suicide",
    "i want to die", "hurt myself", "i'm going to kill you", "i will kill you",
    "i'm going to hurt you", "i'm going to hurt myself"
]
# ----------------------------

def record_wav(path=LOCAL_FILE, duration=RECORD_SECONDS, sr=SAMPLE_RATE):
    print(f"Recording {duration}s — speak now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio, sr, subtype="PCM_16")
    print("Saved:", path)
    return path

def extract_audio_features(path):
    """Return a small set of low-level audio features for fallback / diagnostics."""
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return {"duration_s": 0.0, "rms": 0.0, "tempo": 0.0, "pitch_mean": 0.0}
    rms = float(np.mean(librosa.feature.rms(y=y)))
    # tempo guard (may fail on short speech)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except Exception:
        tempo = 0.0
    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        pitch_mean = float(np.nanmean(f0)) if f0 is not None else 0.0
    except Exception:
        pitch_mean = 0.0
    duration_s = float(librosa.get_duration(y=y, sr=sr))
    return {"duration_s": duration_s, "rms": rms, "tempo": float(tempo), "pitch_mean": pitch_mean}

def transcribe_google(path):
    with open(path, "rb") as f:
        content = f.read()
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        model="latest_short",
        use_enhanced=True
    )
    resp = client.recognize(config=config, audio=audio)
    transcript = ""
    asr_conf = 0.0
    for r in resp.results:
        alt = r.alternatives[0]
        transcript += alt.transcript + " "
        asr_conf = max(asr_conf, float(getattr(alt, "confidence", 0.0)))
    return transcript.strip(), asr_conf

# Load HF pipelines (may take a moment)
print("Loading Hugging Face pipelines (may take some time)...")
audio_pipe = pipeline("audio-classification", model=AUDIO_SER_MODEL, top_k=None)
text_pipe = pipeline("text-classification", model=TEXT_MODEL, top_k=None)

def normalize_and_map_probs(raw_probs, mapping):
    """
    raw_probs: dict label->score from HF
    mapping: mapping from raw label to canonical label
    returns: dict canonical_label -> normalized prob (sums to 1)
    """
    canon = {}
    for lbl,score in raw_probs.items():
        key = mapping.get(lbl.lower(), None)
        if key is None:
            # try exact-case or partial matching
            low = lbl.lower()
            for k in mapping:
                if k in low:
                    key = mapping[k]; break
        if key is None:
            # skip unknown labels
            continue
        canon[key] = canon.get(key, 0.0) + float(score)
    # normalize to sum 1 (if any)
    s = sum(canon.values()) or 0.0
    if s > 0:
        for k in list(canon.keys()):
            canon[k] = canon[k] / s
    return canon

def run_audio_ser(path):
    """Run HF audio-classification pipeline and return canonical probs dict."""
    preds = audio_pipe(path)
    # preds is often a list of dicts {label, score}
    raw = {}
    for p in preds:
        raw[p['label']] = float(p['score'])
    return normalize_and_map_probs(raw, AUDIO_LABEL_MAP)

def run_text_emotion(text):
    """Run HF text-classification pipeline and return canonical probs dict."""
    if not text:
        return {}
    res = text_pipe(text)
    # res might be nested; handle both forms
    raw = {}
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        # nested lists (per-sentence)
        for item in res[0]:
            raw[item['label']] = float(item['score'])
    elif isinstance(res, list):
        for item in res:
            raw[item['label']] = float(item['score'])
    else:
        # unexpected form
        pass
    return normalize_and_map_probs(raw, TEXT_LABEL_MAP)

def safety_check_text(text):
    t = text.lower()
    for kw in SAFETY_KEYWORDS:
        if kw in t:
            return True, kw
    return False, None

def arousal_fallback_label(features):
    """Return a coarse label using raw audio features when models are weak."""
    rms = features.get("rms", 0.0)
    tempo = features.get("tempo", 0.0)
    # simple heuristics
    if rms > AROUSAL_RMS_THRESHOLD and tempo > AROUSAL_TEMPO_THRESHOLD:
        return "energetic", 0.6
    if rms < (AROUSAL_RMS_THRESHOLD*0.5) and tempo < (AROUSAL_TEMPO_THRESHOLD*0.6):
        return "calm", 0.5
    return "neutral", 0.4

def fuse_probabilities(audio_probs, text_probs, asr_conf, features):
    """
    Smart fusion:
    - If ASR confidence is high, trust text more (weighted).
    - If text has a very dominant label and ASR_conf high, use text override.
    - If audio_probs missing, use text_probs (and vice versa).
    - If both missing/flat, use arousal_fallback_label.
    Returns final_label, final_confidence, debug_meta
    """
    # ensure keys present
    a = {k: float(v) for k,v in (audio_probs or {}).items()}
    t = {k: float(v) for k,v in (text_probs or {}).items()}

    # If nothing present -> arousal fallback
    if not a and not t:
        lbl, conf = arousal_fallback_label(features)
        return lbl, conf, {"reason":"arousal_fallback", "features":features}

    # If one missing, return the other (but adjust confidence by asr_conf)
    if not a and t:
        # trust text; confidence scales with ASR
        top = max(t.items(), key=lambda x: x[1])
        final_conf = top[1] * (0.5 + 0.5*asr_conf)  # scale 0.5..1.0 by ASR
        return top[0], float(final_conf), {"reason":"text_only", "text_probs":t, "asr_conf":asr_conf}
    if not t and a:
        top = max(a.items(), key=lambda x: x[1])
        final_conf = top[1] * 0.9
        return top[0], float(final_conf), {"reason":"audio_only", "audio_probs":a}

    # both present -> build aligned vectors over canonical labels
    labels = CANONICAL_LABELS
    a_vec = np.array([a.get(lbl, 0.0) for lbl in labels], dtype=float)
    t_vec = np.array([t.get(lbl, 0.0) for lbl in labels], dtype=float)

    # If text has a dominant label and ASR high -> override
    if asr_conf >= ASR_CONF_THRESHOLD:
        top_label = labels[int(np.argmax(t_vec))]
        top_score = float(np.max(t_vec))
        if top_score >= MIN_TEXT_DOMINANCE_CONF:
            return top_label, float(0.9 * top_score + 0.1 * asr_conf), {"reason":"text_dominant_override", "text_top_score":top_score, "asr_conf":asr_conf}

    # Weighted blending controlled by ASR confidence:
    # trust_text_weight = asr_conf (0..1)
    wt_text = float(asr_conf)
    wt_audio = 1.0 - wt_text
    fused_vec = wt_audio * a_vec + wt_text * t_vec

    # Normalize fused vector
    s = float(np.sum(fused_vec)) or 1.0
    fused_norm = fused_vec / s

    idx = int(np.argmax(fused_norm))
    final_label = labels[idx]
    final_conf = float(fused_norm[idx])  # in 0..1
    debug = {
        "reason":"weighted_fusion",
        "asr_conf": asr_conf,
        "wt_text": wt_text,
        "wt_audio": wt_audio,
        "audio_probs": a,
        "text_probs": t,
        "fused_raw": fused_vec.tolist(),
        "fused_norm": fused_norm.tolist(),
        "features": features
    }
    return final_label, final_conf, debug

# --- New Helpers ---
def derive_intensity(label, features):
    """
    Derive an 'intensity' dimension (low-energy, high-energy, medium) 
    from features + emotion label.
    """
    rms = features.get("rms", 0.0)
    tempo = features.get("tempo", 0.0)

    if rms < 0.007 and tempo < 100:
        return "low-energy"
    elif rms > 0.015 or tempo > 130:
        return "high-energy"
    else:
        return "medium-energy"

def derive_context_emotion(label, transcript):
    """
    Derive a blended contextual emotion description using transcript + emotion.
    Very lightweight heuristic; for production, you may want an LLM here.
    """
    text = transcript.lower()
    ctx = label

    if "alone" in text or "no one" in text or "lonely" in text:
        ctx = f"lonely but {label}"
    elif "tired" in text or "exhausted" in text:
        ctx = f"tired and {label}"
    elif "excited" in text or "can't wait" in text:
        ctx = f"excited and {label}"
    # default: just return the label
    return ctx

# --- Update build_result_json ---
def build_result_json(label, conf, meta, transcript, features, safety_flag, safety_kw):
    intensity = derive_intensity(label, features)
    context_emotion = derive_context_emotion(label, transcript)

    out = {
        "timestamp": time.time(),
        "transcript": transcript,
        "final_emotion": label,
        "intensity": intensity,
        "context_emotion": context_emotion,
        "confidence": conf,
        "meta": meta,
        "audio_features": features,
        "safety_flag": safety_flag,
        "safety_keyword": safety_kw
    }
    return out


# ----------------- Main -----------------
def main():
    wav = record_wav()
    features = extract_audio_features(wav)
    print("Audio features:", features)

    transcript, asr_conf = "", 0.0
    try:
        transcript, asr_conf = transcribe_google(wav)
    except Exception as e:
        print("ASR failed:", e)
    print("Transcript:", transcript, "| ASR_conf:", asr_conf)

    # safety triage from transcript
    safety_flag, safety_kw = safety_check_text(transcript)

    try:
        audio_probs = run_audio_ser(wav)
    except Exception as e:
        print("Audio SER failed:", e)
        audio_probs = {}

    try:
        text_probs = run_text_emotion(transcript)
    except Exception as e:
        print("Text emotion failed:", e)
        text_probs = {}

    print("Audio SER probs:", audio_probs)
    print("Text emotion probs:", text_probs)

    label, conf, debug_meta = fuse_probabilities(audio_probs, text_probs, asr_conf, features)
    result = build_result_json(label, conf, debug_meta, transcript, features, safety_flag, safety_kw)

    # Print pretty JSON for inspection (also easy to pass to LLM)
    print("\n=== RESULT JSON ===")
    print(json.dumps(result, indent=2))
    print("===================\n")

    # If safety_flag true, print urgent notice (you should add escalation logic in production)
    if safety_flag:
        print("!!! SAFETY FLAG TRIGGERED:", safety_kw)
        print("Take immediate escalation steps (show resources / human escalation).")

if __name__ == "__main__":
    main()
