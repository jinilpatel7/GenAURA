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

# NEW: Suppress the pkg_resources deprecation warning from librosa
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

from .gcp_clients import transcribe_wav_bytes, vertex_generate
from .utils import detect_safety

if not hasattr(scipy.signal, "hann"):
    from scipy.signal import windows
    scipy.signal.hann = windows.hann

_logger = logging.getLogger(__name__)
TMP_DIR = "/tmp/empathetic_ai"
os.makedirs(TMP_DIR, exist_ok=True)

def save_upload_and_convert_to_wav(file_bytes: bytes, filename_hint="upload"):
    uid = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR, f"{uid}_{filename_hint}")
    with open(input_path, "wb") as fh:
        fh.write(file_bytes)
    
    out_wav = os.path.join(TMP_DIR, f"{uid}.wav")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-vn", out_wav]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_wav
    except Exception as e:
        _logger.warning("ffmpeg conversion failed: %s — trying librosa fallback", e)
        try:
            import soundfile as sf
            y, sr = librosa.load(input_path, sr=16000, mono=True)
            sf.write(out_wav, y, 16000, subtype="PCM_16")
            return out_wav
        except Exception as e2:
            _logger.error("Fallback conversion failed: %s", e2)
            raise

def extract_audio_features(wav_path: str, target_sr=16000):
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    
    try:
        rms = float(np.mean(librosa.feature.rms(y=y)))
    except Exception:
        rms = 0.0
    
    try:
        pitch = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        pitch_mean = float(np.nanmean(pitch))
        if np.isnan(pitch_mean):
            pitch_mean = 0.0
    except Exception:
        pitch_mean = 0.0
    
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
    # Attempt Vertex; fallback heuristics (same as before)
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
    if not raw:
        lower = transcript.lower()
        text_probs = {"neutral": 0.6, "happy": 0.05, "sad": 0.15, "angry": 0.05, "fear": 0.1, "disgust": 0.02, "surprise": 0.03}
        sadness_keywords = ["sad","depress","down","upset","cry","stressed","stress","overwhelm","overwhelmed","hopeless"]
        if any(w in lower for w in sadness_keywords):
            text_probs = {"sad": 0.45,"fear":0.35,"neutral":0.1,"happy":0.03,"angry":0.03,"disgust":0.02,"surprise":0.02}
        total=sum(text_probs.values())
        text_probs={k:v/total for k,v in text_probs.items()}
        confidence=max(text_probs.values())
        context="Feeling overwhelmed" if text_probs["sad"]>0.3 else "Anxious" if text_probs["fear"]>0.3 else "Neutral"
        return {"final_emotion": max(text_probs, key=text_probs.get),"context_emotion":context,"confidence":confidence,"text_probs":text_probs}
    # parse as before
    import re, json
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"final_emotion":"neutral","context_emotion":"neutral","confidence":0.0,"text_probs":{"neutral":1.0}}
    jt = m.group(0)
    try:
        parsed = json.loads(jt)
    except Exception:
        jt_fixed = jt.replace("'", "\"")
        jt_fixed = re.sub(r",\s*([}```])", r"\1", jt_fixed)
        try:
            parsed = json.loads(jt_fixed)
        except Exception:
            return {"final_emotion":"neutral","context_emotion":"neutral","confidence":0.0,"text_probs":{"neutral":1.0}}
    labels=["neutral","happy","sad","angry","fear","disgust","surprise"]
    tprobs={}
    for l in labels:
        val = parsed.get("text_probs",{}).get(l) if isinstance(parsed.get("text_probs",{}), dict) else None
        try:
            tprobs[l]=float(val) if val is not None else 0.0
        except Exception:
            tprobs[l]=0.0
    total=sum(tprobs.values()) or 1.0
    tprobs={k:v/total for k,v in tprobs.items()}
    confidence=max(tprobs.values())
    return {"final_emotion": parsed.get("final_emotion","neutral"),"context_emotion":parsed.get("context_emotion","neutral"),"confidence":confidence,"text_probs":tprobs}

def derive_intensity(features):
    rms = features.get("rms", 0.0)
    tempo = features.get("tempo", 0.0)
    if rms < 0.007 and tempo < 100:
        return "low-energy"
    elif rms > 0.015 or tempo > 130:
        return "high-energy"
    else:
        return "medium-energy"

def fuse_emotions(asr_conf, text_probs, audio_feats):
    labels = ["neutral","happy","sad","angry","fear","disgust","surprise"]
    energy = audio_feats.get("rms",0.0)
    tempo = audio_feats.get("tempo",0.0)
    pitch = audio_feats.get("pitch_mean", 180.0)
    audio_probs = {
        "neutral": max(0.05, 1.0 - energy * 0.8 - tempo / 200.0),
        "happy": max(0.0, energy * 1.2 * (1.0 + (pitch - 180.0) / 360.0)),
        "sad": 0.25 if energy < 0.01 else 0.08,
        "angry": max(0.0, tempo / 150.0 * (1.0 + (pitch - 150.0) / 300.0)),
        "fear": 0.08 + min(0.2, energy * 0.5),
        "disgust": 0.04,
        "surprise": max(0.0, tempo / 120.0 * (1.0 + (pitch - 200.0) / 400.0))
    }
    s=sum(audio_probs.values()) or 1.0
    audio_probs={k:v/s for k,v in audio_probs.items()}
    has_text = bool(text_probs and any(v>0.1 for v in text_probs.values()))
    if has_text:
        wt_text = max(0.6, min(0.95, float(asr_conf or 0.8)))
        wt_audio = 1.0 - wt_text
    else:
        wt_text = 0.0
        wt_audio = 1.0
    fused_raw={}
    for l in labels:
        tp=float(text_probs.get(l,0.0)) if isinstance(text_probs, dict) else 0.0
        fused_raw[l]=wt_text*tp + wt_audio*audio_probs.get(l,0.0)
    total=sum(fused_raw.values()) or 1.0
    fused_norm={k:v/total for k,v in fused_raw.items()}
    final_emotion = max(fused_norm, key=fused_norm.get)
    confidence = fused_norm[final_emotion]
    return final_emotion, confidence, audio_probs, fused_raw, fused_norm, wt_text, wt_audio

def process_audio_file_bytes(file_bytes: bytes, filename_hint="upload", sample_rate=16000, language_code="en-US"):
    try:
        wav_path = save_upload_and_convert_to_wav(file_bytes, filename_hint=filename_hint)
        feats = extract_audio_features(wav_path)
        with open(wav_path, "rb") as fh:
            wav_bytes = fh.read()
        transcript, asr_conf = transcribe_wav_bytes(wav_bytes, sample_rate=sample_rate, language_code=language_code)
        if transcript is None:
            transcript=""
        safety_flag, safety_keyword = detect_safety(transcript)
        text_emotion = classify_emotion_text(transcript)
        final_emotion, confidence, audio_probs, fused_raw, fused_norm, wt_text, wt_audio = fuse_emotions(asr_conf, text_emotion.get("text_probs",{}), feats)
        intensity = derive_intensity(feats)
        meta = {
            "reason": "weighted_fusion",
            "asr_conf": float(asr_conf or 0.0),
            "wt_text": float(wt_text),
            "wt_audio": float(wt_audio),
            "audio_probs": audio_probs,
            "text_probs": text_emotion.get("text_probs",{}),
            "fused_raw": fused_raw,
            "fused_norm": fused_norm,
            "features": feats
        }
        result_json = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "transcript": transcript,
            "final_emotion": final_emotion,
            "intensity": intensity,
            "context_emotion": text_emotion.get("context_emotion","neutral"),
            "confidence": float(confidence),
            "meta": meta,
            "audio_features": feats,
            "safety_flag": safety_flag,
            "safety_keyword": safety_keyword
        }
        return result_json
    except Exception as e:
        _logger.error("Audio processing pipeline failed: %s", e, exc_info=True)
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
                "audio_probs": {"neutral":1.0},
                "text_probs": {"neutral":1.0},
                "fused_raw": {"neutral":1.0},
                "fused_norm": {"neutral":1.0},
                "features": {"duration_s":0.0,"rms":0.0,"tempo":0.0,"pitch_mean":0.0,"sr":16000}
            },
            "audio_features": {"duration_s":0.0,"rms":0.0,"tempo":0.0,"pitch_mean":0.0,"sr":16000},
            "safety_flag": False,
            "safety_keyword": None
        }