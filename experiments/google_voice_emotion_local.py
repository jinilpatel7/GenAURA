# Experimenting with Google Cloud Speech-to-Text + Natural Language for local emotion recognition locally
import io
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import language_v1
import scipy.signal
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = lambda M: scipy.signal.windows.hann(M)

# CONFIG
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
LOCAL_FILE = "sample_record.wav"

def record_audio_to_wav_bytes(duration=RECORD_SECONDS, sr=SAMPLE_RATE):
    """Record from mic and return WAV bytes (PCM16). Also save local file for librosa."""
    print(f"\nRecording {duration} seconds — speak now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = bio.getvalue()
    with open(LOCAL_FILE, "wb") as f:
        f.write(wav_bytes)
    print(f"Saved local file: {LOCAL_FILE}")
    return wav_bytes

def transcribe_with_google(wav_bytes, sample_rate=SAMPLE_RATE, language_code="en-US"):
    """Send WAV bytes to Google STT sync recognize; return transcript, words list, and ASR confidence."""
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=wav_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        use_enhanced=True,
        model="latest_short"
    )
    resp = client.recognize(config=config, audio=audio)
    transcript = ""
    words = []
    overall_conf = 0.0
    if not resp.results:
        return "", [], 0.0
    for result in resp.results:
        alt = result.alternatives[0]
        transcript += alt.transcript + " "
        overall_conf = max(overall_conf, alt.confidence)
        for w in alt.words:
            words.append({"word": w.word, "start_time": w.start_time.total_seconds(), "end_time": w.end_time.total_seconds()})
    transcript = transcript.strip()
    return transcript, words, float(overall_conf)

def extract_audio_features_from_wavfile(path, sr=SAMPLE_RATE):
    """Return robust prosodic & spectral features (pyin, rms, mfcc, tempo, pauses, elongation)."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    hop_length = 512
    frame_length = 1024

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    try:
        f0, voiced_flags, voiced_probs = librosa.pyin(y, fmin=50, fmax=600, sr=sr, hop_length=hop_length)
        f0_voiced = f0[~np.isnan(f0)]
        pitch_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        pitch_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        voiced_ratio = float(np.sum(~np.isnan(f0)) / len(f0)) if len(f0)>0 else 0.0
    except Exception:
        pitch_mean = 0.0; pitch_std = 0.0; voiced_ratio = 0.0; voiced_flags = None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1).tolist()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    tempo_val = float(tempo if not isinstance(tempo, (list, np.ndarray)) else (tempo[0] if len(tempo)>0 else 0.0))

    energy_thresh = max(1e-6, np.percentile(rms, 10))
    pause_ratio = float(np.sum(rms < energy_thresh)) / len(rms) if len(rms)>0 else 0.0

    elong_s = 0.0
    if voiced_flags is not None:
        max_run = 0; run = 0
        for v in voiced_flags:
            if v:
                run += 1
            else:
                if run > max_run: max_run = run
                run = 0
        elong_s = float((max_run * hop_length) / sr)

    features = {
        "duration_s": float(librosa.get_duration(y=y, sr=sr)),
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "pitch_mean_hz": pitch_mean,
        "pitch_std_hz": pitch_std,
        "voiced_ratio": voiced_ratio,
        "mfcc_means": mfcc_means,
        "tempo": tempo_val,
        "pause_ratio": pause_ratio,
        "elongation_s": elong_s
    }
    return features

def analyze_text_sentiment(text):
    """Return sentiment score (-1..1) and magnitude via Cloud Natural Language (fallback 0,0)."""
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        resp = client.analyze_sentiment(request={"document": document})
        return float(resp.document_sentiment.score), float(resp.document_sentiment.magnitude)
    except Exception as e:
        return 0.0, 0.0

def fuse_emotion(features, transcript, words, asr_conf, text_sent_score):
    """Conservative heuristic fusion: arousal(audio) + valence(text) -> final label + confidence."""
    word_count = len(transcript.split()) if transcript else 0
    dur = features.get("duration_s", 1.0) or 1.0
    speech_rate = word_count / dur

    arousal = 0.0
    arousal += min(1.0, features.get("rms_mean", 0.0) * 100.0)
    arousal += min(1.0, features.get("tempo", 0.0) / 200.0)
    arousal += min(1.0, speech_rate / 4.0)
    arousal = max(0.0, min(1.0, arousal / 3.0))

    valence = float(text_sent_score) if text_sent_score is not None else 0.0
    if transcript and valence == 0.0:
        t = transcript.lower()
        for w in ["happy","good","great","awesome","love","fun","excited","lol"]:
            if w in t:
                valence += 0.6
        for w in ["sad","tired","missed","angry","stressed","hate","lonely","upset"]:
            if w in t:
                valence -= 0.6
        valence = max(-1.0, min(1.0, valence))

    elong = features.get("elongation_s", 0.0)
    pause = features.get("pause_ratio", 0.0)

    label = "neutral"; conf = 0.5
    if arousal > 0.55 and (valence > 0.15 or elong > 0.3):
        label = "energetic"; conf = min(0.98, 0.55 + arousal*0.35 + max(0, valence)*0.2 + asr_conf*0.1)
    elif arousal > 0.65 and valence < -0.1 and features.get("tempo",0) > 140:
        label = "hurried/angry"; conf = min(0.97, 0.5 + arousal*0.45 + (-valence)*0.2 + asr_conf*0.05)
    elif arousal < 0.35 and valence < -0.1 and pause > 0.18:
        label = "sad"; conf = min(0.95, 0.45 + (-valence)*0.4 + (0.35 - arousal))
    elif arousal >= 0.35 and valence > 0.25:
        label = "happy"; conf = min(0.93, 0.4 + valence*0.4 + arousal*0.2)
    elif arousal < 0.35 and valence > 0.2:
        label = "calm"; conf = min(0.9, 0.45 + valence*0.3)
    else:
        label = "neutral"; conf = max(0.4, 1 - abs(0.45 - arousal))

    meta = {"arousal": arousal, "valence": valence, "speech_rate_wps": speech_rate,
            "elongation_s": elong, "pause_ratio": pause, "asr_confidence": asr_conf}
    return label, float(conf), meta

def main():
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        print("ERROR: Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON (see PowerShell setx command).")
        return

    wav_bytes = record_audio_to_wav_bytes()
    features = extract_audio_features_from_wavfile(LOCAL_FILE)
    print("\nAudio features (debug):")
    for k,v in features.items():
        if k=="mfcc_means":
            print(f"{k}: [{', '.join(f'{x:.2f}' for x in v[:5])} ...]")
        else:
            print(f"{k}: {v}")

    try:
        transcript, words, asr_conf = transcribe_with_google(wav_bytes)
        print("\nTranscript:", transcript)
        print("ASR confidence:", asr_conf)
    except Exception as e:
        print("ASR error:", e)
        transcript, words, asr_conf = "", [], 0.0

    text_sent_score, text_sent_mag = 0.0, 0.0
    if transcript:
        try:
            text_sent_score, text_sent_mag = analyze_text_sentiment(transcript)
            print("Text sentiment score:", text_sent_score, "magnitude:", text_sent_mag)
        except Exception as e:
            print("NL error:", e)

    label, conf, meta = fuse_emotion(features, transcript, words, asr_conf, text_sent_score)
    print("\n=== FINAL EMOTION ===")
    print(f"Label: {label}   Confidence: {conf:.2f}")
    print("Meta:", meta)
    print("=====================\n")

if __name__ == "__main__":
    main()
