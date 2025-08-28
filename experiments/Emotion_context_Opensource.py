# Experimenting with the OpenSource models locally
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import whisper
from transformers import pipeline

# Load Whisper and Emotion model
asr_model = whisper.load_model("base")
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Step 1: Record Voice
def record_voice(filename="test.wav", duration=5, fs=44100):
    print("🎤 Recording... Speak now")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("✅ Recording saved as", filename)

# Step 2: Acoustic Analysis
def analyze_acoustics(filename="test.wav"):
    y, sr = librosa.load(filename)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
    energy = float(np.mean(librosa.feature.rms(y=y)))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo) if not isinstance(tempo, (list, np.ndarray)) else float(tempo[0]) if len(tempo) > 0 else 0.0
    return mean_pitch, energy, tempo_val

# Step 3: Speech-to-Text
def transcribe(filename="test.wav"):
    result = asr_model.transcribe(filename)
    return result["text"]

# Step 4: Emotion from Text
def analyze_text(text):
    result = emotion_model(text)
    emotions = {d["label"]: d["score"] for d in result[0]}
    top_emotion = max(emotions, key=emotions.get)
    return top_emotion, emotions

# Step 5: Fusion Logic
def combined_analysis(filename="test.wav"):
    # Acoustic
    pitch, energy, tempo = analyze_acoustics(filename)
    # Transcript
    text = transcribe(filename)
    text_emotion, emo_scores = analyze_text(text)

    print(f"\n--- Acoustic Features ---")
    print(f"Pitch: {pitch:.2f} Hz, Energy: {energy:.4f}, Tempo: {tempo:.2f} BPM")

    print(f"\n--- Transcript ---\n{text}")
    print(f"\n--- Text Emotion --- {text_emotion} ({emo_scores})")

    # Fusion rule (very simple demo)
    if text_emotion == "sadness" and energy < 0.01:
        final = "😔 Sad / Low Energy"
    elif text_emotion == "joy" and energy > 0.02:
        final = "😀 Happy & Excited"
    elif text_emotion == "anger" and tempo > 150:
        final = "😡 Angry / Frustrated"
    else:
        final = f"🙂 Neutral leaning {text_emotion}"

    print(f"\n✅ Final Emotion: {final}")

if __name__ == "__main__":
    record_voice(duration=6)
    combined_analysis()
