# empathetic-ai-therapist/app/gcp_clients.py
"""
GCP + Vertex helper utilities.

This module:
 - Loads environment variables from a .env file (if present) at import time
 - Exposes helper functions for Speech-to-Text (robust to short pauses), Text-to-Speech,
   Vertex generation, and Firestore client.
"""

import os
import io
import wave
import logging
import re
import json
from typing import Any, Dict, Optional, Tuple

# dotenv to read .env files (simple import + call)
from dotenv import load_dotenv, find_dotenv

# Google Cloud imports
from google.cloud import speech
from google.cloud import firestore
from google.cloud import texttospeech

# Vertex AI (optional)
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    _VERTEX_AVAILABLE = True
except Exception:
    vertexai = None
    GenerativeModel = None
    _VERTEX_AVAILABLE = False

# Module logger
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

# Load .env (if present) at import time
try:
    _env_path = find_dotenv()
    if _env_path:
        load_dotenv(_env_path, override=False)
        _logger.debug("Loaded .env from %s", _env_path)
    else:
        load_dotenv(override=False)
        _logger.debug("No explicit .env found via find_dotenv(); attempted default load.")
except Exception as e:
    _logger.warning("Error loading .env: %s", e)

# Configuration from environment variables (read after load_dotenv)
GCP_PROJECT: str = os.environ.get("GCP_PROJECT", "mind-sail-471005")
GCP_LOCATION: str = os.environ.get("GCP_LOCATION", "us-central1")
VERTEX_MODEL_NAME: Optional[str] = os.environ.get("VERTEX_MODEL_NAME")
# Optional SPEECH_MODEL env var to control Speech-to-Text model (e.g. "latest_short", "latest_long")
SPEECH_MODEL: Optional[str] = os.environ.get("SPEECH_MODEL")

_logger.debug(
    "GCP_PROJECT=%s, GCP_LOCATION=%s, VERTEX_MODEL_NAME_set=%s, SPEECH_MODEL=%s",
    GCP_PROJECT,
    GCP_LOCATION,
    bool(VERTEX_MODEL_NAME),
    SPEECH_MODEL,
)

# Internal flag for Vertex initialization
_vertex_initialized = False

# *** MODIFIED: Voice Catalog for high-quality, emotional voices ***
# We prioritize "Studio" voices for their realism and emotional range.
# WaveNet is the next best choice.
VOICE_CATALOG = {
    "en-US": {
        "FEMALE": "en-US-Studio-O",  # Calm, professional female voice
        "MALE": "en-US-Studio-M",    # Calm, professional male voice
    },
    "hi-IN": {
        "FEMALE": "hi-IN-Wavenet-D", # Soothing female Hindi voice
        "MALE": "hi-IN-Wavenet-B",   # Calm male Hindi voice
    }
}


def init_vertex() -> None:
    """Initialize Vertex AI client (safe to call multiple times)."""
    global _vertex_initialized
    if _vertex_initialized or not _VERTEX_AVAILABLE:
        if not _VERTEX_AVAILABLE:
            _logger.debug("Vertex AI python package not available.")
        return

    try:
        _logger.info("Initializing Vertex AI: project=%s, location=%s", GCP_PROJECT, GCP_LOCATION)
        vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
        _vertex_initialized = True
        _logger.info("Vertex AI initialized successfully")
    except Exception as e:
        _logger.exception("Vertex AI initialization failed: %s", e)
        _vertex_initialized = False


def _wav_duration_seconds_from_bytes(wav_bytes: bytes) -> Optional[float]:
    """
    Try reading WAV header from bytes to compute duration in seconds.
    Returns None if it cannot be read.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return None
            return float(frames) / float(rate)
    except Exception:
        return None


def transcribe_wav_bytes(
    wav_bytes: bytes,
    sample_rate: int = 16000,
    language_code: str = "en-US",
) -> Tuple[str, float]:
    """
    Convert speech to text using Google Cloud Speech-to-Text.

    This implementation is now adaptive:
    - For audio shorter than 55 seconds, it uses the fast synchronous `recognize()` API.
    - For audio longer than 55 seconds, it uses the `long_running_recognize()` API
      with a dynamic timeout calculated as (audio_duration + 60 seconds) to
      prevent timeouts on long user inputs.

    Parameters:
      - wav_bytes: raw WAV file bytes (PCM16, mono recommended)
      - sample_rate: sample rate in Hz (for config; actual WAV header will be used by API)
      - language_code: e.g. "en-US", "hi-IN"

    Returns: (transcript, confidence)
    """
    # Define the threshold for using the synchronous vs. long-running API.
    # The official limit is 60s, we use 55s to be safe.
    SYNCHRONOUS_API_LIMIT_SECONDS = 55
    
    try:
        _logger.debug("Starting speech-to-text processing (lang=%s, sample_rate=%d)", language_code, sample_rate)
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=wav_bytes)

        # Build RecognitionConfig; include model only if SPEECH_MODEL is set
        config_kwargs = dict(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language_code, # *** MODIFIED: Uses passed-in language_code
            enable_automatic_punctuation=True,
        )

        model_from_env = SPEECH_MODEL or os.environ.get("SPEECH_MODEL")
        if model_from_env:
            config_kwargs["model"] = model_from_env
            _logger.debug("Using SPEECH_MODEL from env: %s", model_from_env)
        else:
            _logger.debug("No SPEECH_MODEL set; omitting model from RecognitionConfig to use API default")

        config = speech.RecognitionConfig(**config_kwargs)

        # Compute duration to decide which API to use and to set a dynamic timeout
        duration = _wav_duration_seconds_from_bytes(wav_bytes)
        if duration is not None:
            _logger.debug("Detected WAV duration: %.3fs", duration)

        if duration is not None and duration < SYNCHRONOUS_API_LIMIT_SECONDS:
            _logger.debug("Using synchronous recognize() API for short audio.")
            response = client.recognize(config=config, audio=audio)
        else:
            _logger.debug("Using long_running_recognize() API for long audio.")
            dynamic_timeout = (int(duration) + 60) if duration is not None else 180
            _logger.debug("Calling long_running_recognize (will wait up to %ds)...", dynamic_timeout)
            
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=dynamic_timeout)

        if not response.results:
            _logger.info("No speech detected in audio (Speech-to-Text)")
            return "", 0.0

        segments = []
        confidences = []
        for res in response.results:
            if not res.alternatives:
                continue
            alt = res.alternatives[0]
            text = (alt.transcript or "").strip()
            if text:
                segments.append(text)
            try:
                c = float(alt.confidence)
                confidences.append(c)
            except Exception:
                pass

        transcript = " ".join(segments).strip()
        confidence = max(confidences) if confidences else 0.0

        _logger.info("Speech-to-Text result (confidence=%.2f): %s", confidence, transcript)
        return transcript, confidence

    except Exception as e:
        _logger.exception("Speech-to-Text processing failed: %s", e)
        return "", 0.0


def text_to_speech_bytes(
    text: str,
    lang_code: str = "en-US",
    gender: str = "FEMALE", # Defaulting to FEMALE now
    audio_encoding: str = "MP3"
) -> Optional[bytes]:
    """
    Convert text to speech using Google Cloud Text-to-Speech
    Returns audio content bytes or None on failure.
    *** MODIFIED: This function now dynamically selects a high-quality voice. ***
    """
    try:
        if not text or not text.strip():
            _logger.warning("Empty text provided for TTS")
            return None

        _logger.debug("Starting text-to-speech processing for text: %s", (text[:50] + "...") if len(text) > 50 else text)
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Map gender parameter to SSML gender enum
        gender_up = (gender or "FEMALE").upper()
        if gender_up == "FEMALE":
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
        elif gender_up == "MALE":
            ssml_gender = texttospeech.SsmlVoiceGender.MALE
        else:
            ssml_gender = texttospeech.SsmlVoiceGender.NEUTRAL

        # *** MODIFIED: Select a premium, soothing voice from the catalog ***
        voice_name = VOICE_CATALOG.get(lang_code, {}).get(gender_up)
        if voice_name:
            _logger.debug("Selected voice from catalog: %s for lang=%s, gender=%s", voice_name, lang_code, gender_up)
            voice = texttospeech.VoiceSelectionParams(
                language_code=lang_code,
                name=voice_name
            )
        else:
            # Fallback if language/gender combo isn't in our catalog
            _logger.warning("No specific voice found for lang=%s, gender=%s. Using generic selection.", lang_code, gender_up)
            voice = texttospeech.VoiceSelectionParams(
                language_code=lang_code,
                ssml_gender=ssml_gender
            )

        # Configure audio encoding
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95, # Slightly slower for a calmer pace
            pitch=0.0
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        _logger.info("Text-to-Speech completed successfully")
        return response.audio_content

    except Exception as e:
        _logger.exception("Text-to-Speech processing failed: %s", e)
        return None

# ... (rest of the file remains the same)
def _extract_json_from_raw(raw: Optional[str]) -> Optional[str]:
    """Find the first JSON-looking substring in raw LLM output."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return m.group(0)
    return None


def _attempt_parse(jtext: Optional[str]) -> Optional[Dict[str, Any]]:
    """Try to parse JSON with some common repairs (single -> double quotes, trailing commas)."""
    if not jtext:
        return None
    try:
        return json.loads(jtext)
    except Exception:
        # Try naive repairs
        fixed = jtext.replace("'", '"')
        # fixed regex: remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}```])", r"\1", fixed)
        try:
            return json.loads(fixed)
        except Exception:
            return None


def vertex_generate(
    prompt_text: str,
    model_name: Optional[str] = None,
    max_output_chars: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Optional[str]:
    """
    Generate text using Vertex AI's generative models.
    Returns generated text or None on failure.
    - model_name: override the environment model name if provided.
    """
    if not _VERTEX_AVAILABLE:
        _logger.debug("Vertex AI not available in environment")
        return None

    # Use passed model_name or environment value
    model_to_use = model_name or VERTEX_MODEL_NAME
    if not model_to_use:
        _logger.warning("No Vertex model specified (VERTEX_MODEL_NAME is not set).")
        return None

    try:
        init_vertex()
        if not _vertex_initialized:
            _logger.warning("Vertex AI not initialized, cannot generate")
            return None

        _logger.debug("Generating with Vertex AI model: %s", model_to_use)

        # Use the GenerativeModel class as in your original code
        model = GenerativeModel(model_to_use)

        generation_config = {
            "max_output_tokens": max_output_chars,
            "temperature": temperature,
            "top_p": top_p
        }

        response = model.generate_content(prompt_text, generation_config=generation_config)

        # response handling: try common attributes
        if hasattr(response, "text") and response.text:
            _logger.info("Vertex AI generation completed successfully")
            return response.text
        if hasattr(response, "content") and response.content:
            _logger.info("Vertex AI generation completed successfully (content)")
            return response.content
        _logger.warning("Vertex AI response missing text/content; returning stringified response")
        return str(response)

    except Exception as e:
        _logger.exception("Vertex AI generation failed: %s", e)
        return None


def get_firestore_client() -> Optional[firestore.Client]:
    """
    Get Firestore client with error handling
    Returns client or None on failure
    """
    try:
        _logger.debug("Initializing Firestore client for project: %s", GCP_PROJECT)
        return firestore.Client(project=GCP_PROJECT)
    except Exception as e:
        _logger.exception("Firestore client initialization failed: %s", e)
        return None