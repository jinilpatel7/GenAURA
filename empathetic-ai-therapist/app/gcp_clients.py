"""
gcp_clients.py - Google Cloud + Vertex AI Helper Utilities

This module provides utility functions to connect with Google Cloud services
such as:
1. Firestore (database for user/session data)
2. Speech-to-Text (converts audio → text, adaptive for short/long audio)
3. Text-to-Speech (converts text → natural audio using premium voices)
4. Vertex AI Generative Models (text generation, if available in environment)

Main Features:
- Automatically loads environment variables from a `.env` file if available.
- Dynamically chooses short or long-running Speech-to-Text API based on audio duration.
- Uses a curated voice catalog to provide realistic, emotional voices for TTS.
- Provides safe initialization of Vertex AI models with error handling.
- Includes helper functions for cleaning and parsing model outputs.
- Exposes `get_firestore_client()` for Firestore database operations.

This module is **import-safe**:
- If some services (like Vertex AI) are unavailable, functions will fail gracefully
  instead of crashing the entire app.
"""

import os
import io
import wave
import logging
import re
import json
import asyncio # <--- IMPORT ASYNCIO FOR RETRY DELAY
from typing import Any, Dict, Optional, Tuple

# dotenv is used to load environment variables from a .env file
from dotenv import load_dotenv, find_dotenv

# Google Cloud SDK imports
from google.cloud import speech
from google.cloud import firestore
from google.cloud import texttospeech

# Optional Vertex AI imports
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    _VERTEX_AVAILABLE = True
except Exception:
    # If Vertex is not installed or unavailable, mark as not available
    vertexai = None
    GenerativeModel = None
    Part = None
    _VERTEX_AVAILABLE = False

# Module logger (for debug/info/warning/error logs)
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

# --- Load environment variables on import ---
try:
    _env_path = find_dotenv()
    if _env_path:
        load_dotenv(_env_path, override=False)
        _logger.debug("Loaded .env from %s", _env_path)
    else:
        load_dotenv(override=False)
        _logger.debug("No .env found with find_dotenv(); attempted default load.")
except Exception as e:
    _logger.warning("Error loading .env: %s", e)

# --- Environment Configurations (defaults provided) ---
GCP_PROJECT: str = os.environ.get("GCP_PROJECT", "mind-sail-471005")
GCP_LOCATION: str = os.environ.get("GCP_LOCATION", "us-central1")
VERTEX_MODEL_NAME: Optional[str] = os.environ.get("VERTEX_MODEL_NAME")
SPEECH_MODEL: Optional[str] = os.environ.get("SPEECH_MODEL")  # Optional Speech-to-Text model

_logger.debug(
    "GCP_PROJECT=%s, GCP_LOCATION=%s, VERTEX_MODEL_NAME_set=%s, SPEECH_MODEL=%s",
    GCP_PROJECT,
    GCP_LOCATION,
    bool(VERTEX_MODEL_NAME),
    SPEECH_MODEL,
)

# Flag to track Vertex initialization
_vertex_initialized = False

# --- Voice Catalog for Text-to-Speech ---
# Premium voices (Studio, WaveNet) are prioritized for natural and emotional delivery
VOICE_CATALOG = {
    "en-US": {
        "FEMALE": "en-US-Studio-O",  # Calm, professional female
        "MALE": "en-US-Studio-M",    # Calm, professional male
    },
    "hi-IN": {
        "FEMALE": "hi-IN-Wavenet-D", # Soothing female Hindi voice
        "MALE": "hi-IN-Wavenet-B",   # Calm male Hindi voice
    }
}


def init_vertex() -> None:
    """
    Initialize Vertex AI client for text generation.
    - Safe to call multiple times.
    - Does nothing if Vertex AI is not installed or already initialized.
    """
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
    Extract the duration (in seconds) from WAV file bytes.
    Returns None if it cannot be read (invalid/malformed WAV).
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
    Convert audio bytes (WAV) into text using Google Cloud Speech-to-Text.

    Features:
    - For audio < 55s: uses fast synchronous API (recognize()).
    - For audio >= 55s: uses long-running API (long_running_recognize()).
      Uses a dynamic timeout = audio length + 60s to avoid API timeouts.
    - Supports custom model selection via SPEECH_MODEL env var.

    Args:
        wav_bytes: Raw WAV file content (16-bit PCM recommended).
        sample_rate: Audio sample rate in Hz.
        language_code: Language for recognition (e.g., "en-US", "hi-IN").

    Returns:
        (transcript, confidence)
    """
    SYNCHRONOUS_API_LIMIT_SECONDS = 55  # Safety margin below official 60s limit
    
    try:
        _logger.debug("Starting Speech-to-Text (lang=%s, sample_rate=%d)", language_code, sample_rate)
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=wav_bytes)

        # Config settings (with optional custom model)
        config_kwargs = dict(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            enable_automatic_punctuation=True,
        )
        model_from_env = SPEECH_MODEL or os.environ.get("SPEECH_MODEL")
        if model_from_env:
            config_kwargs["model"] = model_from_env
            _logger.debug("Using custom SPEECH_MODEL: %s", model_from_env)

        config = speech.RecognitionConfig(**config_kwargs)

        # Check duration and choose API
        duration = _wav_duration_seconds_from_bytes(wav_bytes)
        if duration is not None:
            _logger.debug("WAV duration: %.3fs", duration)

        if duration is not None and duration < SYNCHRONOUS_API_LIMIT_SECONDS:
            _logger.debug("Using synchronous API for short audio.")
            response = client.recognize(config=config, audio=audio)
        else:
            _logger.debug("Using long-running API for long audio.")
            dynamic_timeout = (int(duration) + 60) if duration else 180
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=dynamic_timeout)

        # Parse results
        if not response.results:
            _logger.info("No speech detected in audio")
            return "", 0.0

        segments, confidences = [], []
        for res in response.results:
            if not res.alternatives:
                continue
            alt = res.alternatives[0]
            text = (alt.transcript or "").strip()
            if text:
                segments.append(text)
            try:
                confidences.append(float(alt.confidence))
            except Exception:
                pass

        transcript = " ".join(segments).strip()
        confidence = max(confidences) if confidences else 0.0

        _logger.info("STT result (confidence=%.2f): %s", confidence, transcript)
        return transcript, confidence

    except Exception as e:
        _logger.exception("Speech-to-Text failed: %s", e)
        return "", 0.0


def text_to_speech_bytes(
    text: str,
    lang_code: str = "en-US",
    gender: str = "FEMALE",
    audio_encoding: str = "MP3"
) -> Optional[bytes]:
    """
    Convert text into natural speech using Google Cloud Text-to-Speech.

    Features:
    - Uses premium voices (Studio/WaveNet) from VOICE_CATALOG when available.
    - Falls back to generic voices if not found.
    - Default voice is female, slightly slowed speaking rate for a calming tone.

    Args:
        text: The text to convert into speech.
        lang_code: Language code (e.g., "en-US", "hi-IN").
        gender: Preferred voice gender ("FEMALE", "MALE", "NEUTRAL").
        audio_encoding: Output format (MP3 by default).

    Returns:
        Audio content as bytes (or None on failure).
    """
    try:
        if not text or not text.strip():
            _logger.warning("Empty text provided for TTS")
            return None

        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Map gender input to API enums
        gender_up = (gender or "FEMALE").upper()
        if gender_up == "FEMALE":
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE
        elif gender_up == "MALE":
            ssml_gender = texttospeech.SsmlVoiceGender.MALE
        else:
            ssml_gender = texttospeech.SsmlVoiceGender.NEUTRAL

        # Try premium catalog voice first
        voice_name = VOICE_CATALOG.get(lang_code, {}).get(gender_up)
        if voice_name:
            voice = texttospeech.VoiceSelectionParams(language_code=lang_code, name=voice_name)
        else:
            # Fallback generic voice
            _logger.warning("No catalog voice for lang=%s, gender=%s", lang_code, gender_up)
            voice = texttospeech.VoiceSelectionParams(language_code=lang_code, ssml_gender=ssml_gender)

        # Audio settings
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95,  # slower pace
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
        _logger.exception("Text-to-Speech failed: %s", e)
        return None


def _extract_json_from_raw(raw: Optional[str]) -> Optional[str]:
    """
    Extract the first JSON-looking substring from raw model output.
    Useful when LLM responses contain extra text around JSON.
    """
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    return m.group(0) if m else None


def _attempt_parse(jtext: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Try to parse JSON safely, repairing common formatting issues:
    - Replace single quotes with double quotes.
    - Remove trailing commas.
    Returns dict or None if parsing fails.
    """
    if not jtext:
        return None
    try:
        return json.loads(jtext)
    except Exception:
        fixed = jtext.replace("'", '"')
        fixed = re.sub(r",\s*([}```])", r"\1", fixed)
        try:
            return json.loads(fixed)
        except Exception:
            return None


async def vertex_generate(
    prompt_text: str,
    model_name: Optional[str] = None,
    max_output_chars: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Optional[str]:
    """
    Generate text using Vertex AI's generative models with resilience.
    - Handles rate limiting with exponential backoff.
    - Explicitly checks for and logs safety filter blocks.

    Args:
        prompt_text: The input prompt to the model.
        model_name: Override the default model if provided.
        max_output_chars: Maximum number of characters in response.
        temperature: Controls creativity (higher = more random).
        top_p: Controls diversity of word selection.

    Returns:
        Generated text, or None if generation fails.
    """
    if not _VERTEX_AVAILABLE:
        _logger.debug("Vertex AI not available")
        return None

    model_to_use = model_name or VERTEX_MODEL_NAME
    if not model_to_use:
        _logger.warning("No Vertex model specified")
        return None

    init_vertex()
    if not _vertex_initialized:
        _logger.warning("Vertex AI not initialized, cannot generate")
        return None

    model = GenerativeModel(model_to_use)
    generation_config = {
        "max_output_tokens": max_output_chars,
        "temperature": temperature,
        "top_p": top_p
    }

    max_retries = 3
    delay = 1.0  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            response = await model.generate_content_async(prompt_text, generation_config=generation_config)

            # --- MAJOR FIX: Explicitly check for safety blocks ---
            if not response.candidates:
                _logger.warning(
                    "Vertex AI response was blocked. Prompt Feedback: %s",
                    response.prompt_feedback
                )
                return None  # Return None immediately if blocked, no retry needed

            # The official way to get text is from the first candidate.
            return response.candidates[0].content.parts[0].text

        except Exception as e:
            _logger.warning("Attempt %d: Vertex AI generation failed with exception: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                _logger.error("All %d retries for Vertex AI generation failed.", max_retries)
                return None
    return None


def get_firestore_client() -> Optional[firestore.Client]:
    """
    Initialize and return a Firestore client.

    Returns:
        Firestore client instance, or None on failure.
    """
    try:
        _logger.debug("Initializing Firestore client for project: %s", GCP_PROJECT)
        return firestore.Client(project=GCP_PROJECT)
    except Exception as e:
        _logger.exception("Firestore client initialization failed: %s", e)
        return None