"""
main.py - FastAPI application entrypoint for Empathetic AI Therapist

Purpose:
- Exposes HTTP endpoints for starting/ending sessions, processing user audio,
  marking tasks done, and developer/debug endpoints.
- Orchestrates the core pipeline:
    upload audio -> audio processing (STT, emotion, safety) -> Brain1 decision -> Brain2 response -> TTS
- Persists conversation turns to Firestore and uses Redis for short-lived session state.
- Uses Vertex AI (when available) for model-driven pieces (decisioning, response generation, summaries).

Design/behavioral notes:
- Redis is used as the authoritative session store (TTL-based expiration).
- Firestore stores persistent conversation logs and session summaries.
- Safety-critical flows short-circuit Brain1 to a fixed safety protocol.
- This file focuses on orchestration and wiring; core algorithms live in other modules.
- The file intentionally avoids changing business logic — only comments and docstrings are provided
  to make the flow easier for new contributors to understand.
"""

import os
import logging
import base64
import uuid
import json
from datetime import datetime
from typing import Any, Dict
import asyncio

# --- NEW ---: Add redis import
import redis

from dotenv import load_dotenv
import pytz

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, Response

# Import app components (these modules contain the heavy lifting)
from .audio_processing import process_audio_file_bytes
from .gcp_clients import get_firestore_client, text_to_speech_bytes, init_vertex, VERTEX_MODEL_NAME
from .brain1_policy import call_brain1
from .brain2_response import call_brain2
from .utils import detect_safety
from . import auth
from . import wellness

# Configure logging (configurable via LOG_LEVEL env var)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

# FastAPI app and routers
app = FastAPI(title="Empathetic AI Therapist")
app.include_router(auth.router)
app.include_router(wellness.router, prefix="/wellness")

# Static files: if a frontend exists under app/static, mount it at /static
import pathlib
BASE_DIR = pathlib.Path(__file__).resolve().parent
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    try:
        from fastapi.staticfiles import StaticFiles
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    except Exception as e:
        _logger.warning("Failed to mount static directory: %s", e)
else:
    _logger.warning("Static directory not found at %s", static_dir)

# App-level constants and configuration
FIRESTORE_SESSIONS_SUBCOLLECTION = "therapy_sessions"
FIRESTORE_USERS_COLLECTION = "users"
SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "30"))

# --- NEW ---: Redis client for session management
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
try:
    # Create a Redis client and verify the connection by pinging
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    _logger.info("Successfully connected to Redis for session storage.")
except redis.exceptions.ConnectionError as e:
    # If Redis is not available, log the error and set client to None.
    _logger.error("Could not connect to Redis at %s. Session management will fail. Error: %s", REDIS_URL, e)
    redis_client = None


# Previously an in-memory dict was used for sessions. Redis replaces it now.
# SESSIONS: Dict[str, Dict[str, Any]] = {}

# Initial / closing messages localized by language code
INITIAL_GREETINGS = {
    "en-US": "Hi, I'm here to listen. How are you feeling today? Take your time to share as much as you're comfortable with.",
    "hi-IN": "नमस्ते, मैं आपकी बात सुनने के लिए यहाँ हूँ। आज आप कैसा महसूस कर रहे हैं? आप जितना चाहें, उतना साझा करने के लिए अपना समय लें।"
}
CLOSING_MESSAGES = {
    "en-US": "Thank you for sharing your time with me today. Remember to be kind to yourself as you continue through your day.",
    "hi-IN": "आज मेरे साथ अपना समय साझा करने के लिए धन्यवाद। जैसे-जैसे आप अपने दिन में आगे बढ़ें, खुद पर दया करना याद रखें।"
}

# -------------------------
# Redis-backed session helpers
# -------------------------
def _save_session(session_id: str, session_data: Dict[str, Any]):
    """
    Save or update a session object in Redis.

    Behavior:
    - Adds/updates `last_active` timestamp in UTC.
    - Stores the session JSON in Redis with TTL equal to SESSION_TIMEOUT_MINUTES.
    - If Redis is unavailable, logs an error (no exception thrown to avoid crashing endpoints).
    """
    if not redis_client:
        _logger.error("Cannot save session %s: Redis client not available.", session_id)
        return
    try:
        # Update last_active so TTL is refreshed and we have a record of recency
        session_data["last_active"] = datetime.utcnow().isoformat() + "Z"
        redis_client.set(
            session_id,
            json.dumps(session_data),
            ex=SESSION_TIMEOUT_MINUTES * 60  # Set expiration to session timeout
        )
    except Exception as e:
        _logger.exception("Failed to save session %s to Redis: %s", session_id, e)


def get_session(session_id: str) -> Dict[str, Any]:
    """
    Retrieve a session object from Redis.

    Raises:
      - HTTPException(503) if Redis is unavailable.
      - HTTPException(404) if session not found or expired.
      - HTTPException(500) on other errors.

    Note:
    - This function does not mutate the stored session here; callers should update the session
      and call _save_session to persist changes and refresh TTL.
    """
    if not redis_client:
        raise HTTPException(status_code=503, detail="Session service unavailable")

    try:
        session_json = redis_client.get(session_id)
        if not session_json:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        session = json.loads(session_json)
        # We intentionally do not update last_active here; callers should call _save_session after changes.
        return session

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Failed to retrieve session %s from Redis: %s", session_id, e)
        raise HTTPException(status_code=500, detail="Could not retrieve session")


# -------------------------
# Firestore persistence helper
# -------------------------
def _save_turn_to_firestore(session: Dict[str, Any], turn_data: Dict[str, Any]):
    """
    Append a conversation turn to a Firestore document for the user session.

    Behavior:
    - If the session does not yet have a firestore_doc_id, create one using IST-local timestamp.
    - Uses firestore.ArrayUnion to append turns so the document can accumulate conversation history.
    - Safe-fails with logging if Firestore client is unavailable.
    """
    user_id = session.get("user_id")
    if not user_id:
        _logger.warning("Cannot save turn to Firestore: missing user_id in session.")
        return

    try:
        client = get_firestore_client()
        if not client:
            _logger.error("Cannot save turn: Firestore client not available.")
            return

        from google.cloud import firestore

        firestore_doc_id = session.get("firestore_doc_id")

        if not firestore_doc_id:
            # Create a human-readable doc id using Asia/Kolkata local time
            utc_now = datetime.utcnow()
            ist_tz = pytz.timezone('Asia/Kolkata')
            ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist_tz)
            firestore_doc_id = ist_now.strftime('%Y-%m-%d_%H-%M-%S_IST')
            session["firestore_doc_id"] = firestore_doc_id
            _logger.info("Creating new Firestore session document for user '%s': %s", user_id, firestore_doc_id)

        doc_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(user_id).collection(FIRESTORE_SESSIONS_SUBCOLLECTION).document(firestore_doc_id)

        # Use ArrayUnion so we append to an array field named 'conversation'
        doc_ref.set(
            {"conversation": firestore.ArrayUnion([turn_data])},
            merge=True
        )
        _logger.debug("Saved turn to Firestore doc: %s", firestore_doc_id)

    except Exception as e:
        _logger.error("Failed to save turn to Firestore for user %s: %s", user_id, e)


# -------------------------
# Small helper utilities
# -------------------------
def _determine_ai_gender(user_gender: str) -> str:
    """
    Very small helper to pick a TTS gender label.
    - The code intentionally maps "male" -> FEMALE voice label here (legacy behavior).
    - If user_gender is not 'male', returns "MALE". Keep this behavior unchanged (project-specific).
    """
    return "FEMALE" if (user_gender or "").lower() == "male" else "MALE"


# -------------------------
# Startup event
# -------------------------
@app.on_event("startup")
async def startup_event():
    """
    App startup hook:
    - Logs startup and attempts to initialize Vertex AI (best-effort).
    """
    _logger.info("Empathetic AI Therapist starting up")
    try:
        init_vertex()
    except Exception as e:
        _logger.warning("Vertex init failed or unavailable: %s", e)


# -------------------------
# Basic endpoints (index, favicon)
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Serve a static index.html from the app/static directory if present,
    otherwise return a simple placeholder HTML response.
    """
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return HTMLResponse("<h1>Empathetic AI Therapist</h1><p>Application running but no frontend found.</p>")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return 204 for favicon requests if none is provided."""
    return Response(status_code=204)


# -------------------------
# Session lifecycle endpoints
# -------------------------
@app.post("/start_session")
async def start_session(
    user_id: str = Form(...),
    language: str = Form("en-US"),
    user_gender: str = Form("female")
):
    """
    Create a new session and store it in Redis.

    Returns:
      - session_id (string): client should keep this and send it with subsequent requests.
      - initial_reply: structured initial reply for UI rendering.
      - tts_b64: optional base64 TTS audio for immediate playback.

    Behavior details:
    - session is saved to Redis (with TTL).
    - initial greeting is also persisted to Firestore as the first 'ai' turn.
    - The session's `history` is initialized with the greeting.
    """
    session_id = f"session_{int(datetime.utcnow().timestamp() * 1000)}_{str(uuid.uuid4())[:6]}"
    now_iso = datetime.utcnow().isoformat() + "Z"
    session = {
        "session_id": session_id, "user_id": user_id, "language": language,
        "user_gender": user_gender, "created_at": now_iso,
        "last_active": now_iso, "history": [],
        "tasks": [], "technique_history": []
    }

    # Save session to Redis
    _save_session(session_id, session)

    initial_text = INITIAL_GREETINGS.get(language, INITIAL_GREETINGS["en-US"])

    # Persist the initial AI turn to Firestore for records
    _save_turn_to_firestore(session, {"who": "ai", "text": f"--- NEW SESSION STARTED --- {initial_text}", "time": datetime.utcnow().isoformat() + "Z"})

    # Choose TTS gender and generate audio bytes (may be None if TTS fails)
    ai_gender = _determine_ai_gender(user_gender)
    tts_bytes = text_to_speech_bytes(initial_text, lang_code=language, gender=ai_gender)
    tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None

    # Structured representation for the UI
    initial_reply_structured = {
        "response_parts": [{"type": "plain", "text": initial_text}],
        "psychology_behind_it": "A warm opening to create a safe and welcoming space for sharing."
    }

    # Add the AI greeting to the session history and save session
    session["history"].append({"who": "ai", "text": initial_text})
    _save_session(session_id, session)

    _logger.info("Started new session %s for user %s", session_id, user_id)
    return {"session_id": session_id, "initial_reply": initial_reply_structured, "tts_b64": tts_b64}


# -------------------------
# Safe wrapper helpers for Brain calls
# -------------------------
def _safe_call_brain1(audio_output, session_history, technique_history=None):
    """
    Wrap call_brain1 to ensure the endpoint remains robust if Brain1 throws.
    Returns a conservative fallback decision on error.
    """
    try:
        return call_brain1(audio_output, session_history, technique_history=technique_history or [])
    except Exception as e:
        _logger.exception("call_brain1 failed: %s", e)
        return {"decision": "empathy+follow_up", "why_this_decision": "Fallback due to internal error."}


def _safe_call_brain2(decision, session_history, language_code):
    """
    Wrap call_brain2 to ensure the endpoint remains robust if Brain2 throws.
    Returns a simple fallback textual reply in case of errors.
    """
    try:
        reply = call_brain2(decision, session_history, language_code=language_code)
        if not isinstance(reply, dict):
            raise ValueError("call_brain2 returned non-dict")
        return reply
    except Exception as e:
        _logger.exception("call_brain2 failed: %s", e)
        fallback_text = "Sorry, I'm having a little trouble. Can you say that again?"
        return {
            "response_text_concatenated": fallback_text,
            "response_parts": [{"type": "plain", "text": fallback_text}],
            "psychology_behind_it": "Fallback due to system error."
        }


# -------------------------
# Core audio processing endpoint
# -------------------------
@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...), session_id: str = Form(...)):
    """
    Accept an uploaded audio file and run the full audio -> response pipeline.

    Steps:
    1. Load session from Redis.
    2. Call process_audio_file_bytes to obtain transcript, emotions, safety flags, etc.
    3. Append user's transcript to session history and Firestore.
    4. Call Brain1 (decision engine) and Brain2 (response generator).
    5. Persist AI reply to session history and Firestore.
    6. Generate TTS for AI reply and return combined payload to client.

    Returns:
      JSON containing audio_output, decision, reply, and base64 TTS audio.
    """
    session = get_session(session_id)
    language = session.get("language", "en-US")
    user_gender = session.get("user_gender", "female")

    _logger.info("Processing audio for session %s", session_id)
    audio_data = await file.read()

    # Run the local audio processing pipeline (STT, emotion classification, safety detection)
    audio_output = process_audio_file_bytes(audio_data, language_code=language)

    # Record the user's transcript into history
    user_text = audio_output.get("transcript", "(silence)")
    session["history"].append({"who": "user", "text": user_text})

    # Persist user turn to Firestore
    _save_turn_to_firestore(session, {"who": "user", "text": user_text, "time": datetime.utcnow().isoformat() + "Z"})

    # Decision-making and response generation
    decision = _safe_call_brain1(audio_output, session["history"], session.get("technique_history"))
    reply = _safe_call_brain2(decision, session["history"], language_code=language)

    # Extract text for TTS and store AI turn
    ai_text = reply.get("response_text_concatenated", "")
    session["history"].append({"who": "ai", "text": ai_text})
    _save_turn_to_firestore(session, {"who": "ai", "text": ai_text, "time": datetime.utcnow().isoformat() + "Z"})

    # Generate TTS bytes for the AI text (may be None if TTS fails)
    ai_gender = _determine_ai_gender(user_gender)
    tts_bytes = text_to_speech_bytes(ai_text, lang_code=language, gender=ai_gender) if ai_text else None
    tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None

    # Save updated session state back to Redis (refresh TTL)
    _save_session(session_id, session)

    return {"audio_output": audio_output, "decision": decision, "reply": reply, "tts_b64": tts_b64}


# -------------------------
# Task completion endpoint
# -------------------------
@app.post("/task_done")
async def task_done(session_id: str = Form(...), task_id: str = Form(...)):
    """
    Called when the client marks a task as completed.

    Behavior:
    - Writes a system-level 'TASK_COMPLETED' entry into session history and Firestore.
    - Calls Brain1 with a minimal input to decide next step after task completion.
    - Calls Brain2 to generate the follow-up reply and returns TTS audio if available.
    """
    session = get_session(session_id)
    language = session.get("language", "en-US")
    user_gender = session.get("user_gender", "female")

    system_text = f"TASK_COMPLETED:{task_id}"
    session["history"].append({"who": "system", "text": system_text})
    _save_turn_to_firestore(session, {"who": "system", "text": system_text, "time": datetime.utcnow().isoformat() + "Z"})

    # Minimal input signals to Brain1 that the task was completed
    task_completion_input = {"transcript": "(user just completed the assigned task)"}
    decision = _safe_call_brain1(task_completion_input, session["history"], session.get("technique_history"))
    reply = _safe_call_brain2(decision, session["history"], language_code=language)

    ai_text = reply.get("response_text_concatenated", "")
    session["history"].append({"who": "ai", "text": ai_text})
    _save_turn_to_firestore(session, {"who": "ai", "text": ai_text, "time": datetime.utcnow().isoformat() + "Z"})

    ai_gender = _determine_ai_gender(user_gender)
    tts_bytes = text_to_speech_bytes(ai_text, lang_code=language, gender=ai_gender) if ai_text else None
    tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None

    # Persist final session state to Redis
    _save_session(session_id, session)

    return {"status": "completed", "decision": decision, "reply": reply, "tts_b64": tts_b64}


# -------------------------
# End session endpoint
# -------------------------
@app.post("/end_session")
async def end_session(session_id: str = Form(...)):
    """
    End a session:
    - Persist a final closing message to Firestore.
    - Kick off an asynchronous task to generate and save a session summary (fire-and-forget).
    - Remove the session from Redis to free resources.
    - Return closing reply and TTS audio for the UI to play.
    """
    session = get_session(session_id)
    language = session.get("language", "en-US")
    user_gender = session.get("user_gender", "female")

    closing_text = CLOSING_MESSAGES.get(language, CLOSING_MESSAGES["en-US"])

    _save_turn_to_firestore(session, {"who": "ai", "text": f"{closing_text} --- SESSION ENDED ---", "time": datetime.utcnow().isoformat() + "Z"})

    # Fire-and-forget generation of session summary (do not block the response)
    asyncio.create_task(
        wellness.generate_session_summary(
            user_id=session.get("user_id"),
            session_history=session.get("history", []),
            session_start_iso=session.get("created_at")
        )
    )

    ai_gender = _determine_ai_gender(user_gender)
    tts_bytes = text_to_speech_bytes(closing_text, lang_code=language, gender=ai_gender)
    tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None

    closing_reply_structured = {"response_parts": [{"type": "plain", "text": closing_text}]}

    # Delete the session key from Redis so it can no longer be used
    if redis_client:
        redis_client.delete(session_id)

    _logger.info("Session ended and removed from Redis: %s", session_id)

    return {"status": "ended", "closing_reply": closing_reply_structured, "tts_b64": tts_b64}


# -------------------------
# Debugging / developer endpoints
# -------------------------
@app.post("/brain1")
async def endpoint_brain1(payload: dict):
    """
    Direct endpoint to test Brain1 decisioning in isolation.
    Accepts payload with audio_output and session_history.
    """
    try:
        audio_output = payload.get("audio_output")
        session_history = payload.get("session_history", [])
        technique_history = payload.get("technique_history", [])
        if not audio_output:
            raise HTTPException(status_code=400, detail="Missing audio_output in payload")
        decision = _safe_call_brain1(audio_output, session_history, technique_history=technique_history)
        return decision
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Brain1 direct call failed")
        raise HTTPException(status_code=500, detail="Brain1 processing failed")


@app.post("/brain2")
async def endpoint_brain2(payload: dict):
    """
    Direct endpoint to test Brain2 response generation in isolation.
    Accepts payload with decision and session_history.
    """
    try:
        decision = payload.get("decision")
        session_history = payload.get("session_history", [])
        language_code = payload.get("language_code", "en-US")
        if not decision:
            raise HTTPException(status_code=400, detail="Missing decision in payload")
        reply = _safe_call_brain2(decision, session_history, language_code=language_code)
        return reply
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Brain2 direct call failed")
        raise HTTPException(status_code=500, detail="Brain2 processing failed")


@app.get("/session/{session_id}")
async def get_session_endpoint(session_id: str):
    """
    Return a brief summary of a session's metadata (useful for debugging and UI status).
    """
    try:
        session = get_session(session_id)
        return {
            "session_id": session["session_id"], "created_at": session["created_at"],
            "last_active": session["last_active"], "history_count": len(session["history"]),
            "task_count": len(session["tasks"]),
            "technique_history": session.get("technique_history", [])[-5:]
        }
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Session retrieval failed")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")


@app.get("/health")
async def health_check():
    """
    Simple health endpoint:
    - reports whether Redis appears available (session count), timestamp, and whether a Vertex model is configured.
    """
    sessions_active = redis_client.dbsize() if redis_client else -1
    return {
        "status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z",
        "sessions_active": sessions_active, "vertex_available": bool(VERTEX_MODEL_NAME)
    }


@app.get("/_debug_env")
async def debug_env():
    """
    Expose a small debug summary of environment variables. Only enabled when ALLOW_DEBUG_ENDPOINT=1.
    Guard this endpoint in production!
    """
    if os.environ.get("ALLOW_DEBUG_ENDPOINT") != "1":
        raise HTTPException(status_code=403, detail="Debug endpoint disabled")
    return {
        "gcp_project": os.environ.get("GCP_PROJECT"), "gcp_location": os.environ.get("GCP_LOCATION"),
        "vertex_model_env": os.environ.get("VERTEX_MODEL_NAME"), "vertex_model_resolved": VERTEX_MODEL_NAME,
        "google_credentials_set": bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    }


# -------------------------
# Run with Uvicorn when executed directly
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
