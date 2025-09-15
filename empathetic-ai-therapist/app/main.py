# empathetic-ai-therapist/app/main.py
"""
Main FastAPI application for Empathetic AI Therapist.
"""

import os
import logging
import base64
import uuid
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv
import pytz # NEW: Import for timezone handling

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, Response

# Import app components
from .audio_processing import process_audio_file_bytes
from .gcp_clients import get_firestore_client, text_to_speech_bytes, init_vertex, VERTEX_MODEL_NAME
from .brain1_policy import call_brain1
from .brain2_response import call_brain2
from .utils import detect_safety
from . import auth

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Empathetic AI Therapist")
app.include_router(auth.router)

# Static files
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

# App-level config
FIRESTORE_SESSIONS_SUBCOLLECTION = "therapy_sessions" # Name of the sub-collection under a user
FIRESTORE_USERS_COLLECTION = "users"
SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "30"))

# In-memory session store
SESSIONS: Dict[str, Dict[str, Any]] = {}

INITIAL_GREETINGS = {
    "en-US": "Hi, I'm here to listen. How are you feeling today? Take your time to share as much as you're comfortable with.",
    "hi-IN": "नमस्ते, मैं आपकी बात सुनने के लिए यहाँ हूँ। आज आप कैसा महसूस कर रहे हैं? आप जितना चाहें, उतना साझा करने के लिए अपना समय लें।"
}
CLOSING_MESSAGES = {
    "en-US": "Thank you for sharing your time with me today. Remember to be kind to yourself as you continue through your day.",
    "hi-IN": "आज मेरे साथ अपना समय साझा करने के लिए धन्यवाद। जैसे-जैसे आप अपने दिन में आगे बढ़ें, खुद पर दया करना याद रखें।"
}

# --- CORRECTED REAL-TIME SAVING HELPER ---
def _save_turn_to_firestore(session: Dict[str, Any], turn_data: Dict[str, Any]):
    """
    Saves a single conversational turn to Firestore in real-time under the correct user.
    Creates a new session document on the first turn, identified by IST timestamp.
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

        # Check if we have already created a Firestore document for this in-memory session
        firestore_doc_id = session.get("firestore_doc_id")
        
        if not firestore_doc_id:
            # This is the FIRST turn for this session, so create the doc ID.
            utc_now = datetime.utcnow()
            ist_tz = pytz.timezone('Asia/Kolkata')
            ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist_tz)
            
            # The document ID is the unique IST timestamp of the session's start
            firestore_doc_id = ist_now.strftime('%Y-%m-%d_%H-%M-%S_IST')
            
            # Store this ID back in the in-memory session to use for all subsequent turns
            session["firestore_doc_id"] = firestore_doc_id
            _logger.info("Creating new Firestore session document for user '%s': %s", user_id, firestore_doc_id)

        # Path: /users/{user_id}/therapy_sessions/{ist_timestamp_id}
        doc_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(user_id).collection(FIRESTORE_SESSIONS_SUBCOLLECTION).document(firestore_doc_id)

        # Atomically add the new turn to the 'conversation' array field
        doc_ref.set(
            {"conversation": firestore.ArrayUnion([turn_data])},
            merge=True
        )
        _logger.debug("Saved turn to Firestore doc: %s", firestore_doc_id)

    except Exception as e:
        _logger.error("Failed to save turn to Firestore for user %s: %s", user_id, e)


def _determine_ai_gender(user_gender: str) -> str:
    return "FEMALE" if (user_gender or "").lower() == "male" else "MALE"

def _cleanup_expired_sessions():
    now = datetime.utcnow()
    expired_sids = [
        sid for sid, session in SESSIONS.items()
        if (now - session["last_active"]).total_seconds() > SESSION_TIMEOUT_MINUTES * 60
    ]
    for sid in expired_sids:
        _logger.info("Cleaning up expired session: %s", sid)
        del SESSIONS[sid]

def get_session(session_id: str):
    _cleanup_expired_sessions()
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    SESSIONS[session_id]["last_active"] = datetime.utcnow()
    return SESSIONS[session_id]

@app.on_event("startup")
async def startup_event():
    _logger.info("Empathetic AI Therapist starting up")
    try:
        init_vertex()
    except Exception as e:
        _logger.warning("Vertex init failed or unavailable: %s", e)

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return HTMLResponse("<h1>Empathetic AI Therapist</h1><p>Application running but no frontend found.</p>")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.post("/start_session")
async def start_session(
    user_id: str = Form(...),
    language: str = Form("en-US"),
    user_gender: str = Form("female")
):
    session_id = f"session_{int(datetime.utcnow().timestamp() * 1000)}_{str(uuid.uuid4())[:6]}"
    # Create the in-memory session object. firestore_doc_id will be added on the first save.
    session = {
        "session_id": session_id, "user_id": user_id, "language": language,
        "user_gender": user_gender, "created_at": datetime.utcnow().isoformat() + "Z",
        "last_active": datetime.utcnow(), "history": [],
        "tasks": [], "technique_history": []
    }
    SESSIONS[session_id] = session

    initial_text = INITIAL_GREETINGS.get(language, INITIAL_GREETINGS["en-US"])
    
    # REAL-TIME SAVE: This will be the first turn, creating the Firestore document.
    _save_turn_to_firestore(session, {"who": "ai", "text": f"--- NEW SESSION STARTED --- {initial_text}", "time": datetime.utcnow().isoformat() + "Z"})
    
    ai_gender = _determine_ai_gender(user_gender)
    tts_bytes = text_to_speech_bytes(initial_text, lang_code=language, gender=ai_gender)
    tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None
    
    initial_reply_structured = {
        "response_parts": [{"type": "plain", "text": initial_text}],
        "psychology_behind_it": "A warm opening to create a safe and welcoming space for sharing."
    }

    session["history"].append({"who": "ai", "text": initial_text})
    _logger.info("Started new session %s for user %s", session_id, user_id)
    return {"session_id": session_id, "initial_reply": initial_reply_structured, "tts_b64": tts_b64}

def _safe_call_brain1(audio_output, session_history, technique_history=None):
    try: return call_brain1(audio_output, session_history, technique_history=technique_history or [])
    except Exception as e:
        _logger.exception("call_brain1 failed: %s", e)
        return {"decision": "empathy+follow_up", "why_this_decision": "Fallback due to internal error."}

def _safe_call_brain2(decision, session_history, language_code):
    try:
        reply = call_brain2(decision, session_history, language_code=language_code)
        if not isinstance(reply, dict): raise ValueError("call_brain2 returned non-dict")
        return reply
    except Exception as e:
        _logger.exception("call_brain2 failed: %s", e)
        fallback_text = "Sorry, I'm having a little trouble. Can you say that again?"
        return {"response_text_concatenated": fallback_text}


@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...), session_id: str = Form(...)):
    session = get_session(session_id)
    language = session.get("language", "en-US")
    user_gender = session.get("user_gender", "female")

    _logger.info("Processing audio for session %s", session_id)
    audio_data = await file.read()
    audio_output = process_audio_file_bytes(audio_data, language_code=language)
    
    user_text = audio_output.get("transcript", "(silence)")
    session["history"].append({"who": "user", "text": user_text})
    
    # REAL-TIME SAVE: Save user's turn
    _save_turn_to_firestore(session, {"who": "user", "text": user_text, "time": datetime.utcnow().isoformat() + "Z"})

    decision = _safe_call_brain1(audio_output, session["history"], session.get("technique_history"))
    reply = _safe_call_brain2(decision, session["history"], language_code=language)

    ai_text = reply.get("response_text_concatenated", "")
    session["history"].append({"who": "ai", "text": ai_text})

    # REAL-TIME SAVE: Save AI's turn
    _save_turn_to_firestore(session, {"who": "ai", "text": ai_text, "time": datetime.utcnow().isoformat() + "Z"})
    
    ai_gender = _determine_ai_gender(user_gender)
    tts_bytes = text_to_speech_bytes(ai_text, lang_code=language, gender=ai_gender) if ai_text else None
    tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None
    
    return {"audio_output": audio_output, "decision": decision, "reply": reply, "tts_b64": tts_b64}


@app.post("/task_done")
async def task_done(session_id: str = Form(...), task_id: str = Form(...)):
    session = get_session(session_id)
    language = session.get("language", "en-US")
    user_gender = session.get("user_gender", "female")

    system_text = f"TASK_COMPLETED:{task_id}"
    session["history"].append({"who": "system", "text": system_text})
    # REAL-TIME SAVE: Save the system event
    _save_turn_to_firestore(session, {"who": "system", "text": system_text, "time": datetime.utcnow().isoformat() + "Z"})

    task_completion_input = {"transcript": "(user just completed the assigned task)"}
    decision = _safe_call_brain1(task_completion_input, session["history"], session.get("technique_history"))
    reply = _safe_call_brain2(decision, session["history"], language_code=language)
    
    ai_text = reply.get("response_text_concatenated", "")
    session["history"].append({"who": "ai", "text": ai_text})
    # REAL-TIME SAVE: Save the AI's follow-up
    _save_turn_to_firestore(session, {"who": "ai", "text": ai_text, "time": datetime.utcnow().isoformat() + "Z"})

    ai_gender = _determine_ai_gender(user_gender)
    tts_bytes = text_to_speech_bytes(ai_text, lang_code=language, gender=ai_gender) if ai_text else None
    tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None

    return {"status": "completed", "reply": reply, "tts_b64": tts_b64}


@app.post("/end_session")
async def end_session(session_id: str = Form(...)):
    session = get_session(session_id)
    language = session.get("language", "en-US")
    user_gender = session.get("user_gender", "female")

    closing_text = CLOSING_MESSAGES.get(language, CLOSING_MESSAGES["en-US"])
    
    # REAL-TIME SAVE: Save the final closing message
    _save_turn_to_firestore(session, {"who": "ai", "text": f"{closing_text} --- SESSION ENDED ---", "time": datetime.utcnow().isoformat() + "Z"})
    
    ai_gender = _determine_ai_gender(user_gender)
    tts_bytes = text_to_speech_bytes(closing_text, lang_code=language, gender=ai_gender)
    tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None
    
    closing_reply_structured = {"response_parts": [{"type": "plain", "text": closing_text}]}
    
    # Clean up the in-memory session
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        _logger.info("Session ended and removed from memory: %s", session_id)
        
    return {"status": "ended", "closing_reply": closing_reply_structured, "tts_b64": tts_b64}


# --- DEBUG ENDPOINTS ---
# (These remain unchanged)

@app.post("/brain1")
async def endpoint_brain1(payload: dict):
    try:
        audio_output = payload.get("audio_output")
        session_history = payload.get("session_history", [])
        technique_history = payload.get("technique_history", [])
        if not audio_output:
            raise HTTPException(status_code=400, detail="Missing audio_output in payload")
        decision = _safe_call_brain1(audio_output, session_history, technique_history=technique_history)
        return decision
    except HTTPException: raise
    except Exception as e:
        _logger.exception("Brain1 direct call failed")
        raise HTTPException(status_code=500, detail="Brain1 processing failed")

@app.post("/brain2")
async def endpoint_brain2(payload: dict):
    try:
        decision = payload.get("decision")
        session_history = payload.get("session_history", [])
        language_code = payload.get("language_code", "en-US")
        if not decision:
            raise HTTPException(status_code=400, detail="Missing decision in payload")
        reply = _safe_call_brain2(decision, session_history, language_code=language_code)
        return reply
    except HTTPException: raise
    except Exception as e:
        _logger.exception("Brain2 direct call failed")
        raise HTTPException(status_code=500, detail="Brain2 processing failed")
        
@app.get("/session/{session_id}")
async def get_session_endpoint(session_id: str):
    try:
        session = get_session(session_id)
        return {
            "session_id": session["session_id"], "created_at": session["created_at"],
            "last_active": session["last_active"].isoformat() + "Z", "history_count": len(session["history"]),
            "task_count": len(session["tasks"]), "therapeutic_focus": session["therapeutic_focus"],
            "emotional_trajectory": session["emotional_trajectory"][-5:],
            "technique_history": session["technique_history"][-5:]
        }
    except HTTPException: raise
    except Exception as e:
        _logger.exception("Session retrieval failed")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z",
        "sessions_active": len(SESSIONS), "vertex_available": bool(VERTEX_MODEL_NAME)
    }

@app.get("/_debug_env")
async def debug_env():
    if os.environ.get("ALLOW_DEBUG_ENDPOINT") != "1":
        raise HTTPException(status_code=403, detail="Debug endpoint disabled")
    return {
        "gcp_project": os.environ.get("GCP_PROJECT"), "gcp_location": os.environ.get("GCP_LOCATION"),
        "vertex_model_env": os.environ.get("VERTEX_MODEL_NAME"), "vertex_model_resolved": VERTEX_MODEL_NAME,
        "google_credentials_set": bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)