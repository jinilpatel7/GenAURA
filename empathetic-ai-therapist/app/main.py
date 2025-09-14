# empathetic-ai-therapist/app/main.py
"""
Main FastAPI application for Empathetic AI Therapist.
"""

import os
import json
import logging
import base64
import uuid
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

# Load .env early
load_dotenv(override=False)

# Import app components
from .audio_processing import process_audio_file_bytes
from .gcp_clients import text_to_speech_bytes, get_firestore_client, init_vertex, VERTEX_MODEL_NAME
from .brain1_policy import call_brain1
from .brain2_response import call_brain2
from .utils import detect_safety

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Empathetic AI Therapist")

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
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "therapy_sessions")
SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "30"))
DEBUG_PROMPTS = bool(os.environ.get("DEBUG_PROMPTS", ""))

# In-memory session store
SESSIONS: Dict[str, Dict[str, Any]] = {}

# *** NEW: Multilingual initial/closing messages ***
INITIAL_GREETINGS = {
    "en-US": "Hi, I'm here to listen. How are you feeling today? Take your time to share as much as you're comfortable with.",
    "hi-IN": "नमस्ते, मैं आपकी बात सुनने के लिए यहाँ हूँ। आज आप कैसा महसूस कर रहे हैं? आप जितना चाहें, उतना साझा करने के लिए अपना समय लें।"
}
CLOSING_MESSAGES = {
    "en-US": "Thank you for sharing your time with me today. Remember to be kind to yourself as you continue through your day.",
    "hi-IN": "आज मेरे साथ अपना समय साझा करने के लिए धन्यवाद। जैसे-जैसे आप अपने दिन में आगे बढ़ें, खुद पर दया करना याद रखें।"
}

def _determine_ai_gender(user_gender: str) -> str:
    """Returns the opposite gender for the AI voice."""
    if (user_gender or "").lower() == "male":
        return "FEMALE"
    return "MALE" # Default to Male if user is Female or prefers not to say

def _cleanup_expired_sessions():
    # ... (function is unchanged)
    now = datetime.utcnow()
    expired = [
        sid for sid, session in SESSIONS.items()
        if (now - session["last_active"]).total_seconds() > SESSION_TIMEOUT_MINUTES * 60
    ]
    for sid in expired:
        _logger.info("Cleaning up expired session: %s", sid)
        del SESSIONS[sid]

def get_session(session_id: str):
    # ... (function is unchanged)
    _cleanup_expired_sessions()
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    SESSIONS[session_id]["last_active"] = datetime.utcnow()
    return SESSIONS[session_id]

@app.on_event("startup")
async def startup_event():
    # ... (function is unchanged)
    _logger.info("Empathetic AI Therapist starting up")
    gcred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    _logger.info("GOOGLE_APPLICATION_CREDENTIALS set: %s", bool(gcred))
    _logger.info("VERTEX_MODEL_NAME (env): %s", os.environ.get("VERTEX_MODEL_NAME"))
    _logger.info("Using VERTEX_MODEL_NAME resolved in gcp_clients: %s", VERTEX_MODEL_NAME)
    try:
        init_vertex()
    except Exception as e:
        _logger.warning("Vertex init failed or unavailable: %s", e)

@app.get("/", response_class=HTMLResponse)
async def index():
    # ... (function is unchanged)
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return HTMLResponse("<h1>Empathetic AI Therapist</h1><p>Application running but no frontend found.</p>")

@app.post("/start_session")
async def start_session(
    user_id: str = Form(None),
    language: str = Form("en-US"),
    user_gender: str = Form("female")
):
    """ *** MODIFIED: Accepts language and user_gender *** """
    try:
        session_id = f"session_{int(datetime.utcnow().timestamp() * 1000)}_{str(uuid.uuid4())[:6]}"
        SESSIONS[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "language": language,
            "user_gender": user_gender,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "last_active": datetime.utcnow(),
            "history": [],
            "tasks": [],
            "points": 0,
            "therapeutic_focus": None,
            "emotional_trajectory": [],
            "technique_history": [],
            "has_active_task": False
        }
        
        initial_text = INITIAL_GREETINGS.get(language, INITIAL_GREETINGS["en-US"])
        ai_gender = _determine_ai_gender(user_gender)
        
        tts_bytes = text_to_speech_bytes(initial_text, lang_code=language, gender=ai_gender)
        tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None
        
        initial_reply_structured = {
            "response_parts": [{"type": "plain", "text": initial_text}],
            "psychology_behind_it": "A warm opening to create a safe and welcoming space for sharing."
        }

        SESSIONS[session_id]["history"].append({
            "who": "ai",
            "text": initial_text,
            "tts_b64": tts_b64,
            "time": datetime.utcnow().isoformat() + "Z",
            "decision": None,
            "reply_structured": initial_reply_structured
        })
        _logger.info("Started new session: %s (lang=%s, user_gender=%s)", session_id, language, user_gender)
        return {
            "session_id": session_id,
            "initial_reply": initial_reply_structured,
            "tts_b64": tts_b64,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        _logger.exception("Session start failed")
        raise HTTPException(status_code=500, detail="Failed to start session")

def _safe_call_brain1(audio_output, session_history, technique_history):
    # ... (function is unchanged)
    try:
        return call_brain1(audio_output, session_history, technique_history=technique_history or [])
    except Exception as e:
        _logger.exception("call_brain1 failed: %s", e)
        return {
            "decision": "empathy+follow_up", "emotional_reflection": "I hear you. Thank you for sharing that with me.",
            "guidance": "- Tone: Gentle and curious.\n- Must: Ask a simple open-ended question to understand more.\n- Avoid: Making assumptions.",
            "why_this_decision": "Fallback due to internal error."
        }

def _safe_call_brain2(decision, session_history, language_code):
    """ *** MODIFIED: Passes language_code to Brain2 *** """
    try:
        reply = call_brain2(decision, session_history, language_code=language_code)
        if not isinstance(reply, dict):
            raise ValueError("call_brain2 returned non-dict")
        return reply
    except Exception as e:
        _logger.exception("call_brain2 failed: %s", e)
        fallback_text = "Sorry, I'm having trouble right now. Can you say that again in a moment?"
        return {
            "response_parts": [{"type": "plain", "text": fallback_text}],
            "response_text_concatenated": fallback_text, "psychology_behind_it": "This is a fallback response due to a temporary system issue.",
            "tts_voice": {"lang": "en-US"}, "display_cards": [], "task_meta": None
        }

@app.post("/process_audio")
async def process_audio(
    file: UploadFile = File(...), 
    session_id: str = Form(...)
):
    try:
        session = get_session(session_id)
        language = session.get("language", "en-US")
        user_gender = session.get("user_gender", "female")
        
        _logger.info("Processing audio for session %s (lang=%s)", session_id, language)
        audio_data = await file.read()

        # Process audio with correct language
        audio_output = process_audio_file_bytes(
            audio_data, 
            filename_hint=file.filename or "upload",
            language_code=language
        )
        audio_output["session_state"] = {"has_active_task": bool(session.get("has_active_task", False))}
        
        session["emotional_trajectory"].append({
            "timestamp": audio_output.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            "emotion": audio_output.get("final_emotion", "neutral"), "confidence": audio_output.get("confidence", 0.0)
        })
        
        session["history"].append({
            "who": "user", "text": audio_output.get("transcript", "(silence)"),
            "time": datetime.utcnow().isoformat() + "Z",
            "audio_analysis": {
                "emotion": audio_output.get("final_emotion", "neutral"), "intensity": audio_output.get("intensity", "medium-energy"),
                "confidence": audio_output.get("confidence", 0.0)
            }
        })
        
        _logger.debug("Calling Brain1 for session %s", session_id)
        decision = _safe_call_brain1(audio_output, session["history"], technique_history=session.get("technique_history", []))
        session["last_decision"] = decision

        _logger.debug("Calling Brain2 for session %s (lang=%s)", session_id, language)
        reply = _safe_call_brain2(decision, session["history"], language_code=language)

        # ... (task handling logic remains the same) ...
        task_meta = reply.get("task_meta")
        if task_meta and isinstance(task_meta, dict):
            task_id = task_meta.get("task_id") or str(uuid.uuid4())
            duration = min(180, max(10, int(task_meta.get("duration_sec", 60))))
            session["tasks"].append({
                "task_id": task_id, "type": task_meta.get("type", "exercise"), "detail": task_meta.get("detail", "a short exercise"),
                "duration_sec": duration, "assigned_at": datetime.utcnow().isoformat() + "Z", "status": "assigned"
            })
            session["has_active_task"] = True
            _logger.info("Assigned task %s to session %s", task_id, session_id)
            session["technique_history"].append({"technique": "task", "timestamp": datetime.utcnow().isoformat() + "Z"})
            session["technique_history"] = session["technique_history"][-30:]
        else:
            technique = None
            d = decision.get("decision","") if isinstance(decision, dict) else ""
            if "follow_up" in d: technique = "follow_up"
            elif "metacognitive" in d: technique = "metacognitive"
            elif "CBT" in d: technique = "cognitive_step"
            if technique:
                session["technique_history"].append({"technique": technique, "timestamp": datetime.utcnow().isoformat() + "Z"})
                session["technique_history"] = session["technique_history"][-30:]
        
        concatenated_text = reply.get("response_text_concatenated", "")
        ai_gender = _determine_ai_gender(user_gender)
        tts_bytes = text_to_speech_bytes(concatenated_text, lang_code=language, gender=ai_gender) if concatenated_text else None
        tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None
        
        session["history"].append({
            "who": "ai", "text": concatenated_text, "reply_structured": reply,
            "tts_b64": tts_b64, "time": datetime.utcnow().isoformat() + "Z", "decision": decision
        })
        
        # ... (Firestore logging and response payload remains mostly the same)
        try:
            client = get_firestore_client()
            if client:
                client.collection(FIRESTORE_COLLECTION).document(session_id).collection("events").add({
                    "timestamp": datetime.utcnow().isoformat() + "Z", "event_type": "audio_processing",
                    "audio_output": audio_output, "decision": decision, "reply": reply
                })
        except Exception as e:
            _logger.warning("Firestore logging failed: %s", e)
        
        _logger.info("Audio processing complete for session %s", session_id)
        response_payload = {
            "session_id": session_id, "timestamp": datetime.utcnow().isoformat() + "Z",
            "audio_output": audio_output, "decision": decision, "reply": reply, "tts_b64": tts_b64,
            "history": session["history"], "tasks": session["tasks"], "therapeutic_focus": session["therapeutic_focus"],
            "emotional_trajectory": session["emotional_trajectory"][-10:],
            "technique_history": session["technique_history"][-10:]
        }
        if DEBUG_PROMPTS:
            response_payload["_debug"] = {"vertex_model_used": VERTEX_MODEL_NAME}
        return response_payload

    except HTTPException: raise
    except Exception as e:
        _logger.exception("Audio processing failed")
        raise HTTPException(status_code=500, detail="Audio processing failed")

@app.post("/task_done")
async def task_done(session_id: str = Form(...), task_id: str = Form(...)):
    try:
        session = get_session(session_id)
        language = session.get("language", "en-US")
        user_gender = session.get("user_gender", "female")
        
        task = next((t for t in session["tasks"] if t["task_id"] == task_id), None)
        if not task: raise HTTPException(status_code=404, detail="Task not found")
        if task["status"] != "assigned": return {"status": "already_completed", "task": task}
        
        task["status"] = "completed"
        task["completed_at"] = datetime.utcnow().isoformat() + "Z"
        session["has_active_task"] = any(t for t in session["tasks"] if t["status"] == "assigned")
        session["history"].append({
            "who": "system", "text": f"TASK_COMPLETED:{task_id}",
            "time": datetime.utcnow().isoformat() + "Z", "task_id": task_id
        })
        
        task_completion_input = {
            "timestamp": datetime.utcnow().isoformat() + "Z", "transcript": "(user just completed the assigned task)",
            "final_emotion": "neutral", "intensity": "low-energy", "context_emotion": "Completed task", "confidence": 1.0,
            "session_state": {"has_active_task": session.get("has_active_task", False), "last_event": "task_completed"}
        }
        
        decision = _safe_call_brain1(task_completion_input, session["history"], technique_history=session.get("technique_history", []))
        session["last_decision"] = decision
        reply = _safe_call_brain2(decision, session["history"], language_code=language)
        
        concatenated_text = reply.get("response_text_concatenated", "")
        ai_gender = _determine_ai_gender(user_gender)
        tts_bytes = text_to_speech_bytes(concatenated_text, lang_code=language, gender=ai_gender) if concatenated_text else None
        tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None
        
        session["history"].append({
            "who": "ai", "text": concatenated_text, "reply_structured": reply,
            "tts_b64": tts_b64, "time": datetime.utcnow().isoformat() + "Z", "decision": decision
        })
        
        _logger.info("Task %s completed in session %s", task_id, session_id)
        return {
            "status": "completed", "decision": decision, "reply": reply, "tts_b64": tts_b64,
            "tasks": session["tasks"], "history": session["history"],
            "technique_history": session.get("technique_history", [])[-10:]
        }
    except HTTPException: raise
    except Exception as e:
        _logger.exception("Task completion handling failed")
        raise HTTPException(status_code=500, detail="Task completion processing failed")

@app.post("/end_session")
async def end_session(session_id: str = Form(...)):
    try:
        session = get_session(session_id)
        language = session.get("language", "en-US")
        user_gender = session.get("user_gender", "female")

        closing_text = CLOSING_MESSAGES.get(language, CLOSING_MESSAGES["en-US"])
        ai_gender = _determine_ai_gender(user_gender)
        
        tts_bytes = text_to_speech_bytes(closing_text, lang_code=language, gender=ai_gender)
        tts_b64 = base64.b64encode(tts_bytes).decode() if tts_bytes else None
        
        closing_reply_structured = {"response_parts": [{"type": "plain", "text": closing_text}]}
        session["history"].append({
            "who": "ai", "text": closing_text, "reply_structured": closing_reply_structured,
            "tts_b64": tts_b64, "time": datetime.utcnow().isoformat() + "Z"
        })
        
        # ... (Firestore saving logic is unchanged)
        
        del SESSIONS[session_id]
        _logger.info("Session ended: %s", session_id)
        return {
            "status": "ended", "session_id": session_id,
            "closing_reply": closing_reply_structured, "tts_b64": tts_b64,
            "total_exchanges": len(session["history"])
        }
    except HTTPException: raise
    except Exception as e:
        _logger.exception("Session end failed")
        raise HTTPException(status_code=500, detail="Failed to end session")

# ... (debug endpoints remain the same)
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
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Brain1 direct call failed")
        raise HTTPException(status_code=500, detail="Brain1 processing failed")

@app.post("/brain2")
async def endpoint_brain2(payload: dict):
    try:
        decision = payload.get("decision")
        session_history = payload.get("session_history", [])
        language_code = payload.get("language_code", "en-US") # Allow debug override
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
    # ... (function is unchanged)
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
    # ... (function is unchanged)
    return {
        "status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z",
        "sessions_active": len(SESSIONS), "vertex_available": bool(VERTEX_MODEL_NAME)
    }

@app.get("/_debug_env")
async def debug_env():
    # ... (function is unchanged)
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