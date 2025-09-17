# empathetic-ai-therapist/app/main.py
"""
main.py - FastAPI application entrypoint for HealAura (formerly Empathetic AI Therapist)

Purpose:
- Exposes HTTP endpoints for starting/ending sessions, processing user audio, and marking tasks done.
- Orchestrates the core pipeline:
    upload audio -> audio processing -> Brain1 decision -> Brain2 response -> TTS
- MODIFIED: Persists conversation turns AND short-lived session state to Firestore,
  making the application completely stateless and cost-effective.
- This file focuses on orchestration and wiring; core algorithms live in other modules.
"""

import os
import logging
import base64
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict

from dotenv import load_dotenv
import pytz

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# Import app components (these modules contain the heavy lifting)
from .audio_processing import process_audio_file_bytes
from .gcp_clients import get_firestore_client, text_to_speech_bytes, init_vertex, VERTEX_MODEL_NAME
from .brain1_policy import call_brain1
from .brain2_response import call_brain2
from .utils import detect_safety
from . import auth
from . import wellness

# --- Basic Configuration ---
load_dotenv()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(title="HealAura")
app.include_router(auth.router)
app.include_router(wellness.router, prefix="/wellness")

# --- Static Files Mounting ---
import pathlib
BASE_DIR = pathlib.Path(__file__).resolve().parent
static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# --- App Constants & Configuration ---
# A dedicated Firestore collection for storing active, temporary session data
FIRESTORE_SESSIONS_COLLECTION = "live_sessions"
# The subcollection for storing permanent conversation history
FIRESTORE_HISTORY_SUBCOLLECTION = "therapy_sessions"
FIRESTORE_USERS_COLLECTION = "users"
SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "30"))

# --- NEW: Firestore-backed Session Management Functions ---

def get_session(session_id: str) -> Dict[str, Any]:
    """
    Retrieves session data from Firestore and implements 'lazy' expiration.

    Behavior:
    - Checks the 'expires_at' timestamp in the document.
    - If the session is expired, it deletes the document and returns a 404.
    - If the session is valid, it refreshes the 'expires_at' timestamp and returns the data.
    """
    client = get_firestore_client()
    if not client:
        raise HTTPException(status_code=503, detail="Session service (Firestore) is unavailable.")
    
    doc_ref = client.collection(FIRESTORE_SESSIONS_COLLECTION).document(session_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session_data = doc.to_dict()
    
    # Check for expiration
    expires_at = datetime.fromisoformat(session_data.get("expires_at", ""))
    if datetime.utcnow() > expires_at:
        doc_ref.delete() # Clean up the expired session document
        _logger.info("Cleaned up expired Firestore session: %s", session_id)
        raise HTTPException(status_code=404, detail="Session has expired.")

    # If valid, refresh the expiration time for another timeout period
    new_expires_at = datetime.utcnow() + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    doc_ref.update({"expires_at": new_expires_at.isoformat()})
    
    return session_data

def save_session(session_id: str, session_data: Dict[str, Any]):
    """Saves or updates session data in Firestore, setting an expiration timestamp."""
    client = get_firestore_client()
    if not client:
        _logger.error("Cannot save session %s, Firestore client is not available.", session_id)
        return
    
    # Add or update the expiration timestamp
    expires_at = datetime.utcnow() + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    session_data["expires_at"] = expires_at.isoformat()
    
    doc_ref = client.collection(FIRESTORE_SESSIONS_COLLECTION).document(session_id)
    doc_ref.set(session_data)

def delete_session(session_id: str):
    """Explicitly deletes a session document from Firestore, e.g., on session end."""
    client = get_firestore_client()
    if not client: return
    try:
        client.collection(FIRESTORE_SESSIONS_COLLECTION).document(session_id).delete()
    except Exception as e:
        _logger.warning("Could not delete session %s from Firestore: %s", session_id, e)

# --- Other Helper Functions ---

def _save_turn_to_firestore_history(session: Dict[str, Any], turn_data: Dict[str, Any]):
    """Appends a conversation turn to a user's permanent session history document."""
    user_id = session.get("user_id")
    if not user_id: return
    try:
        client = get_firestore_client()
        if not client: return
        from google.cloud import firestore

        # Get or create a consistent ID for the historical log document
        firestore_doc_id = session.get("firestore_doc_id")
        if not firestore_doc_id:
            ist_now = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Kolkata'))
            firestore_doc_id = ist_now.strftime('%Y-%m-%d_%H-%M-%S_IST')
            session["firestore_doc_id"] = firestore_doc_id # IMPORTANT: Update session object to save this ID
        
        doc_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(user_id).collection(FIRESTORE_HISTORY_SUBCOLLECTION).document(firestore_doc_id)
        doc_ref.set({"conversation": firestore.ArrayUnion([turn_data])}, merge=True)
    except Exception as e:
        _logger.error("Failed to save turn to Firestore history for user %s: %s", user_id, e)

def _determine_ai_gender(user_gender: str) -> str:
    """Selects an AI voice gender based on user's gender."""
    return "FEMALE" if (user_gender or "").lower() == "male" else "MALE"

# --- FastAPI Events ---

@app.on_event("startup")
async def startup_event():
    _logger.info("HealAura starting up")
    init_vertex()

# --- Core API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(index_path, "r", encoding="utf-8") as fh:
        return fh.read()

INITIAL_GREETINGS = {
    "en-US": "Hi, I'm here to listen. How are you feeling today? Take your time to share as much as you're comfortable with.",
    "hi-IN": "नमस्ते, मैं आपकी बात सुनने के लिए यहाँ हूँ। आज आप कैसा महसूस कर रहे हैं? आप जितना चाहें, उतना साझा करने के लिए अपना समय लें।"
}
CLOSING_MESSAGES = {
    "en-US": "Thank you for sharing your time with me today. Remember to be kind to yourself as you continue through your day.",
    "hi-IN": "आज मेरे साथ अपना समय साझा करने के लिए धन्यवाद। जैसे-जैसे आप अपने दिन में आगे बढ़ें, खुद पर दया करना याद रखें।"
}

@app.post("/start_session")
async def start_session(user_id: str = Form(...), language: str = Form("en-US"), user_gender: str = Form("female")):
    session_id = f"session_{uuid.uuid4()}"
    session = {
        "session_id": session_id, "user_id": user_id, "language": language,
        "user_gender": user_gender, "created_at": datetime.utcnow().isoformat() + "Z",
        "history": [], "tasks": [], "technique_history": [], "firestore_doc_id": None
    }

    initial_text = INITIAL_GREETINGS.get(language, INITIAL_GREETINGS["en-US"])
    _save_turn_to_firestore_history(session, {"who": "ai", "text": f"--- NEW SESSION --- {initial_text}", "time": session["created_at"]})
    
    tts_bytes = text_to_speech_bytes(initial_text, lang_code=language, gender=_determine_ai_gender(user_gender))
    initial_reply = {"response_parts": [{"type": "plain", "text": initial_text}]}
    
    session["history"].append({"who": "ai", "text": initial_text})
    save_session(session_id, session) # Save the initial session state to Firestore
    
    _logger.info("Started new Firestore-backed session %s for user %s", session_id, user_id)
    return {"session_id": session_id, "initial_reply": initial_reply, "tts_b64": base64.b64encode(tts_bytes).decode() if tts_bytes else None}

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...), session_id: str = Form(...)):
    session = get_session(session_id)
    
    audio_data = await file.read()
    audio_output = process_audio_file_bytes(audio_data, language_code=session["language"])
    
    user_text = audio_output.get("transcript", "(silence)")
    session["history"].append({"who": "user", "text": user_text})
    _save_turn_to_firestore_history(session, {"who": "user", "text": user_text, "time": datetime.utcnow().isoformat() + "Z"})
    
    decision = call_brain1(audio_output, session["history"], session.get("technique_history"))
    reply = call_brain2(decision, session["history"], language_code=session["language"])
    
    ai_text = reply.get("response_text_concatenated", "")
    session["history"].append({"who": "ai", "text": ai_text})
    _save_turn_to_firestore_history(session, {"who": "ai", "text": ai_text, "time": datetime.utcnow().isoformat() + "Z"})
    
    save_session(session_id, session) # Persist the updated session state
    
    tts_bytes = text_to_speech_bytes(ai_text, lang_code=session["language"], gender=_determine_ai_gender(session["user_gender"]))
    return {"audio_output": audio_output, "decision": decision, "reply": reply, "tts_b64": base64.b64encode(tts_bytes).decode() if tts_bytes else None}

@app.post("/task_done")
async def task_done(session_id: str = Form(...), task_id: str = Form(...)):
    session = get_session(session_id)
    
    system_text = f"TASK_COMPLETED:{task_id}"
    session["history"].append({"who": "system", "text": system_text})
    _save_turn_to_firestore_history(session, {"who": "system", "text": system_text, "time": datetime.utcnow().isoformat() + "Z"})

    task_input = {"transcript": "(user just completed the assigned task)"}
    decision = call_brain1(task_input, session["history"], session.get("technique_history"))
    reply = call_brain2(decision, session["history"], language_code=session["language"])
    
    ai_text = reply.get("response_text_concatenated", "")
    session["history"].append({"who": "ai", "text": ai_text})
    _save_turn_to_firestore_history(session, {"who": "ai", "text": ai_text, "time": datetime.utcnow().isoformat() + "Z"})
    
    save_session(session_id, session) # Persist the updated session state

    tts_bytes = text_to_speech_bytes(ai_text, lang_code=session["language"], gender=_determine_ai_gender(session["user_gender"]))
    return {"status": "completed", "decision": decision, "reply": reply, "tts_b64": base64.b64encode(tts_bytes).decode() if tts_bytes else None}

@app.post("/end_session")
async def end_session(background_tasks: BackgroundTasks, session_id: str = Form(...)):
    session = get_session(session_id)
    
    closing_text = CLOSING_MESSAGES.get(session["language"], CLOSING_MESSAGES["en-US"])
    _save_turn_to_firestore_history(session, {"who": "ai", "text": f"--- SESSION ENDED ---", "time": datetime.utcnow().isoformat() + "Z"})
    
    # Run summary generation in the background to not delay the user's response
    background_tasks.add_task(
        wellness.generate_session_summary,
        user_id=session["user_id"], session_history=session["history"], session_start_iso=session["created_at"]
    )
    
    delete_session(session_id) # Delete the temporary session document
    _logger.info("Session ended and removed from Firestore 'live_sessions': %s", session_id)
    
    tts_bytes = text_to_speech_bytes(closing_text, lang_code=session["language"], gender=_determine_ai_gender(session["user_gender"]))
    closing_reply = {"response_parts": [{"type": "plain", "text": closing_text}]}
    return {"status": "ended", "closing_reply": closing_reply, "tts_b64": base64.b64encode(tts_bytes).decode() if tts_bytes else None}

# --- Health & Debug Endpoints ---

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    client = get_firestore_client()
    return {"status": "healthy", "firestore_available": bool(client), "vertex_available": bool(VERTEX_MODEL_NAME)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)