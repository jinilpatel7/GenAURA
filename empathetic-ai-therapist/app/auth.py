import logging
from datetime import datetime

from fastapi import APIRouter, Form, HTTPException
from passlib.context import CryptContext

from .gcp_clients import get_firestore_client

# Suppress the specific, harmless warning from passlib about the bcrypt backend
logging.getLogger("passlib.handlers.bcrypt").setLevel(logging.ERROR)

# Module logger
_logger = logging.getLogger(__name__)

# Create a router to organize auth endpoints
router = APIRouter()

# Firestore collection name for users
FIRESTORE_USERS_COLLECTION = "users"

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    """Verifies a plain password against a hashed one."""
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password):
    """Hashes a plain password."""
    return pwd_context.hash(password)

@router.post("/signup")
async def signup(
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    """Endpoint to create a new anonymous user account."""
    if password != confirm_password:
        _logger.warning("Signup failed for user '%s': Passwords do not match.", username)
        raise HTTPException(status_code=400, detail="Passwords do not match.")
    if len(password) < 6:
        _logger.warning("Signup failed for user '%s': Password too short.", username)
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")

    client = get_firestore_client()
    if not client:
        _logger.error("Signup failed: Could not connect to Firestore.")
        raise HTTPException(status_code=500, detail="Could not connect to database.")

    try:
        user_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(username)
        if user_ref.get().exists:
            _logger.warning("Signup failed: Username '%s' already exists.", username)
            raise HTTPException(status_code=400, detail="Username already exists.")

        hashed_pwd = hash_password(password)
        user_data = {
            "username": username,
            "hashed_password": hashed_pwd,
            "created_at": datetime.utcnow().isoformat() + "Z",
            # REMOVED: Email preferences are no longer stored for anonymity.
            # NEW: Initialize the wellness cache structure for performance optimization
            "wellness_cache": {
                "last_updated_utc": None,
                "data": None
            }
        }
        user_ref.set(user_data)
        _logger.info("New user created successfully: %s", username)
        return {"status": "success", "message": "Account created successfully."}
    except Exception as e:
        _logger.exception("An unexpected error occurred during signup for user '%s': %s", username, e)
        raise HTTPException(status_code=500, detail="An internal error occurred during signup.")


@router.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Endpoint to log a user in."""
    client = get_firestore_client()
    if not client:
        _logger.error("Login failed: Could not connect to Firestore.")
        raise HTTPException(status_code=500, detail="Could not connect to database.")

    try:
        user_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(username)
        user_doc = user_ref.get()

        if not user_doc.exists:
            _logger.warning("Login failed for user '%s': User not found.", username)
            raise HTTPException(status_code=404, detail="Invalid username or password.")

        user_data = user_doc.to_dict()
        hashed_pwd = user_data.get("hashed_password")

        if not hashed_pwd or not verify_password(password, hashed_pwd):
            _logger.warning("Login failed for user '%s': Invalid password.", username)
            raise HTTPException(status_code=401, detail="Invalid username or password.")

        _logger.info("User logged in successfully: %s", username)
        return {"status": "success", "user_id": user_doc.id}
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("An unexpected error occurred during login for user '%s': %s", username, e)
        raise HTTPException(status_code=500, detail="An internal error occurred during login.")