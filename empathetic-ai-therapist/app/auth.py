"""
auth.py - Authentication module for Empathetic AI Therapist

This module provides user authentication features using FastAPI and Google Firestore.
It includes the following functionality:
1. User Signup:
   - Creates a new user account with username and password.
   - Validates password confirmation and minimum length.
   - Stores hashed password in Firestore for security.
   - Initializes a wellness cache for future wellness-related data.
   
2. User Login:
   - Verifies the username and password entered by the user.
   - Compares input password with securely stored hashed password.
   - Returns a success message and user_id if credentials are valid.

Security Features:
- Passwords are hashed using bcrypt via Passlib (never stored in plain text).
- Firestore is used as the backend database to persist user data.
- Basic error handling with logging for debugging and monitoring.

Note:
- This module does NOT use JWT tokens or session management.
- It is designed for anonymous accounts (no email is required).
"""

import logging 
from datetime import datetime

from fastapi import APIRouter, Form, HTTPException
from passlib.context import CryptContext

from .gcp_clients import get_firestore_client

# Suppress harmless bcrypt warnings from Passlib
logging.getLogger("passlib.handlers.bcrypt").setLevel(logging.ERROR)

# Logger for tracking authentication activities
_logger = logging.getLogger(__name__)

# Create a FastAPI router to group authentication endpoints
router = APIRouter()

# Firestore collection where user documents will be stored
FIRESTORE_USERS_COLLECTION = "users"

# Password hashing context (bcrypt is used for secure hashing)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    """
    Verifies a plain password against its hashed version.
    Returns True if the password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password):
    """
    Hashes a plain password using bcrypt.
    This ensures passwords are never stored in plain text.
    """
    return pwd_context.hash(password)


@router.post("/signup")
async def signup(
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    """
    Endpoint to create a new user account.
    Steps:
    1. Validate that passwords match and meet length requirements.
    2. Check if the username already exists in Firestore.
    3. Hash the password and store user data in Firestore.
    4. Initialize a wellness cache field for later use.
    """
    # Ensure passwords match
    if password != confirm_password:
        _logger.warning("Signup failed for user '%s': Passwords do not match.", username)
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    # Enforce minimum password length
    if len(password) < 6:
        _logger.warning("Signup failed for user '%s': Password too short.", username)
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")

    # Get Firestore client
    client = get_firestore_client()
    if not client:
        _logger.error("Signup failed: Could not connect to Firestore.")
        raise HTTPException(status_code=500, detail="Could not connect to database.")

    try:
        # Reference to the user document
        user_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(username)

        # Check if username already exists
        if user_ref.get().exists:
            _logger.warning("Signup failed: Username '%s' already exists.", username)
            raise HTTPException(status_code=400, detail="Username already exists.")

        # Securely hash the password
        hashed_pwd = hash_password(password)

        # Prepare user data to store in Firestore
        user_data = {
            "username": username,
            "hashed_password": hashed_pwd,
            "created_at": datetime.utcnow().isoformat() + "Z",  # Store timestamp in UTC
            "wellness_cache": {  # Placeholder for wellness-related data
                "last_updated_utc": None,
                "data": None
            }
        }

        # Save user document in Firestore
        user_ref.set(user_data)

        _logger.info("New user created successfully: %s", username)
        return {"status": "success", "message": "Account created successfully."}

    except Exception as e:
        # Catch unexpected errors (e.g., network/database issues)
        _logger.exception("An unexpected error occurred during signup for user '%s': %s", username, e)
        raise HTTPException(status_code=500, detail="An internal error occurred during signup.")


@router.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """
    Endpoint to log in an existing user.
    Steps:
    1. Look up the user in Firestore using the username.
    2. If the user exists, retrieve the hashed password.
    3. Verify that the provided password matches the stored hash.
    4. Return success with the user_id if authentication passes.
    """
    # Get Firestore client
    client = get_firestore_client()
    if not client:
        _logger.error("Login failed: Could not connect to Firestore.")
        raise HTTPException(status_code=500, detail="Could not connect to database.")

    try:
        # Reference to the user document
        user_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(username)
        user_doc = user_ref.get()

        # Check if user exists
        if not user_doc.exists:
            _logger.warning("Login failed for user '%s': User not found.", username)
            raise HTTPException(status_code=404, detail="Invalid username or password.")

        user_data = user_doc.to_dict()
        hashed_pwd = user_data.get("hashed_password")

        # Verify the password
        if not hashed_pwd or not verify_password(password, hashed_pwd):
            _logger.warning("Login failed for user '%s': Invalid password.", username)
            raise HTTPException(status_code=401, detail="Invalid username or password.")

        _logger.info("User logged in successfully: %s", username)
        return {"status": "success", "user_id": user_doc.id}

    except HTTPException:
        # Re-raise known exceptions (invalid password, user not found, etc.)
        raise
    except Exception as e:
        # Catch unexpected errors (e.g., Firestore issues)
        _logger.exception("An unexpected error occurred during login for user '%s': %s", username, e)
        raise HTTPException(status_code=500, detail="An internal error occurred during login.")
