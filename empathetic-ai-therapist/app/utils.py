"""
utils.py - Utility functions for Empathetic AI Therapist

This module provides safety-related checks for user transcripts.
It is mainly used to detect potentially harmful or crisis-related language.

Key Features:
1. Maintains a list of **safety keywords/phrases** that may indicate
   self-harm, suicidal ideation, or urgent crises.
2. Uses regular expressions to detect these keywords in user transcripts.
3. Returns whether a transcript contains a concerning keyword
   and provides the matched keyword for further handling.

Example Usage:
- If the user says: "I want to die", the function will flag it as a crisis.
- If the user says: "I can't go on with work tasks", it will NOT flag it,
  since this is a non-crisis context (false positive prevention).
"""

import re

# List of safety-related keywords and phrases.
# These cover expressions of suicidal thoughts, self-harm,
# hopelessness, and emergency situations.
SAFETY_KEYWORDS = [
    "suicide", "kill myself", "end my life", "hurt myself", "want to die",
    "i will kill myself", "i'm going to kill myself", "i want to die", "i wish i was dead",
    "no reason to live", "better off dead", "can't go on", "put an end to it",
    "cut myself", "self-harm", "hurting myself", "harm myself", "self injury",
    "emergency", "call the police", "911", "crisis", "can't cope", "breaking point", "at my limit",
    "no hope", "no point", "won't get better", "drug overdose", "hang myself", "gun", "weapon"
]

# Precompiled regex pattern for performance.
# It matches any of the safety keywords (case-insensitive).
_SAFETY_RE = re.compile(
    r"\b(" + r"|".join(re.escape(k) for k in SAFETY_KEYWORDS) + r")\b", 
    flags=re.IGNORECASE
)

def detect_safety(transcript: str):
    """
    Detects whether a transcript contains any safety keyword.

    Args:
        transcript (str): The user's input text.

    Returns:
        tuple:
            - (bool) True if a safety keyword is detected, False otherwise.
            - (str or None) The matched keyword, or None if nothing matched.

    Special Handling:
    - Avoids false positives in cases like:
        "help me feel better" (not a crisis, though "help me" is present).
        "can't go on with work tasks" (work-related frustration, not a crisis).
    """
    # Handle empty or whitespace-only input
    if not transcript or not transcript.strip():
        return False, None
    
    # Search for a safety keyword in the transcript
    m = _SAFETY_RE.search(transcript)
    if not m:
        return False, None
    
    # Extract the matched keyword
    matched_keyword = m.group(0)
    context = transcript.lower()  # Lowercased transcript for phrase checks

    # Phrases starting with "help me ..." are usually requests, not crisis
    non_crisis_phrases = [
        "help me feel", "help me to", "help me be", "help me get", "help me with",
        "help me understand", "help me know", "help me see", "help me find"
    ]
    
    # Exception 1: Ignore "help me" if it's part of a harmless phrase
    if matched_keyword.lower() == "help me" and any(phrase in context for phrase in non_crisis_phrases):
        return False, None
    
    # Exception 2: Ignore "can't go on" if context is about work/tasks (not crisis)
    if matched_keyword.lower() == "can't go on" and "work" in context and "tasks" in context:
        return False, None
    
    # If no exception matched, return as a crisis detection
    return True, matched_keyword
