import re

SAFETY_KEYWORDS = [
    "suicide", "kill myself", "end my life", "hurt myself", "want to die",
    "i will kill myself", "i'm going to kill myself", "i want to die", "i wish i was dead",
    "no reason to live", "better off dead", "can't go on", "put an end to it",
    "cut myself", "self-harm", "hurting myself", "harm myself", "self injury",
    "emergency", "call the police", "911", "crisis", "can't cope", "breaking point", "at my limit",
    "no hope", "no point", "won't get better", "drug overdose", "hang myself", "gun", "weapon"
]

_SAFETY_RE = re.compile(
    r"\b(" + r"|".join(re.escape(k) for k in SAFETY_KEYWORDS) + r")\b", 
    flags=re.IGNORECASE
)

def detect_safety(transcript: str):
    if not transcript or not transcript.strip():
        return False, None
    
    m = _SAFETY_RE.search(transcript)
    if not m:
        return False, None
    
    matched_keyword = m.group(0)
    context = transcript.lower()
    non_crisis_phrases = [
        "help me feel", "help me to", "help me be", "help me get", "help me with",
        "help me understand", "help me know", "help me see", "help me find"
    ]
    
    if matched_keyword.lower() == "help me" and any(phrase in context for phrase in non_crisis_phrases):
        return False, None
    
    if matched_keyword.lower() == "can't go on" and "work" in context and "tasks" in context:
        return False, None
    
    return True, matched_keyword