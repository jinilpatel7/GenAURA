# empathetic-ai-therapist/app/brain2_response.py
import json
import re
import logging
import uuid
from typing import Optional, List, Dict, Any
from .gcp_clients import vertex_generate

_logger = logging.getLogger(__name__)

# *** NEW: Language name mapping for clearer prompts ***
LANGUAGE_NAMES = {
    "en-US": "English",
    "hi-IN": "Hindi"
}

# --- Helper functions to get full session history ---

def _get_all_ai_texts(session_history: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Return ALL AI replies from session_history in chronological order."""
    if not session_history:
        return []
    return [h.get('text', '') for h in session_history if h.get('who') == 'ai']

def _get_all_user_texts(session_history: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Return ALL user messages from session_history in chronological order."""
    if not session_history:
        return []
    return [h.get('text', '') for h in session_history if h.get('who') == 'user']

# --- Core Prompting and Response Generation ---

def _build_prompt(decision_json: dict, session_history: list, language_code: str) -> str:
    """
    Builds the prompt for Brain2, instructing it to generate a therapeutic response
    based on Brain1's decision and the full conversation context.
    *** MODIFIED: Now includes language instruction. ***
    """
    all_ai_replies = _get_all_ai_texts(session_history)
    all_user_messages = _get_all_user_texts(session_history)
    decision = decision_json.get("decision", "")
    language_name = LANGUAGE_NAMES.get(language_code, "English")

    # Step 1: Explain the role of Brain2 in detail
    prompt = (
        "You are Brain2 — the THERAPEUTIC RESPONSE GENERATOR. Your role is to convert a strategic decision from Brain1 into a warm, natural, and effective user-facing response. You are the voice of the therapist.\n\n"
    )

    # *** MODIFIED: Added critical language instruction at the top ***
    prompt += (
        f"**CRITICAL LANGUAGE INSTRUCTION:** You MUST generate your entire response in {language_name} (language code: {language_code}). All text in your final JSON output must be in {language_name}.\n\n"
    )

    # Step 2: Explain state management and tool usage via JSON output
    prompt += (
        "STATE MANAGEMENT & TOOLS:\n"
        "Your primary tool is the JSON output you generate. To manage states like tasks or provide resources, you must create specific JSON structures.\n"
        "- For Tasks: If Brain1's decision includes a task, generate a `task_meta` object and a corresponding `display_cards` entry for a timer.\n"
        "  - `task_meta` example: `{\"task_id\": \"...\", \"type\": \"breathing\", \"detail\": \"1. Breathe in for 4s. 2. Hold for 7s. 3. Exhale for 8s.\", \"duration_sec\": 30}`\n"
        "  - `display_cards` example: `[{\"type\": \"timer\", \"duration_sec\": 30, \"button_id\": \"mark_done_...\"}]`\n"
        "- For Resources: If you need to suggest an article or link, include it in the `response_text` and optionally create a `display_cards` entry of type 'link'.\n\n"
    )

    # Step 3: Provide all context and the specific instructions from Brain1
    prompt += (
        "--- CONTEXT & INSTRUCTIONS ---\n"
        "Here is the complete session history and the decision made by Brain1. Analyze everything carefully to craft your response.\n\n"
        "FULL USER MESSAGE HISTORY:\n"
        + ("\n".join(f"- \"{u}\"" for u in all_user_messages) if all_user_messages else "None") + "\n\n"
        "FULL AI REPLY HISTORY (Avoid repeating phrasing from these):\n"
        + ("\n".join(f"- \"{a}\"" for a in all_ai_replies) if all_ai_replies else "None") + "\n\n"
        f"BRAIN1's DECISION & GUIDANCE (Your primary instruction):\n"
        f"{json.dumps(decision_json, indent=2)}\n\n"
    )
    
    if decision == "safety_crisis+task":
        prompt += (
            "**CRITICAL SAFETY INSTRUCTION:** Brain1 has detected a safety crisis. Your response must be calm, direct, and non-judgmental. Your first response part MUST be of type 'safety_crisis' and provide immediate resources (like the 988 hotline in the US, or an appropriate equivalent for the language). Follow this with a gentle introduction to a simple grounding task with clear, numbered steps. Crucially, end your response in a way that keeps the conversation open, inviting the user to continue after the task if they feel ready. This is not about blocking them, but about providing immediate support while keeping the door open.\n\n"
        )
    elif decision.endswith("+task"):
        prompt += (
            "**TASK INSTRUCTION GUIDANCE:** Brain1 has decided a task is needed. Your response must include:\n"
            "1. An empathetic lead-in.\n"
            "2. A `task_instruction` part that introduces the task and gives **clear, simple, numbered, step-by-step instructions**.\n"
            "3. A `task_meta` object where the `detail` field contains a concise summary of the steps (e.g., \"1. Inhale for 4s. 2. Hold for 4s. 3. Exhale for 6s.\"). This `detail` will be displayed on screen.\n\n"
        )

    # Step 4: Define the required output format and new instructions
    prompt += (
        "--- YOUR TASK ---\n"
        f"Based on all the above, generate the final response for the user in {language_name}. Follow Brain1's guidance precisely for tone, content, and required elements.\n\n"
        "**NEW INSTRUCTIONS & OUTPUT FORMAT:**\n"
        "1. **Structured Response:** You MUST break your response down into a list of objects in the `response_parts` field. Each part has a `type` and `text`.\n"
        "2. **Allowed Types:** The `type` for each part must be one of the following: `empathy`, `safety_crisis`, `question`, `suggestion`, `task_instruction`, `plain`.\n"
        "3. **Provide Scaffolding:** When asking a question (`type: 'question'`), add a helpful example or a gentle starting point in a separate part with `type: 'suggestion'`.\n"
        "4. **Explain the 'Why':** Always provide a brief, simple explanation for your technique in the `psychology_behind_it` field. This helps the user understand the purpose of the exercise or question.\n"
        "5. **Plain Text Only:** Do NOT use any Markdown formatting (like asterisks, hyphens, or hashes). The response must be clean, plain text ready for Text-to-Speech.\n\n"
        "Return STRICT JSON only with the following keys:\n"
        "{\n"
        "  \"response_parts\": [\n"
        "     {\"type\": \"empathy\", \"text\": \"The empathetic reflection goes here.\"},\n"
        "     {\"type\": \"task_instruction\", \"text\": \"Now let's try a simple exercise. First, ... Second, ... Third, ...\"},\n"
        "     {\"type\": \"question\", \"text\": \"The main question for the user.\"}\n"
        "  ],\n"
        "  \"psychology_behind_it\": \"A brief, gentle explanation of the therapeutic principle behind this response.\",\n"
        "  \"tts_voice\": {\"lang\": \"en-US\", \"voice\": \"en-US-Studio-O\"}, \n"
        "  \"display_cards\": [{\"type\": \"timer\", ...}] or [],\n"
        "  \"task_meta\": {\"task_id\": \"...\", \"type\": \"breathing\", \"detail\": \"1. Step one. 2. Step two.\", ...} or null\n"
        "}\n\n"
        "Return JSON only."
    )
    return prompt

def _api_failure_fallback(reason: str) -> dict:
    # ... (fallback remains the same, it's a system message)
    _logger.error("Brain2 failed and is using the critical API fallback. Reason: %s", reason)
    return {
        "response_parts": [{
            "type": "plain",
            "text": "I'm sorry, I'm having a little trouble connecting at the moment. Could you please say that again?"
        }],
        "response_text_concatenated": "I'm sorry, I'm having a little trouble connecting at the moment. Could you please say that again?",
        "psychology_behind_it": "This is a fallback response due to a temporary system issue.",
        "tts_voice": {"lang": "en-US", "voice": "en-US-Studio-O"},
        "display_cards": [],
        "task_meta": None
    }

def _validate_and_sanitize_output(parsed: dict, language_code: str) -> dict:
    """
    Ensures the LLM output conforms to the required data structure and is clean for TTS.
    *** MODIFIED: Now sets the correct language code in the output. ***
    """
    if not parsed or not isinstance(parsed, dict):
        raise ValueError("Parsed output is not a dictionary.")

    response_parts = parsed.get("response_parts", [])
    if not response_parts or not isinstance(response_parts, list):
         raise ValueError("'response_parts' is missing or not a list.")

    sanitized_parts = []
    concatenated_text = []
    for part in response_parts:
        if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
            text = part["text"]
            sanitized_text = re.sub(r'[\*_`#\-]', ' ', text)
            sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip()
            
            if sanitized_text:
                sanitized_parts.append({
                    "type": part.get("type", "plain"),
                    "text": sanitized_text
                })
                concatenated_text.append(sanitized_text)

    if not sanitized_parts:
        raise ValueError("'response_parts' contained no valid text parts.")
    
    final_concatenated_text = " ".join(concatenated_text)

    psychology_behind_it = parsed.get("psychology_behind_it")
    if not isinstance(psychology_behind_it, str):
        psychology_behind_it = "The goal is to provide a supportive space for you to explore your thoughts and feelings."

    tts_voice = parsed.get("tts_voice", {})
    if not isinstance(tts_voice, dict):
        tts_voice = {}
    
    # *** MODIFIED: Set the language code from the session ***
    tts_voice["lang"] = language_code
    # The LLM might suggest a voice, but we just need the language. The specific voice is chosen in gcp_clients.
    tts_voice.pop("voice", None)


    display_cards = parsed.get("display_cards", [])
    if not isinstance(display_cards, list):
        display_cards = []
        
    task_meta = parsed.get("task_meta")
    if task_meta and not isinstance(task_meta, dict):
        task_meta = None
    if task_meta:
        task_meta.setdefault("task_id", str(uuid.uuid4()))

    return {
        "response_parts": sanitized_parts,
        "response_text_concatenated": final_concatenated_text,
        "psychology_behind_it": psychology_behind_it.strip(),
        "tts_voice": tts_voice,
        "display_cards": display_cards,
        "task_meta": task_meta
    }

def call_brain2(decision_json: dict, session_history: list, language_code: str = "en-US"):
    """
    Calls the LLM to generate the final user-facing response based on Brain1's decision.
    *** MODIFIED: Accepts language_code. ***
    """
    try:
        prompt = _build_prompt(decision_json, session_history or [], language_code)
        raw_response = vertex_generate(prompt, max_output_chars=4096, temperature=0.5)

        if not raw_response:
            return _api_failure_fallback("LLM returned an empty response")

        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if not match:
            _logger.warning("Brain2 response did not contain a JSON object. Raw: %s", raw_response)
            return _api_failure_fallback("No JSON object found in LLM response")

        json_str = match.group(0)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            _logger.warning("Brain2 JSON parsing failed: %s. Attempting to repair. Raw JSON: %s", e, json_str)
            repaired = json_str.replace("'", '"')
            repaired = re.sub(r'("\s*:\s*".*?")\s*,\s*({)', r'\1}, \2', repaired)
            repaired = re.sub(r",\s*([}```])", r"\1", repaired)
            
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as e2:
                _logger.warning("Brain2 JSON repair attempt also failed: %s", e2)
                return _api_failure_fallback(f"JSONDecodeError: {e}; repair failed: {e2}")

        return _validate_and_sanitize_output(parsed, language_code)

    except Exception as e:
        _logger.exception("An unexpected error occurred in Brain2. Returning critical fallback.")
        return _api_failure_fallback(f"Unhandled exception: {str(e)}")