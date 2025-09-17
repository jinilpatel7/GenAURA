"""
brain1_policy.py - Decision engine (Brain1) for empathetic voice sessions

Purpose:
- Brain1 is the DECISION ENGINE for short empathetic voice sessions.
- It analyzes the latest audio-derived output, full session history, and
  past techniques to choose a single, therapeutically-appropriate next step.
- The decision is returned as strict JSON (decision + short guidance) and is
  intended to be consumed by Brain2 (the response generator / voice component).

Key behaviors:
- Respects a whitelist of allowed decision combinations (ALLOWED_COMBINATIONS).
- Includes a safety override: if audio processing flagged a potential crisis,
  Brain1 immediately returns the "safety_crisis+task" decision with explicit
  guidance (no model call required).
- When Vertex AI is used to make the decision, Brain1 constructs a detailed
  prompt with session context, instructs the model to return JSON only, and
  includes robust fallback handling for API failures or malformed responses.
- Designed to be conservative: if the LLM output is missing, malformed, or
  proposes an invalid decision, Brain1 returns a minimal, safe fallback JSON
  designed to keep the conversation open and supportive.

Note:
- This file documents intent and control flow only. No therapeutic "content"
  logic (e.g., wording of empathy statements) is enforced here — Brain2 is
  expected to implement the raw guidance and phrasing.
"""
import json
import re
import logging
from typing import List, Optional, Dict, Any
from .gcp_clients import vertex_generate

_logger = logging.getLogger(__name__)

# *** MODIFIED: Allowed decisions now include safety_crisis+task ***
ALLOWED_COMBINATIONS = [
    "empathy+task",
    "empathy+follow_up",
    "empathy+metacognitive",
    "empathy+CBT",
    "safety_crisis+task"
]

# Therapeutic concepts remain the same
THERAPEUTIC_CONCEPTS = """
1. Empathy
Definition: Empathy is the AI’s sensitive recognition and validation of the user’s emotional experience, reflecting both the words spoken and the tone detected.
Example Response: “I hear that you’re under a lot of pressure with so many tasks from your boss—it makes sense that you’d feel stressed right now.”

2. Task (or Exercise)
Definition: A task is a short, guided, actionable activity (30s-3min) to help the user regulate emotions or shift perspective.
Examples: Breathing exercise to lower stress, a grounding technique to reduce anxiety, or a micro-action like drinking water.

3. Metacognition (Thought Awareness & Reflection)
Definition: Metacognition is the ability to notice and reflect on one's own thoughts and emotions. The AI facilitates this by gently prompting users to become more aware of what’s happening in their mind.
Example Prompt: “What part of today’s workload felt the most overwhelming?”

4. CBT Micro-Step Guidance
Definition: CBT micro-steps are tiny, evidence-based cognitive techniques from Cognitive Behavioral Therapy (CBT). The AI offers them in short, practical forms to help users challenge unhelpful thoughts.
Examples:
- Reframing → “Is there another way to look at this situation?”
- Controllable Action → “What’s one small thing you can do differently right now?”

5. Follow-up Question
Definition: A question to ensure ongoing support, gather more information, and keep the conversation flowing naturally.

6. Resource Recommendation
Definition: When appropriate, suggest external support materials (articles, videos) or professional help for deeper guidance. This can be part of the guidance for Brain2.

7. Motivational Feedback
Definition: Effort-based encouragement that reinforces user engagement and celebrates progress. It makes users feel valued and supported.
Example: “Taking a moment to pause and reflect shows real strength.”
"""

# *** MODIFIED: Updated description for safety_crisis to include a task ***
POSSIBLE_COMBINATIONS = """
Here are the allowed decision combinations. You must choose exactly one:
1. empathy+task: An empathetic statement followed by a relevant, short, guided exercise. Use this when the user seems overwhelmed and needs immediate regulation. After the task is completed, the next turn's goal is to offer motivational feedback and then pivot back to the original problem with a follow-up or metacognitive question.
2. empathy+follow_up: An empathetic statement followed by a gentle, open-ended question to explore their feelings or situation further.
3. empathy+metacognitive: An empathetic statement followed by a question that encourages the user to reflect on their own thought patterns.
4. empathy+CBT: An empathetic statement followed by a small, actionable cognitive step to help the user reframe their thoughts.
5. safety_crisis+task: Use this ONLY if there is clear, intentional language about self-harm, suicide, or crisis. This involves providing immediate crisis resources followed by a relevant, short, guided exercise to help the user regulate in the moment.
"""

def _recent_ai_texts(session_history: Optional[List[Dict[str, Any]]]) -> List[str]:
    """
    Return ALL AI replies from session_history in chronological order.
    If session_history is empty or None, return empty list.
    """
    if not session_history:
        return []
    return [h.get('text', '') for h in session_history if h.get('who') == 'ai']

def _recent_user_texts(session_history: Optional[List[Dict[str, Any]]]) -> List[str]:
    """
    Return ALL user messages from session_history in chronological order.
    If session_history is empty or None, return empty list.
    """
    if not session_history:
        return []
    return [h.get('text', '') for h in session_history if h.get('who') == 'user']

def _recent_ai_techniques(tech_history: Optional[List[Dict[str, Any]]]) -> List[str]:
    """
    Return list of technique names used previously (all of them).
    If no technique history, return empty list.
    """
    if not tech_history:
        return []
    return [t.get('technique', '') for t in tech_history]

def _build_prompt(audio_output: Dict[str, Any], session_history: List[Dict[str, Any]], technique_history: Optional[List[Dict[str, Any]]]) -> str:
    """
    Builds the full educational prompt for the Brain1 LLM.
    Uses the full session history and special state-based instructions.
    """
    recent_ai = _recent_ai_texts(session_history or [])
    recent_user = _recent_user_texts(session_history or [])
    recent_techs = _recent_ai_techniques(technique_history or [])
    session_state = audio_output.get('session_state', {})
    last_event = session_state.get('last_event')
    audio_emotion = audio_output.get('final_emotion', 'neutral')
    audio_confidence = audio_output.get('confidence', 0.0)

    # Step 1: Define the LLM's role
    prompt = (
        "You are Brain1 — the DECISION ENGINE for a short empathetic voice session. Your role is to act as a thoughtful therapist, analyzing the user's input and deciding the most appropriate therapeutic next step.\n\n"
    )

    # Step 2: Teach the LLM the concepts and combinations
    prompt += (
        "First, learn these core therapeutic concepts:\n"
        f"{THERAPEUTIC_CONCEPTS}\n\n"
        f"{POSSIBLE_COMBINATIONS}\n\n"
    )

    # Step 3: Provide all the user inputs and context
    prompt += (
        "Now, analyze the following user input and session history to make your decision. Be diverse in your choices over time; do not choose the same strategy repeatedly unless therapeutically necessary.\n\n"
    )

    # Logic to handle pivoting back after a task.
    if last_event == 'task_completed':
        prompt += (
            "**CRITICAL INSTRUCTION**: The user has just completed a therapeutic task. Their latest message is a direct response to that task.\n"
            "**YOUR GOAL**: Your primary goal is to PIVOT back to the user's original problem. Follow these steps precisely:\n"
            "1. Briefly acknowledge their current state with motivational feedback.\n"
            "2. Immediately reconnect to the initial issue that prompted the task.\n"
            "3. Choose a 'metacognitive' or 'follow_up' decision to explore that original problem from their new, calmer perspective.\n\n"
        )
        
    # *** MODIFIED: Enhanced logic to encourage tasks and guide Brain2 more specifically. ***
    elif audio_emotion in ['sad', 'angry', 'fear'] and audio_confidence > 0.55:
        prompt += (
            "**PRIORITY INSTRUCTION**: The user's emotional state is detected as negative with high confidence. "
            "This indicates they may be feeling overwhelmed. Strongly consider choosing 'empathy+task' to provide an immediate regulation tool. "
            "This is often more helpful than just asking another question when someone is distressed. Avoid a task only if it was the *very last* thing the AI did.\n"
            "**If you choose a task, in your `guidance` for Brain2, suggest a specific, relevant task type (e.g., 'a simple box breathing exercise for stress' or 'a 5-senses grounding technique for anxiety').**\n\n"
        )

    prompt += (
        "--- INPUT DATA ---\n"
        f"USER'S LATEST MESSAGE (TRANSCRIPT): \"{(audio_output.get('transcript') or '').strip()}\"\n"
        f"FULL AUDIO ANALYSIS: {json.dumps(audio_output)}\n\n"
        "--- SESSION CONTEXT ---\n"
        "PREVIOUS USER MESSAGES:\n"
        + ("\n".join(f"- \"{r}\"" for r in recent_user) if recent_user else "None") + "\n\n"
        "RECENT AI REPLIES (Avoid repeating the same style, phrasing, or technique):\n"
        + ("\n".join(f"- \"{r}\"" for r in recent_ai) if recent_ai else "None") + "\n\n"
        f"RECENT TECHNIQUES USED (try to vary): {', '.join([t for t in recent_techs if t]) or 'None'}\n\n"
        f"SESSION STATE: {json.dumps(session_state)}\n"
        "-------------------\n\n"
    )

    # Step 4: Define the task and the required output format
    prompt += (
        "Based on all the information, choose the best decision for the next step. Return STRICT JSON only with the following keys:\n"
        "{\n"
        "  \"decision\": \"one of the allowed decisions from the list above\",\n"
        "  \"emotional_reflection\": \"A 1-2 sentence empathetic reflection that captures the user's content and emotional tone.\",\n"
        "  \"guidance\": \"A short bullet list for Brain2. Describe the tone to use, any must-have elements in the response, and what to avoid (e.g., repeating past phrases).\",\n"
        "  \"why_this_decision\": \"A brief, clear justification for your choice, explaining why it's the most therapeutically relevant action right now based on the user's state and history.\"\n"
        "}\n\n"
        "Return JSON only."
    )
    return prompt

def _api_failure_fallback(reason: str) -> Dict[str, str]:
    """
    A minimal fallback ONLY for when the LLM API call fails or returns malformed data.
    """
    _logger.warning("Brain1 API call or parsing failed: %s. Using minimal fallback.", reason)
    return {
        "decision": "empathy+follow_up",
        "emotional_reflection": "I hear you. Thank you for sharing that with me.",
        "guidance": "- Tone: Gentle and curious.\n- Must: Ask a simple open-ended question to understand more.\n- Avoid: Making assumptions.",
        "why_this_decision": "This is a fallback response due to a system error. The goal is to keep the conversation open and supportive without making a complex therapeutic move."
    }

def call_brain1(audio_output: Dict[str, Any], session_history: List[Dict[str, Any]], technique_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Primary interface for Brain1.
    """
    try:
        # *** MODIFIED: Safety flag override now triggers safety_crisis+task ***
        if audio_output.get("safety_flag", False):
            _logger.warning("Safety flag detected from audio processing. Overriding LLM to trigger safety_crisis+task protocol.")
            return {
                "decision": "safety_crisis+task",
                "emotional_reflection": "I hear the pain in your voice and I'm very concerned by what you said. Please know you're not alone and help is available.",
                "guidance": (
                    "- Tone: Direct, calm, and extremely compassionate.\n"
                    "- Must: First, immediately provide crisis resources (e.g., crisis hotline number like 988 in the US).\n"
                    "- Must: After providing resources, gently guide the user through a simple grounding task, like the 5-4-3-2-1 technique, to help them connect with the present moment.\n"
                    "- Must: Ensure the tone is non-blocking and invites the user to continue the conversation after the grounding exercise if they feel up to it.\n"
                    "- Avoid: Any other therapeutic techniques, complex questions, or analysis."
                ),
                "why_this_decision": "A safety flag was detected, indicating a potential crisis. The immediate priority is to ensure user safety by providing direct resources and then offering a grounding task to help with immediate overwhelming feelings."
            }

        prompt = _build_prompt(audio_output or {}, session_history or [], technique_history or [])
        raw = vertex_generate(prompt, max_output_chars=2048, temperature=0.4)

        if not raw:
            return _api_failure_fallback("Vertex returned an empty response")

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            _logger.warning("Brain1 response did not contain valid JSON structure. Raw: %s", raw)
            return _api_failure_fallback("Could not find JSON in LLM response")

        json_str = m.group(0)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            _logger.warning("Brain1 JSON parsing failed: %s. Raw JSON string: %s", e, json_str)
            repaired = json_str.replace("'", '"')
            # *** MODIFIED: Corrected regex for trailing commas ***
            repaired = re.sub(r",\s*([}```])", r"\1", repaired)
            try:
                parsed = json.loads(repaired)
            except Exception as e2:
                _logger.warning("Brain1 JSON repair attempt failed: %s", e2)
                return _api_failure_fallback(f"JSONDecodeError: {e}; repair failed: {e2}")

        decision = parsed.get("decision")
        if decision not in ALLOWED_COMBINATIONS:
            _logger.warning("Brain1 returned an invalid decision '%s'.", decision)
            return _api_failure_fallback(f"Invalid decision '{decision}' received")

        emotional_reflection = parsed.get("emotional_reflection", "I hear what you're saying.")
        guidance = parsed.get("guidance", "Proceed with a gentle and supportive tone.")
        why_this_decision = parsed.get("why_this_decision", "No justification provided by the model.")

        return {
            "decision": decision,
            "emotional_reflection": emotional_reflection,
            "guidance": guidance,
            "why_this_decision": why_this_decision
        }

    except Exception as e:
        _logger.exception("An unexpected error occurred in Brain1. Returning API failure fallback.")
        return _api_failure_fallback(f"Exception: {str(e)}")