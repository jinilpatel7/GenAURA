# empathetic-ai-therapist/app/wellness.py
import logging
from datetime import datetime
import json
import re
from collections import defaultdict
import numpy as np

from fastapi import APIRouter, Depends, Form, HTTPException, Body, Query
from google.cloud import firestore
import pytz

from .gcp_clients import get_firestore_client, vertex_generate

_logger = logging.getLogger(__name__)
router = APIRouter()

FIRESTORE_USERS_COLLECTION = "users"
FIRESTORE_LOGS_SUBCOLLECTION = "wellness_logs"
FIRESTORE_SUMMARIES_SUBCOLLECTION = "session_summaries"
LOCAL_TIMEZONE = pytz.timezone('Asia/Kolkata')

def get_current_user_id(user_id: str = Form(...)):
    if not user_id: raise HTTPException(status_code=400, detail="user_id is required.")
    return user_id

def get_current_user_id_from_body(payload: dict = Body(...)):
    user_id = payload.get("user_id")
    if not user_id: raise HTTPException(status_code=400, detail="user_id is required in the request body.")
    return user_id

def _extract_json_from_raw(raw: str) -> dict:
    if not raw: return {}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match: return {}
    try: return json.loads(match.group(0))
    except json.JSONDecodeError: return {}

async def _analyze_text_for_wellness(text: str) -> dict:
    prompt = (
        "You are an expert emotion and topic analysis AI. Analyze the following user journal entry. "
        "Respond ONLY with a single JSON object with these exact keys:\n"
        "- 'primary_emotion': A single dominant emotion from [neutral, happy, sad, angry, fear, anxious, stressed, content].\n"
        "- 'intensity': The intensity of the emotion from 1 (low) to 5 (high).\n"
        "- 'key_topics': A list of 1-3 key topics or themes mentioned (e.g., ['work', 'family', 'deadline pressure']).\n"
        "- 'sentiment': A float from -1.0 (very negative) to 1.0 (very positive).\n\n"
        f"User text: \"{text}\"\n\n"
        "JSON response only."
    )
    raw_response = vertex_generate(prompt, temperature=0.2, max_output_chars=1024)
    return _extract_json_from_raw(raw_response)

async def generate_session_summary(user_id: str, session_history: list, session_start_iso: str):
    if not user_id or not session_history: return
    _logger.info("Generating session summary for user %s", user_id)
    full_transcript = "\n".join(f"{turn['who']}: {turn['text']}" for turn in session_history)
    prompt = (
        "You are a clinical psychologist AI. Based on the following therapy session transcript, provide a concise summary. Respond ONLY with a single JSON object with these exact keys:\n"
        "- 'title': A short, 4-6 word title for the session.\n"
        "- 'key_emotions_discussed': A list of primary emotions the user expressed.\n"
        "- 'key_topics_covered': A list of the main topics discussed.\n"
        "- 'ai_suggestion_summary': A brief, 1-2 sentence summary of the main coping strategy the AI suggested.\n"
        "- 'overall_summary': A 3-4 sentence narrative summary of the session's flow.\n\n"
        f"Transcript:\n---\n{full_transcript}\n---\n\n"
        "JSON response only."
    )
    try:
        raw_summary = vertex_generate(prompt, temperature=0.3, max_output_chars=2048)
        summary_data = _extract_json_from_raw(raw_summary)
        if not summary_data: return
        session_start_dt = datetime.fromisoformat(session_start_iso.replace("Z", "+00:00"))
        summary_data["session_start_utc"] = session_start_dt.isoformat()
        client = get_firestore_client()
        if not client: return
        doc_id = session_start_dt.astimezone(LOCAL_TIMEZONE).strftime('%Y-%m-%d_%H-%M-%S')
        doc_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(user_id).collection(FIRESTORE_SUMMARIES_SUBCOLLECTION).document(doc_id)
        doc_ref.set(summary_data)
        _logger.info("Successfully saved session summary for user %s.", user_id)
    except Exception as e:
        _logger.exception("Error during session summary generation for user %s: %s", user_id, e)

@router.post("/log_daily_entry")
async def log_daily_entry(text: str = Form(...), hour: int = Form(...), date_str: str = Form(...), user_id: str = Depends(get_current_user_id)):
    if not (0 <= hour <= 23): raise HTTPException(status_code=400, detail="Invalid hour.")
    analysis = await _analyze_text_for_wellness(text)
    if not analysis: raise HTTPException(status_code=500, detail="Failed to analyze text entry.")
    log_data = {"type": "hourly_text", "timestamp_utc": datetime.utcnow().isoformat() + "Z", "log_date_local": date_str, "log_hour_local": hour, "raw_text": text, "analysis": analysis}
    client = get_firestore_client()
    if not client: raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        doc_id = f"{date_str}_{hour:02d}"
        doc_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(user_id).collection(FIRESTORE_LOGS_SUBCOLLECTION).document(doc_id)
        doc_ref.set(log_data)
        # MODIFIED: Return the newly created log data object to the frontend for a local update.
        return {"status": "success", "data": log_data}
    except Exception as e:
        _logger.exception("Failed to save daily log for user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Failed to save log entry.")

@router.post("/log_structured_entry")
async def log_structured_entry(
    date_str: str = Form(...),
    sleep_quality: str = Form(None),
    social_interactions: list[str] = Form([]),
    activities: list[str] = Form([]),
    food_breakfast: str = Form(None),
    food_lunch: str = Form(None),
    food_dinner: str = Form(None),
    user_id: str = Depends(get_current_user_id)
):
    log_data = {
        "type": "structured",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "log_date_local": date_str,
        "data": {
            "sleep_quality": sleep_quality,
            "social_interactions": social_interactions,
            "activities": activities,
            "food_intake": {
                "breakfast": food_breakfast,
                "lunch": food_lunch,
                "dinner": food_dinner
            }
        }
    }
    client = get_firestore_client()
    if not client: raise HTTPException(status_code=500, detail="Database connection failed.")
    try:
        doc_id = f"{date_str}_structured"
        doc_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(user_id).collection(FIRESTORE_LOGS_SUBCOLLECTION).document(doc_id)
        doc_ref.set(log_data)
        return {"status": "success", "message": "Daily reflection saved."}
    except Exception as e:
        _logger.exception("Failed to save structured log for user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Failed to save structured entry.")

def _aggregate_log_data(logs: list) -> dict:
    text_logs = [log for log in logs if log.get('type') == 'hourly_text' and 'analysis' in log]
    structured_logs = {log['log_date_local']: log['data'] for log in logs if log.get('type') == 'structured'}

    daily_sentiment = defaultdict(lambda: {'sum': 0, 'count': 0})
    emotion_counts, word_freq = defaultdict(int), defaultdict(int)
    stop_words = set(["i", "me", "my", "myself", "we", "our", "a", "an", "the", "and", "but", "if", "or", "because", "as", "of", "at", "by", "for", "with", "about", "to", "from", "in", "out", "on", "off", "so", "then", "very", "was", "is", "are", "am", "had", "have", "has", "it", "that", "this", "he", "she", "they", "went", "got", "with", "and", "the", "for", "in", "of", "to"])

    for log in text_logs:
        analysis = log.get('analysis', {})
        sentiment = analysis.get('sentiment')
        if sentiment is None: continue
        
        date_str = log.get('log_date_local')
        daily_sentiment[date_str]['sum'] += sentiment
        daily_sentiment[date_str]['count'] += 1
        
        emotion = analysis.get('primary_emotion')
        if emotion: emotion_counts[str(emotion)] += 1

        text = log.get('raw_text', '').lower()
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            if word not in stop_words and not word.isdigit(): word_freq[word] += 1
    
    final_daily_sentiment = {d: v['sum']/v['count'] for d, v in daily_sentiment.items() if v['count'] > 0}

    correlation_data = defaultdict(lambda: defaultdict(list))
    food_map = {"Healthy": 1, "Semi": 0, "Junk": -1}

    for date, data in structured_logs.items():
        if date not in final_daily_sentiment: continue
        sentiment = final_daily_sentiment[date]

        if data.get('sleep_quality'):
            correlation_data['sleep_quality'][data['sleep_quality']].append(sentiment)
        
        for activity in data.get('activities', []):
            correlation_data['activity'][activity].append(sentiment)
            
        for social in data.get('social_interactions', []):
            correlation_data['social'][social].append(sentiment)

        food_score = 0
        food_count = 0
        food_intake = data.get('food_intake', {})
        for meal_type in ['breakfast', 'lunch', 'dinner']:
            quality = food_intake.get(meal_type)
            if quality in food_map:
                food_score += food_map[quality]
                food_count += 1
        if food_count > 0:
            avg_food_score = food_score / food_count
            correlation_data['food_quality_vs_sentiment']['scores'].append(avg_food_score)
            correlation_data['food_quality_vs_sentiment']['sentiments'].append(sentiment)
            
    final_correlations = {}
    for category, items in correlation_data.items():
        if category == 'food_quality_vs_sentiment':
            final_correlations[category] = items
            continue
        final_correlations[category] = {item: np.mean(sentiments) for item, sentiments in items.items()}

    top_words_list = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
    word_cloud_data = [{"text": word, "value": freq} for word, freq in top_words_list]

    return {
        "daily_sentiment": final_daily_sentiment,
        "correlation_metrics": final_correlations,
        "emotion_counts": dict(emotion_counts),
        "word_cloud_data": word_cloud_data,
    }

async def _generate_wellness_insights(aggregated_data: dict) -> dict:
    prompt_context = json.dumps(aggregated_data, indent=2)
    prompt = (
        "You are a compassionate, insightful, and highly intelligent wellness coach AI. Based on the user's wellness data summary, generate personalized insights. Your tone should be gentle, encouraging, and non-judgmental. Be detailed and explain your reasoning. "
        "Respond ONLY with a single JSON object with these exact keys:\n"
        "- 'causal_inference': A 'We've Noticed...' section. Find the most interesting and impactful correlation in the data. Describe this potential link in 2-3 sentences, phrasing it as a gentle observation, not a proven fact (e.g., 'It seems there might be a connection between your sleep and mood. We've noticed that on days you reported 'Good' sleep, your average sentiment was significantly higher, suggesting that restful nights could be a powerful tool for your well-being.').\n"
        "- 'positive_pattern': Identify one clear positive pattern. Describe it in detail (2-3 sentences), explaining why it's beneficial and offering specific encouragement (e.g., 'It's wonderful to see that you consistently engage in your hobbies. This pattern is a fantastic form of self-care, as it provides a creative outlet and a way to decompress. Keep nurturing this positive habit!').\n"
        "- 'challenge_pattern': Identify one potential challenge or area for awareness. Describe it kindly and non-judgmentally (2-3 sentences), framing it as an opportunity for self-discovery (e.g., 'There's a gentle pattern suggesting that days logged as 'Work-heavy' tend to correlate with lower sentiment scores. This isn't about work being 'bad,' but it might be an invitation to explore how work-related stress impacts your energy and what small changes could bring more balance.').\n"
        "- 'area_for_growth': Based on the challenge_pattern, suggest a specific area for reflection in 2-3 sentences. This should be a 'thinking' task, not an 'action' task (e.g., 'A potential area for growth could be in exploring your relationship with work-life balance. You might reflect on what a 'balanced' day feels like to you, or what boundaries could help protect your energy during busy periods.').\n"
        "- 'actionable_suggestion': Provide a single, concrete, and easy-to-do actionable tip based on the data analysis. Explain *why* this tip could be helpful in a separate sentence. (e.g., 'Based on the positive impact of social connection, perhaps you could try scheduling one short, 15-minute call with a friend this week. This small step can reinforce those positive feelings of connection without feeling overwhelming.').\n"
        "- 'weekly_goals': A list of 2-3 specific, achievable, and personalized goals. For each goal, provide a 'goal' and a 'rationale'. The rationale should be a short, 1-sentence explanation of why it might be beneficial. Structure it as a list of JSON objects: `[{\"goal\": \"...\", \"rationale\": \"...\"}]`.\n\n"
        f"User Data Summary:\n---\n{prompt_context}\n---\n\n"
        "JSON response only. Ensure the output is a valid JSON object with rich, detailed content."
    )
    raw_response = vertex_generate(prompt, temperature=0.6, max_output_chars=4096)
    insights = _extract_json_from_raw(raw_response)
    if not insights or 'positive_pattern' not in insights:
        return {
            "causal_inference": "As you log more daily reflections, we'll be able to spot interesting connections in your wellness journey.",
            "positive_pattern": "Keep logging to discover your emotional patterns. Consistency is the first step and you're doing great.",
            "challenge_pattern": "Patterns will become clearer with more entries. Be patient and kind with yourself.",
            "area_for_growth": "Building a consistent habit of reflection is a powerful tool for self-awareness.",
            "actionable_suggestion": "Try to fill out your Daily Reflection for at least 3 days this week.",
            "weekly_goals": [
                {"goal": "Log your hourly thoughts at least once a day.", "rationale": "This helps build a habit of checking in with yourself."},
                {"goal": "Complete the Daily Reflection on a day you feel particularly good or bad.", "rationale": "This can help highlight what factors contribute to your emotional state."}
            ]
        }
    return insights

async def _regenerate_and_cache_wellness_data(user_id: str, client: firestore.Client) -> dict:
    _logger.info("Regenerating wellness cache for user: %s", user_id)
    logs_query = client.collection(FIRESTORE_USERS_COLLECTION).document(user_id).collection(FIRESTORE_LOGS_SUBCOLLECTION).order_by("timestamp_utc", direction=firestore.Query.DESCENDING).limit(90 * 25)
    logs = [doc.to_dict() for doc in logs_query.stream()]
    summaries_query = client.collection(FIRESTORE_USERS_COLLECTION).document(user_id).collection(FIRESTORE_SUMMARIES_SUBCOLLECTION).order_by("session_start_utc", direction=firestore.Query.DESCENDING).limit(30)
    summaries = [doc.to_dict() for doc in summaries_query.stream()]
    
    aggregated_data = _aggregate_log_data(logs)
    insights = await _generate_wellness_insights(aggregated_data)
    
    full_data_payload = {"raw_logs": logs, "summaries": summaries, "aggregated_data": aggregated_data, "insights": insights}
    
    user_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(user_id)
    user_ref.update({"wellness_cache": {"last_updated_utc": datetime.utcnow().isoformat() + "Z", "data": json.loads(json.dumps(full_data_payload, default=str))}})
    
    return full_data_payload

@router.post("/get_wellness_data")
async def get_wellness_data(user_id: str = Depends(get_current_user_id)):
    client = get_firestore_client()
    if not client: raise HTTPException(status_code=500, detail="Database connection failed.")
    
    try:
        user_ref = client.collection(FIRESTORE_USERS_COLLECTION).document(user_id)
        user_doc = user_ref.get()
        if not user_doc.exists: raise HTTPException(status_code=404, detail="User not found.")
        
        user_data = user_doc.to_dict()
        cache = user_data.get("wellness_cache", {})
        cached_data = cache.get("data")
        last_updated_str = cache.get("last_updated_utc")

        if cached_data and last_updated_str:
            last_updated_dt = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
            
            latest_log_query = user_ref.collection(FIRESTORE_LOGS_SUBCOLLECTION).order_by("timestamp_utc", direction=firestore.Query.DESCENDING).limit(1)
            latest_log = next(latest_log_query.stream(), None)
            
            latest_summary_query = user_ref.collection(FIRESTORE_SUMMARIES_SUBCOLLECTION).order_by("session_start_utc", direction=firestore.Query.DESCENDING).limit(1)
            latest_summary = next(latest_summary_query.stream(), None)

            is_stale = False
            if latest_log and datetime.fromisoformat(latest_log.to_dict()['timestamp_utc'].replace("Z", "+00:00")) > last_updated_dt:
                is_stale = True
            if not is_stale and latest_summary and datetime.fromisoformat(latest_summary.to_dict()['session_start_utc'].replace("Z", "+00:00")) > last_updated_dt:
                is_stale = True

            if not is_stale:
                _logger.info("Serving fresh wellness data from cache for user: %s", user_id)
                return cached_data
        
        fresh_data = await _regenerate_and_cache_wellness_data(user_id, client)
        return fresh_data

    except Exception as e:
        _logger.exception("Failed to fetch wellness data for user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Could not retrieve wellness data.")

@router.post("/summarize_day_logs")
async def summarize_day_logs(payload: dict = Body(...), user_id: str = Depends(get_current_user_id_from_body)):
    logs_for_day = payload.get("logs", [])
    if not logs_for_day: return {"summary": "No logs were recorded for this day."}
    log_texts = [f"- At {log['log_hour_local']}:00, I felt '{log.get('analysis',{}).get('primary_emotion','unknown')}' about: {log['raw_text']}" for log in sorted(logs_for_day, key=lambda x: x['log_hour_local'])]
    full_text = "\n".join(log_texts)
    prompt = (
        "You are a compassionate AI assistant. The user has provided their journal entries for a single day. "
        "Create a brief, gentle, and human-readable summary of their day based on these logs. "
        "Start with a general emotional overview, then mention key topics. Do not use markdown.\n\n"
        f"User's logs for the day:\n{full_text}\n\n"
        "Summary:"
    )
    try:
        summary = vertex_generate(prompt, temperature=0.4, max_output_chars=1024)
        return {"summary": summary or "Could not generate a summary for this day."}
    except Exception as e:
        _logger.exception("Failed to summarize day logs for user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Failed to generate summary.")