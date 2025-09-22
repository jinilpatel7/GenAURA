# HealAura: An Empathetic AI Mental Wellness Companion

**HealAura** is an AI-powered, confidential, and empathetic mental wellness solution designed to support and guide youth in overcoming stigma and accessing help.  
It combines principles of Cognitive Behavioral Therapy (CBT), metacognition, and multimodal emotion detection to deliver personalized, therapeutic-like interactions.

The system goes beyond typical chatbots by analyzing not just *what* users say, but also *how* they say it—detecting emotions from vocal tone and speech patterns. Based on this deep understanding, it adapts its responses to provide calming guidance, motivational reinforcement, or gentle cognitive reframing.

---

## ✨ Key Features

### 🎙️ Feature 1: The Real-Time Conversational Therapist
- **Multi-Lingual Support:** English 🇬🇧 & Hindi 🇮🇳  
- **Multimodal Emotion Detection:** Vocal tone analysis + text sentiment fusion.  
- **Adaptive Dual-Brain Dialogue:** Brain1 decides next action, Brain2 crafts empathetic reply.  
- **In-the-Moment Guided Exercises:** Breathing, grounding, stress relief.  
- **Session Summaries:** Stored in the user’s private dashboard.  
🎥 [Feature 1 Demo](https://drive.google.com/file/d/1a3dcRTOVvzfd-44exAwq15EljmViffu8/view?usp=drive_link)

---

### 📊 Feature 2: The Wellness Tracker
- **Daily Logging:** Hourly reflections + structured daily entries (sleep, food, activities).  
- **AI-Powered Dashboard:** Narrative insights, challenges, areas of growth.  
- **Personalized Weekly Goals** with rationale.  
- **Visualization:** Correlation charts, sentiment journey, heatmaps, emotion pie chart, word clouds.  
🎥 [Feature 2 Demo](https://drive.google.com/file/d/18kYT3MaTz6DrYmpevqjcPvlKyL4gC36/view?usp=drive_link)

---

## 💡 Why HealAura is Different
- **🤫 Truly Anonymous:** No email/phone required, reducing stigma.  
- **🧠 Voice + Emotion Fusion:** Understands *how* you feel, not just what you type.  
- **🚀 Dual-Brain Architecture:** Brain1 = Strategist, Brain2 = Empathetic Speaker.  
- **📖 Transparency:** Every AI suggestion includes the *psychology behind it*.  
- **🛟 Guided Crisis Intervention:** Not just a hotline, but live empathetic regulation.  
- **📊 Beyond Conversation:** Integrated with a proactive Wellness Tracker & Insights.  

---

## 🛠️ Technology Stack

| Category         | Tech                                                                                   |
|------------------|----------------------------------------------------------------------------------------|
| **Frontend**     | HTML5, CSS3, JavaScript (ES6+), Chart.js, D3.js                                       |
| **Backend**      | Python 3, FastAPI, Uvicorn                                                             |
| **Audio**        | ffmpeg, librosa, scipy, numpy (fusion of audio & text emotion)                        |
| **Database**     | Google Cloud Firestore                                                                 |
| **AI Services**  | Vertex AI (Brain1 & Brain2 + Insights), Cloud Speech-to-Text, Cloud Text-to-Speech      |
| **DevOps/Tools** | Docker, GitHub, Passlib (bcrypt), dotenv, logging                                      |

---

## 🏛️ System Architecture

HealAura is built on a modern, scalable, and stateless architecture using Google Cloud Platform.

<p align="center">
  <img src="https://raw.githubusercontent.com/jinilpatel7/GenAURA/main/assets/GenAURA_Architecture.jpg" alt="HealAura System Architecture" width="800"/>
</p>

- **Frontend SPA:** Captures audio, renders empathy/chat UI, dashboard visualizations.  
- **Backend (FastAPI):** Stateless orchestrator exposing REST APIs.  
- **Firestore (DB):** Persists users, logs, transcripts, summaries, cache.  
- **AI Services:**  
  - Vertex AI LLMs (strategic reasoning, structured response, summaries, insights).  
  - Cloud STT → transcript; TTS → soothing AI voice.  
- **Audio Processing:** ffmpeg + librosa → standard 16kHz WAV + audio features.  

---

## ⚙️ Setup & Installation

### Prerequisites
- **Google Cloud Platform project** with APIs enabled:
  - Firestore
  - Vertex AI
  - Cloud Speech-to-Text
  - Cloud Text-to-Speech
- **Service Account Key JSON (`key.json`)** with roles for those services.
- **Python 3.10+** (for local run) or Docker 🐳
- **FFmpeg** (auto-installed in Docker image; required locally if running without Docker).

---

### Clone Repo
```bash
git clone https://github.com/jinilpatel7/GenAURA.git
cd GenAURA
```
## Run Locally
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Linux or macOS: export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/key.json"
Windows: set GOOGLE_APPLICATION_CREDENTIALS=%cd%\gcp_key.json
uvicorn empathetic-ai-therapist.app.main:app --reload --port 8000
```

## Run with Docker (Recommended)

```bash
docker pull jinilpatel/healaura:latest

docker run -it -p 8080:8080 \
  -v $(pwd)/key.json:/app/key.json \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/key.json" \
  -e GCP_PROJECT="your-gcp-project-id" \
  -e GCP_LOCATION="us-central1" \
  jinilpatel/healaura:latest
```
⚠️ Note: Replace your-gcp-project-id with your own.
Use the image name jinilpatel/healaura (do not replace the DockerHub username unless forking).

### **Step 3: Access the Application**
Once the server is running (either via Docker or locally), open your web browser and go to:
**`http://localhost:8000`**



## 💖 Contributing

We welcome contributions! Please feel free to submit a pull request or open an issue to discuss your ideas.
