let currentUserId = null, sessionId = null, mediaRecorder = null, wellnessData = null, activeTask = null, timerInterval = null, timerRemaining = 0;
let audioChunks = [], charts = {};
const dom = {
    authContainer: document.getElementById('authContainer'), appContainer: document.getElementById('appContainer'),
    loginForm: document.getElementById('loginForm'), signupForm: document.getElementById('signupForm'),
    showSignupLink: document.getElementById('showSignupLink'), showLoginLink: document.getElementById('showLoginLink'),
    logoutBtn: document.getElementById('logoutBtn'),
    showSessionBtn: document.getElementById('showSessionBtn'), showWellnessBtn: document.getElementById('showWellnessBtn'),
    therapyContainer: document.getElementById('therapySessionContainer'), wellnessContainer: document.getElementById('wellnessContainer'),
    sessionIdEl: document.getElementById('sessionId'), endSessionBtn: document.getElementById('endSession'),
};

function showAlert(title, message, type = 'info') {
    const existingModal = document.querySelector('.modal-overlay');
    if (existingModal) existingModal.remove();
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    const typeColor = type === 'error' ? 'var(--danger)' : type === 'success' ? 'var(--success)' : 'var(--accent-dark)';
    modal.innerHTML = `
        <div class="modal-box">
            <h3 style="color:${typeColor};">${title}</h3>
            <p>${message}</p>
            <div class="modal-actions" style="justify-content: center;">
                <button class="primary" onclick="this.closest('.modal-overlay').remove()">OK</button>
            </div>
        </div>`;
    document.body.appendChild(modal);
}

function showLoader(button, isLoading) {
    if(!button) return;
    button.classList.toggle('loading', isLoading);
    button.disabled = isLoading;
}

function toggleInterpret(el) {
    const textEl = el.nextElementSibling;
    textEl.classList.toggle('hidden');
    el.textContent = textEl.classList.contains('hidden') ? 'How to Interpret ▼' : 'Hide Interpretation ▲';
}

function appendTurn(who, data, meta = null, decision = '') {
    const turnEl = document.createElement('div');
    turnEl.className = `turn ${who}`;
    const avatar = document.createElement('div');
    avatar.className = `avatar avatar-${who}`;
    avatar.textContent = who === 'ai' ? 'AI' : 'You';
    const msgContent = document.createElement('div');
    msgContent.className = 'msg-content';
    const speaker = document.createElement('div');
    speaker.className = 'speaker';
    speaker.textContent = who === 'ai' ? 'AI Therapist' : 'You';
    const msgBubble = document.createElement('div');
    msgBubble.className = 'msg-bubble';
    const replyData = data.reply_structured || {};
    if (who === 'ai' && Array.isArray(replyData.response_parts)) {
        replyData.response_parts.forEach(part => {
            const partEl = document.createElement('div');
            partEl.className = `response-${part.type || 'plain'}`;
            partEl.innerText = part.text;
            msgBubble.appendChild(partEl);
        });
        if (replyData.task_meta?.detail) {
            const instructionsEl = document.createElement('div');
            instructionsEl.className = 'task-instructions';
            const ol = document.createElement('ol');
            const steps = replyData.task_meta.detail.replace(/\d+\.\s*/g, '').split(/;\s*|\.\s*/).filter(s => s.trim());
            steps.forEach(step => { const li = document.createElement('li'); li.textContent = step; ol.appendChild(li); });
            instructionsEl.appendChild(ol);
            msgBubble.appendChild(instructionsEl);
        }
    } else { msgBubble.innerText = data.text || "(no text)"; }
    msgContent.appendChild(speaker);
    msgContent.appendChild(msgBubble);
    if (meta) { const metaEl = document.createElement('div'); metaEl.className = 'meta'; metaEl.textContent = meta; msgContent.appendChild(metaEl); }
    if (who === 'ai' && replyData.psychology_behind_it) {
        const psychEl = document.createElement('div');
        const decisionType = (decision || '').split('+')[1] || (decision || '').split('+')[0];
        psychEl.className = `psychology-note psychology-note--${decisionType}`;
        psychEl.innerText = replyData.psychology_behind_it;
        msgContent.appendChild(psychEl);
    }
    turnEl.appendChild(avatar);
    turnEl.appendChild(msgContent);
    dom.conversation.appendChild(turnEl);
    dom.conversation.scrollTop = dom.conversation.scrollHeight;
    const startTask = () => {
        const timerCard = (replyData.display_cards || []).find(c => c.type === 'timer');
        if (timerCard) {
            const task = { task_id: replyData.task_meta?.task_id || `task_${Date.now()}`, type: replyData.task_meta?.type || 'exercise', detail: replyData.task_meta?.detail || 'Follow spoken instructions.', duration_sec: timerCard.duration_sec };
            startCountdown(timerCard.duration_sec, task);
        }
    };
    if (who === 'ai' && data.tts_b64) {
        try { const audioBlob = new Blob([Uint8Array.from(atob(data.tts_b64), c => c.charCodeAt(0))], { type: 'audio/mpeg' }); const audioUrl = URL.createObjectURL(audioBlob); const audio = new Audio(audioUrl); audio.play().catch(e => { console.warn("TTS playback failed", e); startTask(); }); audio.onended = () => { URL.revokeObjectURL(audioUrl); startTask(); }; } catch (e) { console.warn("TTS processing failed", e); startTask(); }
    } else { startTask(); }
}

function startCountdown(durationSec, task) {
    clearActiveTask();
    timerRemaining = parseInt(durationSec || 0);
    activeTask = task;
    dom.taskControls.classList.remove('hidden');
    if (timerRemaining <= 0) { dom.markDoneBtn.disabled = false; return; }
    dom.timerBox.classList.remove('hidden');
    dom.timerValue.textContent = `${timerRemaining}s`;
    dom.markDoneBtn.disabled = true;
    timerInterval = setInterval(() => {
        timerRemaining -= 1;
        dom.timerValue.textContent = `${timerRemaining}s`;
        if (timerRemaining <= 0) { clearInterval(timerInterval); timerInterval = null; dom.timerValue.textContent = 'Done!'; dom.markDoneBtn.disabled = false; }
    }, 1000);
}

function clearActiveTask() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = null;
    if(dom.taskControls) dom.taskControls.classList.add('hidden');
    if(dom.timerBox) dom.timerBox.classList.add('hidden');
    activeTask = null;
    if(dom.markDoneBtn) dom.markDoneBtn.disabled = true;
}

function updateStatus(status, processing=false){
    if(!dom.recStatus || !dom.statusIndicator) return;
    dom.recStatus.textContent = status;
    dom.statusIndicator.className = 'status-indicator';
    if(processing) dom.statusIndicator.classList.add('status-processing');
    else if(status.includes('Recording')) dom.statusIndicator.classList.add('status-listening');
    else dom.statusIndicator.classList.add('status-idle');
}

document.addEventListener('DOMContentLoaded', () => {
    // Populate Auth Container
    dom.authContainer.innerHTML = `<div class="auth-container"><form id="loginForm" class="auth-form"><h2>Welcome Back to HealAura</h2><input type="text" id="loginUsername" name="username" placeholder="Username" required><input type="password" id="loginPassword" name="password" placeholder="Password" required><button type="submit" class="primary"><span class="btn-text">Login</span><div class="loader"></div></button><div class="form-footer">Don't have an account? <a id="showSignupLink">Create one</a></div></form><form id="signupForm" class="auth-form hidden"><h2>Create Your HealAura Account</h2><div style="border:1px solid #f0ad4e; padding:12px; border-radius:6px; background:#fff7e6;">
  <strong>Important:</strong>
  <p style="margin:6px 0 0;">
    This is an anonymous account. We do not collect email or other credentials. <strong>Save your username and password now</strong>; they cannot be recovered later.
  </p>
</div>
<input type="text" id="signupUsername" name="username" placeholder="Username" required><input type="password" id="signupPassword" name="password" placeholder="Password (min 6 chars)" required><input type="password" id="signupConfirmPassword" name="confirm_password" placeholder="Confirm Password" required><button type="submit" class="primary"><span class="btn-text">Create Account</span><div class="loader"></div></button><div class="form-footer">Already have an account? <a id="showLoginLink">Login</a></div></form></div>`;
    // Populate Therapy Container
    dom.therapyContainer.innerHTML = `<div class="session-setup" id="sessionSetup"><div><label for="languageSelect" class="small">Language:</label><select id="languageSelect"><option value="en-US" selected>English</option><option value="hi-IN">Hindi</option></select></div><div class="gender-options"><span>Your Gender:</span><div class="gender-btn"><input type="radio" name="userGender" value="female" id="genderF" checked><label for="genderF">Female</label></div><div class="gender-btn"><input type="radio" name="userGender" value="male" id="genderM"><label for="genderM">Male</label></div><div class="gender-btn"><input type="radio" name="userGender" value="other" id="genderO"><label for="genderO">Other</label></div></div><button id="startSession" class="primary" style="margin-left: auto;"><span class="btn-text">Start New Session</span><div class="loader"></div></button></div><div class="recording hidden" id="recordingControls"><button id="startRec">Start Recording</button><button id="stopRec" disabled class="danger">Stop Recording</button><button id="submitRec" disabled class="primary"><span class="btn-text">Submit</span><div class="loader"></div></button><div style="margin-left: auto; display: flex; align-items: center; gap: 6px;"><span class="status-indicator status-idle" id="statusIndicator"></span><span id="recStatus" class="small">Ready</span></div></div><div id="conversation"></div><div class="taskbar hidden" id="taskControls"><button id="markDone" class="success" disabled><span class="btn-text">I'm Done</span><div class="loader"></div></button><div id="timerBox" class="hidden"><span class="chip">Timer</span><span class="timer-value" id="timerValue">60s</span></div></div>`;
    // Populate Wellness Container
    dom.wellnessContainer.innerHTML = `
        <h2 style="font-size:26px; margin-bottom:24px; color:var(--text-primary);">Your Wellness Dashboard</h2>
        <div class="card"><div class="card-header"><h3>Daily Logging</h3></div><div class="card-content">
            <h4>Part 1: Quick Thoughts</h4><p class="small" style="margin-bottom:16px;">How are you feeling? Click an hour to log your thoughts and emotions. Logged hours are disabled.</p><div class="hourly-logger" id="hourlyLogger"></div>
            <hr style="margin: 28px 0; border: none; border-top: 1px solid var(--border);">
            <h4>Part 2: Daily Reflection</h4><p class="small" style="margin-bottom:16px;">Fill this out once per day to find deeper patterns.</p><form id="dailyReflectionForm" class="daily-reflection-form"></form>
        </div></div>
        <div class="card"><div class="card-header"><h3>This Week's Goals</h3></div><div class="card-content goals-grid" id="goalsContainer"></div></div>
        <div class="card"><div class="card-header"><h3>AI-Powered Insights</h3></div><div class="card-content"><div class="insights-grid" id="insightsContainer"></div></div></div>
        <div class="charts-grid">
            <div class="card"><div class="card-header"><h3>Sleep Quality vs. Sentiment</h3></div><div class="card-content"><canvas id="sleepChart"></canvas><a class="interpret-btn" onclick="toggleInterpret(this)">How to Interpret ▼</a><div class="interpret-text hidden">This chart shows your average mood based on sleep quality. A green bar means the activity is linked to a positive mood on average, while a red bar suggests a link to a negative mood. The height of the bar shows the strength of this connection.</div></div></div>
            <div class="card"><div class="card-header"><h3>Activity vs. Sentiment</h3></div><div class="card-content"><canvas id="activityChart"></canvas><a class="interpret-btn" onclick="toggleInterpret(this)">How to Interpret ▼</a><div class="interpret-text hidden">This compares how different activities correlate with your mood. A green bar means the activity is linked to a positive mood on average, while a red bar suggests a link to a negative mood. The height of the bar shows the strength of this connection.</div></div></div>
        </div>
        <div class="charts-grid">
            <div class="card"><div class="card-header"><h3>Social vs. Sentiment</h3></div><div class="card-content"><canvas id="socialChart"></canvas><a class="interpret-btn" onclick="toggleInterpret(this)">How to Interpret ▼</a><div class="interpret-text hidden">This chart shows the connection between who you spend time with and your overall mood. A green bar means the interaction is linked to a positive mood on average, while a red bar suggests a link to a negative mood. The height of the bar shows the strength of this connection.</div></div></div>
            <div class="card"><div class="card-header"><h3>Sentiment Over Time</h3></div><div class="card-content"><canvas id="sentimentOverTimeChart"></canvas><a class="interpret-btn" onclick="toggleInterpret(this)">How to Interpret ▼</a><div class="interpret-text hidden">This line chart tracks your average daily mood. Upward trends indicate periods of positivity, while downward trends might highlight challenging times. It helps you see your emotional journey at a glance.</div></div></div>
        </div>
        <div class="charts-grid">
            <div class="card"><div class="card-header"><h3>Emotion Distribution</h3></div><div class="card-content"><canvas id="emotionDistributionChart"></canvas><a class="interpret-btn" onclick="toggleInterpret(this)">How to Interpret ▼</a><div class="interpret-text hidden">This chart shows the percentage of each emotion you've logged. It gives a high-level overview of your most frequent feelings. Are there any surprises here?</div></div></div>
            <div class="card"><div class="card-header"><h3>Topic Word Cloud</h3></div><div class="card-content"><div id="wordCloudContainer"></div><a class="interpret-btn" onclick="toggleInterpret(this)">How to Interpret ▼</a><div class="interpret-text hidden">The bigger the word, the more often it appears in your logs. This helps you quickly see the topics that are on your mind the most, offering clues about what drives your emotions.</div></div></div>
        </div>
        <div class="charts-grid">
            <div class="card"><div class="card-header"><h3>Daily Sentiment (Last 30 Days)</h3></div><div class="card-content"><div id="calendarHeatmap"></div><a class="interpret-btn" onclick="toggleInterpret(this)">How to Interpret ▼</a><div class="interpret-text hidden">Greener days are more positive, redder days are more negative. Click any day to get an AI-generated summary of your logs for that day.</div></div></div>
            <div class="card"><div class="card-header"><h3>Recent Session Summaries</h3></div><div class="card-content"><ul class="session-summary-list" id="sessionSummaryList">(No session summaries found)</ul></div></div>
        </div>
        `;

    // Re-assign all DOM elements
    Object.assign(dom, {
        loginForm: document.getElementById('loginForm'), signupForm: document.getElementById('signupForm'), showSignupLink: document.getElementById('showSignupLink'), showLoginLink: document.getElementById('showLoginLink'), sessionSetup: document.getElementById('sessionSetup'), languageSelect: document.getElementById('languageSelect'), startSessionBtn: document.getElementById('startSession'), recordingControls: document.getElementById('recordingControls'), startRecBtn: document.getElementById('startRec'), stopRecBtn: document.getElementById('stopRec'), submitRecBtn: document.getElementById('submitRec'), conversation: document.getElementById('conversation'), statusIndicator: document.getElementById('statusIndicator'), recStatus: document.getElementById('recStatus'), taskControls: document.getElementById('taskControls'), markDoneBtn: document.getElementById('markDone'), timerBox: document.getElementById('timerBox'), timerValue: document.getElementById('timerValue'), hourlyLogger: document.getElementById('hourlyLogger'), insightsContainer: document.getElementById('insightsContainer'), goalsContainer: document.getElementById('goalsContainer'), dailyReflectionForm: document.getElementById('dailyReflectionForm'), sessionSummaryList: document.getElementById('sessionSummaryList'), calendarHeatmap: document.getElementById('calendarHeatmap'), wordCloudContainer: document.getElementById('wordCloudContainer')
    });
    
    // Attach event listeners
    dom.showSignupLink.onclick = () => { dom.loginForm.classList.add('hidden'); dom.signupForm.classList.remove('hidden'); };
    dom.showLoginLink.onclick = () => { dom.signupForm.classList.add('hidden'); dom.loginForm.classList.remove('hidden'); };
    dom.signupForm.onsubmit = onSignup;
    dom.loginForm.onsubmit = onLogin;
    dom.logoutBtn.onclick = onLogout;
    dom.showSessionBtn.onclick = () => { dom.therapyContainer.classList.remove('hidden'); dom.wellnessContainer.classList.add('hidden'); dom.showSessionBtn.classList.add('primary'); dom.showWellnessBtn.classList.remove('primary'); };
    dom.showWellnessBtn.onclick = () => { dom.therapyContainer.classList.add('hidden'); dom.wellnessContainer.classList.remove('hidden'); dom.showWellnessBtn.classList.add('primary'); dom.showSessionBtn.classList.remove('primary'); if (!wellnessData) fetchAndRenderWellnessData(); };
    dom.startSessionBtn.onclick = onStartSession;
    dom.endSessionBtn.onclick = onEndSession;
    dom.startRecBtn.onclick = onStartRecording;
    dom.stopRecBtn.onclick = onStopRecording;
    dom.submitRecBtn.onclick = onSubmitRecording;
    dom.markDoneBtn.onclick = onTaskDone;
    dom.dailyReflectionForm.onsubmit = onSaveStructuredData;
});

// --- AUTH & SESSION FUNCTIONS ---
async function onSignup(e) { e.preventDefault(); const btn = dom.signupForm.querySelector('button'); showLoader(btn, true); try { const form = new FormData(dom.signupForm); const res = await fetch('/signup', { method: 'POST', body: form }); const data = await res.json(); if (!res.ok) throw new Error(data.detail); showAlert('Success', 'Account created. Please log in.', 'success'); dom.signupForm.reset(); dom.showLoginLink.onclick(); } catch (err) { showAlert('Signup Error', err.message, 'error'); } finally { showLoader(btn, false); } }
async function onLogin(e) { e.preventDefault(); const btn = dom.loginForm.querySelector('button'); showLoader(btn, true); try { const form = new FormData(dom.loginForm); const res = await fetch('/login', { method: 'POST', body: form }); const data = await res.json(); if (!res.ok) throw new Error(data.detail); currentUserId = data.user_id; dom.authContainer.classList.add('hidden'); dom.appContainer.classList.remove('hidden'); dom.showSessionBtn.click(); } catch (err) { showAlert('Login Error', err.message, 'error'); } finally { showLoader(btn, false); } }
function onLogout() { if(sessionId) onEndSession(); currentUserId=null; sessionId=null; wellnessData=null; dom.appContainer.classList.add('hidden'); dom.authContainer.classList.remove('hidden'); dom.loginForm.reset(); dom.signupForm.reset(); dom.conversation.innerHTML=''; dom.endSessionBtn.disabled=true; clearActiveTask(); }
async function onStartSession() { showLoader(dom.startSessionBtn, true); updateStatus("Starting...", true); try { const form=new FormData(); form.append('user_id', currentUserId); form.append('language', dom.languageSelect.value); form.append('user_gender', document.querySelector('input[name="userGender"]:checked').value); const res = await fetch('/start_session', {method:'POST', body: form}); if(!res.ok) throw new Error(await res.text()); const j = await res.json(); sessionId = j.session_id; dom.sessionSetup.classList.add('hidden'); dom.recordingControls.classList.remove('hidden'); dom.sessionIdEl.textContent = `Session: ...${sessionId.slice(-6)}`; dom.conversation.innerHTML = ""; appendTurn('ai', { reply_structured: j.initial_reply, tts_b64: j.tts_b64 }, null, "start"); dom.endSessionBtn.disabled = false; wellnessData = null; clearActiveTask(); updateStatus("Ready", false); } catch(e) { showAlert('Error', e.message, 'error'); updateStatus("Error", false); } finally { showLoader(dom.startSessionBtn, false); } }
async function onEndSession() { if(!sessionId) return; showLoader(dom.endSessionBtn, true); updateStatus("Ending...", true); try { const form=new FormData(); form.append('session_id', sessionId); const res = await fetch('/end_session',{method:'POST',body:form}); const j = await res.json(); appendTurn('ai',{reply_structured: j.closing_reply,tts_b64:j.tts_b64}, null, "end"); sessionId=null; dom.sessionIdEl.textContent="(no active session)"; dom.sessionSetup.classList.remove('hidden'); dom.recordingControls.classList.add('hidden'); dom.endSessionBtn.disabled=true; clearActiveTask();} catch(e) { showAlert('Error', `Could not end session: ${e.message}`, 'error'); } finally { showLoader(dom.endSessionBtn, false); updateStatus("Ended", false); } }
async function onStartRecording() { updateStatus("Requesting mic...", true); audioChunks = []; try { const stream = await navigator.mediaDevices.getUserMedia({audio:true}); mediaRecorder = new MediaRecorder(stream); mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); }; mediaRecorder.onstop = () => { updateStatus("Stopped", false); dom.submitRecBtn.disabled = false; }; mediaRecorder.start(); dom.startRecBtn.disabled = true; dom.stopRecBtn.disabled = false; dom.startRecBtn.classList.add('recording-active'); updateStatus("Recording...", false); } catch (e) { showAlert('Mic Error', e.message, 'error'); updateStatus("Mic Error", false); } }
function onStopRecording() { if(mediaRecorder?.state === "recording"){ mediaRecorder.stop(); mediaRecorder.stream.getTracks().forEach(t => t.stop()); dom.startRecBtn.disabled = false; dom.stopRecBtn.disabled = true; dom.startRecBtn.classList.remove('recording-active'); } }
async function onSubmitRecording() {
    if (!audioChunks.length || !sessionId) return;
    showLoader(dom.submitRecBtn, true);
    updateStatus("Processing...", true);
    try {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        const form = new FormData();
        form.append('file', blob, 'rec.webm');
        form.append('session_id', sessionId);
        const res = await fetch('/process_audio', { method: 'POST', body: form });
        if (!res.ok) throw new Error((await res.json()).detail);
        const j = await res.json();
        appendTurn('user', { text: j.audio_output.transcript || "(silence)" });
        appendTurn('ai', { reply_structured: j.reply, tts_b64: j.tts_b64 }, null, j.decision.decision);
        audioChunks = [];
        dom.submitRecBtn.disabled = true;
    } catch (e) {
        showAlert('Error', e.message, 'error');
    } finally {
        showLoader(dom.submitRecBtn, false);
        updateStatus("Ready", false);
    }
}
async function onTaskDone() {
    if (!sessionId || !activeTask) return;
    showLoader(dom.markDoneBtn, true);
    try {
        const form = new FormData();
        form.append('session_id', sessionId);
        form.append('task_id', activeTask.task_id);
        const res = await fetch('/task_done', { method: 'POST', body: form });
        if (!res.ok) throw new Error(await res.text());
        const j = await res.json();
        appendTurn('ai', { reply_structured: j.reply, tts_b64: j.tts_b64 }, null, j.decision.decision);
    } catch (e) {
        showAlert('Task Error', e.message, 'error');
    } finally {
        showLoader(dom.markDoneBtn, false);
        clearActiveTask();
    }
}
async function onSaveStructuredData(e) { e.preventDefault(); const btn = dom.dailyReflectionForm.querySelector('button'); showLoader(btn, true); try { const form = new FormData(dom.dailyReflectionForm); form.append('user_id', currentUserId); form.append('date_str', new Date().toISOString().split('T')[0]); const res = await fetch('/wellness/log_structured_entry', { method: 'POST', body: form }); if (!res.ok) throw new Error(await res.text()); showAlert('Success', 'Reflection saved.', 'success'); wellnessData = null; fetchAndRenderWellnessData(); } catch(e) { showAlert('Error', e.message, 'error'); } finally { showLoader(btn, false); } }

// --- WELLNESS DASHBOARD FUNCTIONS ---
async function fetchAndRenderWellnessData() {
    showLoader(dom.showWellnessBtn, true);
    const wellnessCards = dom.wellnessContainer.querySelectorAll('.card');
    wellnessCards.forEach(card => {
        const overlay = document.createElement('div');
        overlay.className = 'card-loader-overlay';
        overlay.innerHTML = `<div class="loader" style="display:block; width: 40px; height: 40px; border-width: 4px;"></div>`;
        card.appendChild(overlay);
    });

    try {
        const form = new FormData(); form.append('user_id', currentUserId);
        const res = await fetch('/wellness/get_wellness_data', { method: 'POST', body: form });
        if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Failed to fetch data.'); }
        wellnessData = await res.json();
        const { raw_logs, summaries, aggregated_data, insights } = wellnessData;
        renderDailyLogging(raw_logs);
        renderSummaries(summaries);
        renderInsights(insights);
        renderGoals(insights.weekly_goals);
        renderCalendarHeatmap(aggregated_data.daily_sentiment);
        renderAllCharts(aggregated_data);
    } catch (e) { showAlert('Dashboard Error', e.message, 'error'); } 
    finally { 
        showLoader(dom.showWellnessBtn, false); 
        wellnessCards.forEach(card => {
            const overlay = card.querySelector('.card-loader-overlay');
            if (overlay) overlay.remove();
        });
    }
}

function renderDailyLogging(logs) {
    dom.hourlyLogger.innerHTML = '';
    const todayStr = new Date().toISOString().split('T')[0];
    const todaysTextLogs = logs.filter(l => l.type === 'hourly_text' && l.log_date_local === todayStr);
    for (let i = 0; i < 24; i++) {
        const btn = document.createElement('button');
        btn.className = 'hour-btn';
        btn.innerHTML = `<span class="btn-text">${i}:00</span><div class="loader"></div>`; 
        const existingLog = todaysTextLogs.find(l => l.log_hour_local === i);
        if (existingLog) { btn.classList.add('logged'); btn.title = `Logged: ${existingLog.analysis.primary_emotion}`; btn.disabled = true; }
        btn.onclick = (e) => logForHour(i, e.currentTarget); // MODIFIED: Simplified arguments
        dom.hourlyLogger.appendChild(btn);
    }
    const todaysStructuredLog = logs.find(l => l.type === 'structured' && l.log_date_local === todayStr);
    renderStructuredForm(todaysStructuredLog);
}

// --- MODIFICATION START: Replaced prompt() with a custom modal ---
function logForHour(hour, buttonEl) {
    showHourlyLogModal(hour, buttonEl);
}

function showHourlyLogModal(hour, buttonEl) {
    const existingModal = document.querySelector('.modal-overlay');
    if (existingModal) existingModal.remove();

    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-box">
            <form class="modal-form" id="hourlyLogForm">
                <h3>How were you feeling at ${hour}:00?</h3>
                <textarea id="hourlyLogText" placeholder="Write down your thoughts and feelings..." required minlength="3"></textarea>
                <div class="modal-actions">
                    <button type="button" class="secondary" id="cancelLogBtn">Cancel</button>
                    <button type="submit" class="primary">
                        <span class="btn-text">Save Log</span>
                        <div class="loader"></div>
                    </button>
                </div>
            </form>
        </div>`;
    
    document.body.appendChild(modal);
    const form = modal.querySelector('#hourlyLogForm');
    const textarea = modal.querySelector('#hourlyLogText');
    const cancelBtn = modal.querySelector('#cancelLogBtn');
    const saveBtn = form.querySelector('button[type="submit"]');

    textarea.focus();
    cancelBtn.onclick = () => modal.remove();

    form.onsubmit = async (e) => {
        e.preventDefault();
        const text = textarea.value.trim();
        if (!text) return;

        showLoader(saveBtn, true);
        try {
            const newLog = await submitHourlyLog(hour, text);
            if (newLog) {
                // Update the original hour button on success
                buttonEl.classList.add('logged');
                buttonEl.disabled = true;
                buttonEl.title = `Logged: ${newLog.analysis.primary_emotion}`;
                 if (wellnessData && wellnessData.raw_logs) {
                    wellnessData.raw_logs.push(newLog);
                }
            }
        } finally {
            // Always remove the modal after submission attempt
            modal.remove();
        }
    };
}

async function submitHourlyLog(hour, text) { 
    const form = new FormData(); 
    form.append('user_id', currentUserId); 
    form.append('hour', hour); 
    form.append('text', text); 
    
    const now = new Date();
    const localDateStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
    form.append('date_str', localDateStr);
    
    try { 
        const res = await fetch('/wellness/log_daily_entry', { method: 'POST', body: form }); 
        if (!res.ok) {
             const errorData = await res.json();
             throw new Error(errorData.detail || 'Failed to submit log.');
        }
        const result = await res.json();
        return result.data; // Return the new log data on success
    } catch(e) { 
        showAlert('Log Error', e.message, 'error'); 
        return null; // Return null on failure
    }
}
// --- MODIFICATION END ---

function renderSummaries(summaries) { dom.sessionSummaryList.innerHTML = (!summaries || summaries.length === 0) ? '<li>(No recent session summaries)</li>' : summaries.map(s => `<li class="session-summary-item"><h5>${s.title || 'Session Summary'}</h5><p class="small">${new Date(s.session_start_utc).toLocaleString()}</p><p style="font-size: 14px;">${s.overall_summary || ''}</p></li>`).join(''); }

function renderStructuredForm(log) {
    const data = log ? log.data : {};
    const social = data.social_interactions || [];
    const activities = data.activities || [];
    const food = data.food_intake || {};
    const fields = {
        sleep_quality: { label: 'How was your sleep quality?', options: ['Good', 'OK', 'Poor'], type: 'radio' },
        social_interactions: { label: 'Who did you spend most time with?', options: ['Family', 'Partner', 'Friends', 'Alone'], type: 'checkbox' },
        activities: { label: 'What activities did you do?', options: ['Work', 'Exercise', 'Hobby', 'Relax', 'Lazy'], type: 'checkbox' }
    };
    let formHTML = '';
    for (const [key, val] of Object.entries(fields)) {
        formHTML += `<div class="form-group"><label>${val.label}</label><div class="options">`;
        val.options.forEach(opt => {
            const checked = val.type === 'radio' ? (data[key] === opt ? 'checked' : '') : (data[key]?.includes(opt) ? 'checked' : '');
            formHTML += `<input type="${val.type}" name="${key}" value="${opt}" id="${key}_${opt}" ${checked} ${log ? 'disabled' : ''}><label for="${key}_${opt}">${opt}</label>`;
        });
        formHTML += `</div></div>`;
    }
    formHTML += `<div class="form-group"><label>Food Intake Quality</label>
        <div class="food-group"><span>Breakfast:</span> <select name="food_breakfast" ${log ? 'disabled' : ''}><option ${food.breakfast === 'Healthy' ? 'selected' : ''}>Healthy</option><option ${food.breakfast === 'Semi' ? 'selected' : ''}>Semi</option><option ${food.breakfast === 'Junk' ? 'selected' : ''}>Junk</option></select></div>
        <div class="food-group"><span>Lunch:</span> <select name="food_lunch" ${log ? 'disabled' : ''}><option ${food.lunch === 'Healthy' ? 'selected' : ''}>Healthy</option><option ${food.lunch === 'Semi' ? 'selected' : ''}>Semi</option><option ${food.lunch === 'Junk' ? 'selected' : ''}>Junk</option></select></div>
        <div class="food-group"><span>Dinner:</span> <select name="food_dinner" ${log ? 'disabled' : ''}><option ${food.dinner === 'Healthy' ? 'selected' : ''}>Healthy</option><option ${food.dinner === 'Semi' ? 'selected' : ''}>Semi</option><option ${food.dinner === 'Junk' ? 'selected' : ''}>Junk</option></select></div>
    </div>`;
    if (!log) { formHTML += `<button type="submit" class="primary" style="grid-column: 1 / -1; margin-top: 16px;"><span class="btn-text">Save Daily Reflection</span><div class="loader"></div></button>`; } 
    else { formHTML += `<p style="grid-column: 1 / -1; text-align: center; color: var(--muted); margin-top: 16px;">Today's reflection has been saved.</p>`; }
    dom.dailyReflectionForm.innerHTML = formHTML;
}

function renderInsights(insights) { 
    dom.insightsContainer.innerHTML = `
        <div class="insight-card insight-notice"><h4>✨ We've Noticed...</h4><p>${insights.causal_inference || 'N/A'}</p></div>
        <div class="insight-card insight-positive"><h4>✅ Positive Pattern</h4><p>${insights.positive_pattern || 'N/A'}</p></div>
        <div class="insight-card insight-negative"><h4>⚠️ Challenge Pattern</h4><p>${insights.challenge_pattern || 'N/A'}</p></div>
        <div class="insight-card insight-growth"><h4>🌱 Area for Growth</h4><p>${insights.area_for_growth || 'N/A'}</p></div>
        <div class="insight-card insight-action" style="grid-column: 1 / -1;"><h4>💡 Actionable Suggestion</h4><p>${insights.actionable_suggestion || 'N/A'}</p></div>`;
}

function renderGoals(goals) { 
    dom.goalsContainer.innerHTML = (goals && goals.length > 0) ? goals.map(g => `
        <div class="goal-card">
            <div class="goal-card-content">
                <span class="goal-card-icon">🎯</span>
                <p>${g.goal}</p>
            </div>
            ${g.rationale ? `<div class="rationale">${g.rationale}</div>` : ''}
        </div>`).join('') : '<p>Your personalized goals will appear here soon.</p>'; 
}

const getSentimentColor = s => { if (s === null || s === undefined) return '#f5f5f4'; const g = s > 0 ? 234 : 128, b = s > 0 ? 109 : 128, r = s < 0 ? 240 : 221; const a = Math.abs(s) * .8 + .2; return `rgba(${r}, ${g}, ${b}, ${a})`; };
async function renderCalendarHeatmap(dailySentiments) {
    dom.calendarHeatmap.innerHTML = '';
    const today = new Date();
    const startDate = new Date();
    startDate.setDate(today.getDate() - 29);
    for (let d = new Date(startDate); d <= today; d.setDate(d.getDate() + 1)) {
        const dateStr = d.toISOString().split('T')[0];
        const cell = document.createElement('div');
        cell.className = 'calendar-cell';
        cell.style.backgroundColor = getSentimentColor(dailySentiments[dateStr]);
        cell.onclick = async () => {
            const logsForDay = (wellnessData.raw_logs || []).filter(l => l.type === 'hourly_text' && l.log_date_local === dateStr);
            showAlert(`Summary for ${dateStr}`, 'Generating summary...', 'info');
            try {
                const res = await fetch('/wellness/summarize_day_logs', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ user_id: currentUserId, logs: logsForDay }) });
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                showAlert(`Summary for ${dateStr}`, data.summary, 'info');
            } catch (e) { showAlert('Error', `Could not generate summary: ${e.message}`, 'error'); }
        };
        cell.innerHTML = `<span class="tooltip">${dateStr}<br>Sentiment: ${dailySentiments[dateStr]?.toFixed(2) || 'N/A'}<br><b>Click for summary</b></span>`;
        dom.calendarHeatmap.appendChild(cell);
    }
}
function renderAllCharts(aggregated) {
    const chartColors = ['#ddea6d', '#68d391', '#f6ad55', '#7180d4', '#b794f4', '#4fd1c5', '#f687b3', '#a3aed2'];
    const correlationMetrics = aggregated.correlation_metrics || {};

    renderCorrelationChart('sleepChart', correlationMetrics.sleep_quality, 'Avg. Sentiment');
    renderCorrelationChart('activityChart', correlationMetrics.activity, 'Avg. Sentiment');
    renderCorrelationChart('socialChart', correlationMetrics.social, 'Avg. Sentiment');

// This is the CORRECTED code
    if (charts.emotion) charts.emotion.destroy();
    const emotionCtx = document.getElementById('emotionDistributionChart').getContext('2d'); // <--- CORRECTED
    const emotionData = aggregated.emotion_counts || {};
    charts.emotion = new Chart(emotionCtx, { type: 'pie',
    
    data: { labels: Object.keys(emotionData), datasets: [{ data: Object.values(emotionData), backgroundColor: chartColors }] }});
    if (charts.sentiment) charts.sentiment.destroy();
    const sentimentCtx = document.getElementById('sentimentOverTimeChart').getContext('2d');
    const dailyData = aggregated.daily_sentiment || {};
    const sortedDates = Object.keys(dailyData).sort();
    charts.sentiment = new Chart(sentimentCtx, { type: 'line', data: { labels: sortedDates, datasets: [{ label: 'Average Sentiment', data: sortedDates.map(d => dailyData[d]), borderColor: '#c8d75a', tension: 0.1, fill: false, borderWidth: 3 }] }, options: { scales: { x: { type: 'time', time: { unit: 'day' } }, y: { min: -1, max: 1 } } } });
    
    dom.wordCloudContainer.innerHTML = '';
    const words = (aggregated.word_cloud_data || []).map(d => ({text: d.text, size: 10 + Math.pow(d.value, 0.7) * 10}));
    if (words.length > 0) {
        const color = d3.scaleOrdinal(chartColors);
        const layout = d3.layout.cloud().size([dom.wordCloudContainer.clientWidth > 0 ? dom.wordCloudContainer.clientWidth : 300, 300]).words(words).padding(5).rotate(() => 0).fontSize(d => d.size).on("end", draw);
        layout.start();
        function draw(words) { d3.select("#wordCloudContainer").append("svg").attr("width", layout.size()[0]).attr("height", layout.size()[1]).append("g").attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")").selectAll("text").data(words).enter().append("text").style("font-size", d => d.size + "px").style("font-family", "Inter").style("fill", (d,i) => color(i)).attr("text-anchor", "middle").attr("transform", d => "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")").text(d => d.text); }
    }
}

// MODIFICATION START: Replaced with new function for dynamic scaling and better visuals
function renderCorrelationChart(canvasId, data, label) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    if (charts[canvasId]) charts[canvasId].destroy();

    if (!data || Object.keys(data).length === 0) {
        ctx.font = "14px Inter";
        ctx.fillStyle = "#a8a29e";
        ctx.textAlign = "center";
        ctx.fillText("Not enough data to show this chart yet.", ctx.canvas.width / 2, ctx.canvas.height / 2);
        return;
    };
    
    const orderMap = {
        'sleepChart': ['Good', 'OK', 'Poor'],
        'activityChart': ['Work', 'Exercise', 'Hobby', 'Relax', 'Lazy'],
        'socialChart': ['Family', 'Partner', 'Friends', 'Alone']
    };
    const desiredOrder = orderMap[canvasId] || Object.keys(data);
    
    const sortedLabels = desiredOrder;
    const sortedValues = sortedLabels.map(label => data[label] || 0.0);
    
    // --- DYNAMIC SCALING LOGIC ---
    // 1. Find the largest absolute value to determine the data range.
    const maxAbsValue = Math.max(...sortedValues.map(v => Math.abs(v)));
    
    // 2. Set a minimum scale boundary for visual consistency, even with very small numbers.
    // This prevents the chart from looking "empty" or overly dramatic for tiny values.
    const minScaleBoundary = 0.2;
    
    // 3. Calculate the scale max by adding 20% padding, but not going below the minimum boundary.
    const scaleMax = Math.max(minScaleBoundary, maxAbsValue * 1.2);
    // --- END DYNAMIC SCALING LOGIC ---

    const backgroundColors = sortedValues.map(v => v >= 0 ? 'rgba(104, 211, 145, 0.8)' : 'rgba(240, 128, 128, 0.8)');
    const borderColors = sortedValues.map(v => v >= 0 ? 'rgba(104, 211, 145, 1)' : 'rgba(240, 128, 128, 1)');

    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedLabels,
            datasets: [{
                label: label,
                data: sortedValues,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1,
                maxBarThickness: 60 // Controls the maximum width of the bars for a cleaner look
            }]
        },
        options: {
            scales: {
                y: {
                    // Use the dynamically calculated scale
                    min: -scaleMax,
                    max: scaleMax
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}
