/**
 * chat.js — Chat + voice UI logic
 *
 * Sections:
 *  1. Session & DOM refs
 *  2. Waveform visualizer (Web Audio API)
 *  3. Message rendering helpers (appendMessage, showLoading, etc.)
 *  4. TTS — playTTS() hooks into waveform so the canvas animates during playback
 *  5. Core sendMessage()
 *  6. Event listeners (send btn, Enter key)
 *  7. STT — mic recording; hooks into waveform for live mic animation
 */

// ── 1. Session & DOM refs ─────────────────────────────────────
let sessionId = localStorage.getItem("chatbot_session_id") || null;

const messagesEl     = document.getElementById("chat-messages");
const inputEl        = document.getElementById("user-input");
const sendBtn        = document.getElementById("send-btn");
const waveformStatus = document.getElementById("waveform-status");
const waveCanvas     = document.getElementById("waveform-canvas");
const micBtn         = document.getElementById("mic-btn");

// Wire up close button — hides the card (non-destructive)
const closeBtn  = document.getElementById("close-btn");
const mainCard  = document.getElementById("main-card");
if (closeBtn && mainCard) {
  closeBtn.addEventListener("click", () => {
    mainCard.style.opacity = "0";
    mainCard.style.transform = "scale(0.96)";
    mainCard.style.transition = "opacity 0.25s ease, transform 0.25s ease";
    setTimeout(() => { mainCard.style.display = "none"; }, 260);
  });
}

// ── 2. Waveform Visualizer (Web Audio API) ───────────────────
// 
// The visualizer uses an AnalyserNode to get frequency-domain data in real time.
// It supports three modes:
//   - "idle":      gentle sine-wave animation, no audio input
//   - "mic":       live microphone amplitude via getUserMedia stream
//   - "tts":       audio element connected via createMediaElementSource
//
// Only ONE mode is active at a time. The animationFrameId guards the loop.

const waveCtx = waveCanvas ? waveCanvas.getContext("2d") : null;
let audioCtx       = null;   // Created lazily on first user gesture
let analyser       = null;   // AnalyserNode — shared across modes
let waveMode       = "idle"; // "idle" | "mic" | "tts"
let animFrameId    = null;   // requestAnimationFrame handle
let idlePhase      = 0;      // Phase accumulator for idle sine wave

/** Lazily create the AudioContext (browser requires a user gesture first). */
function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser  = audioCtx.createAnalyser();
    analyser.fftSize = 256;           // 128 frequency bins
    analyser.smoothingTimeConstant = 0.8;
    analyser.connect(audioCtx.destination);
  }
}

/** Resize the canvas to match its CSS display size (avoids blurry canvas). */
function resizeCanvas() {
  if (!waveCanvas) return;
  const rect = waveCanvas.getBoundingClientRect();
  waveCanvas.width  = rect.width  * window.devicePixelRatio;
  waveCanvas.height = rect.height * window.devicePixelRatio;
  waveCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
}
window.addEventListener("resize", resizeCanvas);
// Delay initial resize so layout is settled
setTimeout(resizeCanvas, 100);

/**
 * Draw one frame of the waveform.
 * Mode "idle" → smooth sine curve
 * Mode "mic" or "tts" → amplitude-driven multi-line wave
 */
function drawWave() {
  if (!waveCtx || !waveCanvas) return;

  const W = waveCanvas.width  / window.devicePixelRatio;
  const H = waveCanvas.height / window.devicePixelRatio;
  const cx = waveCtx;

  cx.clearRect(0, 0, W, H);

  // Build a gradient for the wave stroke (cyan → indigo)
  const grad = cx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0,    "#38bdf8");
  grad.addColorStop(0.5,  "#818cf8");
  grad.addColorStop(1,    "#38bdf8");

  cx.strokeStyle = grad;
  cx.lineWidth   = 2.2;
  cx.lineCap     = "round";
  cx.lineJoin    = "round";

  if (waveMode === "idle") {
    // Smooth sine wave — gently pulsing when nothing is happening
    idlePhase += 0.025;
    cx.globalAlpha = 0.45;
    cx.beginPath();
    for (let x = 0; x <= W; x++) {
      const t   = x / W;
      const amp = 12;
      const y   = H / 2 + Math.sin(t * Math.PI * 4 + idlePhase) * amp
                        + Math.sin(t * Math.PI * 2 - idlePhase * 0.7) * (amp * 0.4);
      x === 0 ? cx.moveTo(x, y) : cx.lineTo(x, y);
    }
    cx.stroke();
    cx.globalAlpha = 1;

  } else {
    // Live data from AnalyserNode (mic or tts)
    // getByteTimeDomainData returns 0-255 where 128 = silence
    const bufferLength = analyser.frequencyBinCount;
    const dataArray    = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);

    cx.beginPath();
    const sliceW = W / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;      // normalise to 0.0–2.0
      const y = (v * H) / 2;
      i === 0 ? cx.moveTo(x, y) : cx.lineTo(x, y);
      x += sliceW;
    }
    cx.lineTo(W, H / 2);
    cx.stroke();
  }

  animFrameId = requestAnimationFrame(drawWave);
}

/** Start the animation loop (idempotent). */
function startWaveAnimation(mode) {
  waveMode = mode;
  if (!animFrameId) {
    animFrameId = requestAnimationFrame(drawWave);
  }
}

/** Stop the loop and reset to idle. */
function stopWaveAnimation() {
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  waveMode = "idle";
  // Restart idle gentle animation
  startWaveAnimation("idle");
}

/** Update the status text below the waveform. */
function setWaveStatus(text, active = false) {
  if (!waveformStatus) return;
  waveformStatus.textContent = text;
  waveformStatus.classList.toggle("active", active);
}

// Start idle animation immediately
startWaveAnimation("idle");

// ── 3. Message helpers ────────────────────────────────────────

function appendMessage(text, role) {
  const msgDiv    = document.createElement("div");
  const bubbleDiv = document.createElement("div");

  msgDiv.classList.add("message", role);
  bubbleDiv.classList.add("bubble");

  if (role === "bot") {
    const rawHtml   = marked.parse(text);
    const cleanHtml = DOMPurify.sanitize(rawHtml);
    bubbleDiv.innerHTML = cleanHtml;
  } else {
    bubbleDiv.textContent = text;
  }

  msgDiv.appendChild(bubbleDiv);

  // Speak button for bot messages
  if (role === "bot") {
    const speakBtn = document.createElement("button");
    speakBtn.classList.add("speak-btn");
    speakBtn.innerHTML = "🔊";
    speakBtn.title     = "Speak (TTS)";
    speakBtn.onclick   = () => playTTS(text, speakBtn);
    msgDiv.appendChild(speakBtn);
  }

  messagesEl.appendChild(msgDiv);
  scrollToBottom();
  return msgDiv;
}

function showLoading() {
  const wrapper   = document.createElement("div");
  const indicator = document.createElement("div");
  wrapper.classList.add("message", "bot");
  indicator.classList.add("loading-indicator");
  for (let i = 0; i < 3; i++) indicator.appendChild(document.createElement("span"));
  wrapper.appendChild(indicator);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setInputDisabled(disabled) {
  inputEl.disabled  = disabled;
  sendBtn.disabled  = disabled;
}

// Auto-grow textarea
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = inputEl.scrollHeight + "px";
});

// ── 4. TTS (Text-to-Speech) with waveform hook ───────────────
//
// After getting the MP3 blob we create an Audio element and connect it to the
// Web Audio graph via createMediaElementSource → analyser → destination.
// While audio plays the waveform canvas animates in "tts" mode.

let ttsSourceNode = null; // Keep reference to avoid duplicate connections

async function playTTS(text, btnElement) {
  const originalHtml = btnElement.innerHTML;
  btnElement.innerHTML = "⏳";
  btnElement.disabled  = true;
  setWaveStatus("Generating speech…", true);

  try {
    const response = await fetch("/api/tts", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });

    if (!response.ok) throw new Error("TTS failed");

    const blob     = await response.blob();
    const audioUrl = URL.createObjectURL(blob);
    const audio    = new Audio(audioUrl);

    // ── Web Audio hook: route the <audio> element through our AnalyserNode ──
    ensureAudioCtx();

    // Disconnect any previous TTS source to avoid multiple connections
    if (ttsSourceNode) {
      try { ttsSourceNode.disconnect(); } catch (_) {}
      ttsSourceNode = null;
    }

    // createMediaElementSource can only be called once per element
    const sourceNode = audioCtx.createMediaElementSource(audio);
    ttsSourceNode    = sourceNode;
    sourceNode.connect(analyser);  // analyser already connected to destination

    audio.addEventListener("play",  () => {
      startWaveAnimation("tts");
      setWaveStatus("Speaking…", true);
    });

    audio.addEventListener("ended", () => {
      stopWaveAnimation();
      setWaveStatus("Tap the mic to speak…", false);
      btnElement.innerHTML = originalHtml;
      btnElement.disabled  = false;
      URL.revokeObjectURL(audioUrl);
    });

    // Resume AudioContext if suspended (browser autoplay policy)
    if (audioCtx.state === "suspended") await audioCtx.resume();
    await audio.play();

  } catch (err) {
    console.error(err);
    stopWaveAnimation();
    setWaveStatus("Voice unavailable", false);
    alert("Voice feature unavailable right now");
    btnElement.innerHTML = originalHtml;
    btnElement.disabled  = false;
  }
}

// ── 5. Core sendMessage ───────────────────────────────────────

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  inputEl.style.height = "auto";

  appendMessage(text, "user");
  setInputDisabled(true);
  const loadingEl = showLoading();

  try {
    const response = await fetch("/api/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text, session_id: sessionId }),
    });

    loadingEl.remove();
    const data = await response.json();

    if (!response.ok || data.error) {
      appendMessage(data.error || "Something went wrong. Please try again.", "error");
    } else {
      if (data.session_id) {
        sessionId = data.session_id;
        localStorage.setItem("chatbot_session_id", sessionId);
      }
      const botMsg = appendMessage(data.reply, "bot");
      // Update waveform status to a short preview of the reply
      setWaveStatus(data.reply.slice(0, 60) + (data.reply.length > 60 ? "…" : ""), false);
    }

  } catch (networkError) {
    loadingEl.remove();
    appendMessage("Could not reach the server. Is the FastAPI app running?", "error");
  } finally {
    setInputDisabled(false);
    inputEl.focus();
  }
}

// ── 6. Event listeners ────────────────────────────────────────

sendBtn.addEventListener("click", sendMessage);

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

// ── 7. STT (Speech-to-Text) with live waveform ───────────────
//
// When recording starts we connect the microphone stream to the same
// AnalyserNode so the canvas visualizes the user's voice in real time.

let mediaRecorder    = null;
let audioChunks      = [];
let isRecording      = false;
let micSourceNode    = null; // MediaStreamAudioSourceNode for mic

if (micBtn) {
  micBtn.addEventListener("click", async () => {
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  });
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks   = [];

    // ── Web Audio hook: pipe mic into AnalyserNode ──
    ensureAudioCtx();
    if (audioCtx.state === "suspended") await audioCtx.resume();

    // Connect mic stream to the shared analyser (NOT to destination — avoids feedback)
    micSourceNode = audioCtx.createMediaStreamSource(stream);
    micSourceNode.connect(analyser);

    mediaRecorder.addEventListener("dataavailable", e => audioChunks.push(e.data));

    mediaRecorder.addEventListener("stop", async () => {
      // Disconnect mic from analyser when done
      if (micSourceNode) {
        try { micSourceNode.disconnect(); } catch (_) {}
        micSourceNode = null;
      }
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      stream.getTracks().forEach(t => t.stop());
      await transcribeAudio(audioBlob);
    });

    mediaRecorder.start();
    isRecording = true;
    startWaveAnimation("mic");
    setWaveStatus("Listening…", true);
    micBtn.innerHTML = "⏹";
    micBtn.classList.add("recording");

  } catch (err) {
    console.error("Mic access denied or error:", err);
    alert("Microphone access is required for voice input.");
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    isRecording = false;
    stopWaveAnimation();
    setWaveStatus("Transcribing…", true);
    // Restore mic icon via SVG (innerHTML reset after transcription finishes)
    micBtn.innerHTML = "⏳";
    micBtn.classList.remove("recording");
  }
}

async function transcribeAudio(audioBlob) {
  micBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");

    const response = await fetch("/api/stt", { method: "POST", body: formData });
    if (!response.ok) throw new Error("STT failed");

    const data = await response.json();
    if (data.error) throw new Error(data.error);

    // VAD found no human speech — give the user a hint without an alert
    if (data.vad === "no_speech" || !data.text) {
      setWaveStatus("No speech detected — try again", false);
      return;
    }

    const cur = inputEl.value.trim();
    inputEl.value = cur ? cur + " " + data.text : data.text;
    inputEl.style.height = "auto";
    inputEl.style.height = inputEl.scrollHeight + "px";
    setWaveStatus(data.text, false);

  } catch (err) {
    console.error(err);
    setWaveStatus("Transcription failed", false);
    alert("Voice feature unavailable right now");
  } finally {
    // Restore mic SVG
    micBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"/>
      <path d="M19 10a1 1 0 0 0-2 0 5 5 0 0 1-10 0 1 1 0 0 0-2 0 7 7 0 0 0 6 6.93V19H9a1 1 0 0 0 0 2h6a1 1 0 0 0 0-2h-2v-2.07A7 7 0 0 0 19 10z"/>
    </svg>`;
    micBtn.disabled = false;
  }
}
