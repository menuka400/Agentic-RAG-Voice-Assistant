/**
 * chat.js — Chat + voice UI logic
 */

let sessionId = localStorage.getItem("chatbot_session_id") || null;

const messagesEl     = document.getElementById("chat-messages");
const inputEl        = document.getElementById("user-input");
const sendBtn        = document.getElementById("send-btn");
const waveformStatus = document.getElementById("waveform-status");
const waveCanvas     = document.getElementById("waveform-canvas");
const micBtn         = document.getElementById("mic-btn");

// NEW BUTTONS
const interruptBtn   = document.getElementById("interrupt-btn");
const muteBtn        = document.getElementById("mute-btn");
const holdBtn        = document.getElementById("hold-btn");
const iconMuted      = document.querySelector(".icon-muted");
const iconUnmuted    = document.querySelector(".icon-unmuted");
const muteLabel      = document.querySelector(".mute-label");

// GLOBALS FOR NEW LOGIC
let chatAbortController = null;
let currentAudio = null;
let isMuted = false;
let isHeld = false;

// Wire up close button (kept for existing refs)
const closeBtn  = document.getElementById("close-btn");
const mainCard  = document.getElementById("main-card");
if (closeBtn && mainCard) {
  closeBtn.addEventListener("click", () => {
    mainCard.style.display = "none";
  });
}

// ── NEW BUTTON LOGIC ──────────────────────────────────────────

// 1. Interrupt Button
interruptBtn.addEventListener("click", () => {
  if (chatAbortController) {
    chatAbortController.abort();
    chatAbortController = null;
  }
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
  
  // Cleanup UI
  stopWaveAnimation();
  setWaveStatus("Interrupted.", false);
  setInputDisabled(false);
  setInterruptActive(false);
  
  // Remove loading indicator if exists
  const loading = document.querySelector(".loading-indicator");
  if (loading) loading.parentElement.remove();
});

function setInterruptActive(active) {
  interruptBtn.disabled = !active;
  interruptBtn.classList.toggle("active", active);
}

// 2. Mute Button
muteBtn.addEventListener("click", () => {
  isMuted = !isMuted;
  muteBtn.classList.toggle("muted", isMuted);
  if (isMuted) {
    iconUnmuted.style.display = "none";
    iconMuted.style.display = "block";
    muteLabel.textContent = "Unmute";
    // Pause currently playing TTS if muted mid-stream
    if (currentAudio && !currentAudio.paused) {
      currentAudio.pause();
      stopWaveAnimation();
      setWaveStatus("Muted.", false);
      setInterruptActive(false);
    }
  } else {
    iconUnmuted.style.display = "block";
    iconMuted.style.display = "none";
    muteLabel.textContent = "Mute";
  }
});

// 3. Hold Button
holdBtn.addEventListener("click", () => {
  if (!isRecording || !mediaRecorder) return;
  
  if (!isHeld) {
    // Pause recording
    mediaRecorder.pause();
    isHeld = true;
    holdBtn.classList.add("held");
    stopWaveAnimation();
    setWaveStatus("Paused — tap Hold to resume", false);
    
    // Pause TTS if playing
    if (currentAudio && !currentAudio.paused) {
      currentAudio.pause();
    }
  } else {
    // Resume recording
    mediaRecorder.resume();
    isHeld = false;
    holdBtn.classList.remove("held");
    startWaveAnimation("mic");
    setWaveStatus("Listening…", true);
  }
});

function setHoldActive(active) {
  holdBtn.disabled = !active;
  if (!active) {
    isHeld = false;
    holdBtn.classList.remove("held");
  }
}

// ── 2. Waveform Visualizer ────────────────────────────────────

const waveCtx = waveCanvas ? waveCanvas.getContext("2d") : null;
let audioCtx       = null;
let analyser       = null;
let waveMode       = "idle"; 
let animFrameId    = null;   
let idlePhase      = 0;      

function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser  = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.8;
    // NOTE: analyser is intentionally NOT connected to destination here.
    // Connecting it globally would route live mic input straight to the
    // speakers (feedback/echo). Each source connects to destination
    // individually only when it actually needs to be heard (TTS), while
    // the mic source only feeds the analyser for visualization.
  }
}

function resizeCanvas() {
  if (!waveCanvas) return;
  const rect = waveCanvas.getBoundingClientRect();
  waveCanvas.width  = rect.width  * window.devicePixelRatio;
  waveCanvas.height = rect.height * window.devicePixelRatio;
  waveCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
}
window.addEventListener("resize", resizeCanvas);
setTimeout(resizeCanvas, 100);

function drawWave() {
  if (!waveCtx || !waveCanvas) return;

  const W = waveCanvas.width  / window.devicePixelRatio;
  const H = waveCanvas.height / window.devicePixelRatio;
  const cx = waveCtx;

  cx.clearRect(0, 0, W, H);

  const grad = cx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0,    "#38bdf8");
  grad.addColorStop(0.5,  "#818cf8");
  grad.addColorStop(1,    "#38bdf8");

  cx.strokeStyle = grad;
  cx.lineWidth   = 2.5;
  cx.lineCap     = "round";
  cx.lineJoin    = "round";

  if (waveMode === "idle") {
    idlePhase += 0.02;
    cx.globalAlpha = 0.5;
    cx.beginPath();
    for (let x = 0; x <= W; x++) {
      const t   = x / W;
      const amp = 15;
      const y   = H / 2 + Math.sin(t * Math.PI * 3 + idlePhase) * amp
                        + Math.sin(t * Math.PI * 2 - idlePhase * 0.5) * (amp * 0.5);
      x === 0 ? cx.moveTo(x, y) : cx.lineTo(x, y);
    }
    cx.stroke();
    cx.globalAlpha = 1;

  } else {
    const bufferLength = analyser.frequencyBinCount;
    const dataArray    = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);

    cx.beginPath();
    const sliceW = W / bufferLength;
    let x = 0;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0; 
      const y = (v * H) / 2;
      i === 0 ? cx.moveTo(x, y) : cx.lineTo(x, y);
      x += sliceW;
    }
    cx.lineTo(W, H / 2);
    cx.stroke();
  }

  animFrameId = requestAnimationFrame(drawWave);
}

function startWaveAnimation(mode) {
  waveMode = mode;
  if (!animFrameId) {
    animFrameId = requestAnimationFrame(drawWave);
  }
}

function stopWaveAnimation() {
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  waveMode = "idle";
  startWaveAnimation("idle");
}

function setWaveStatus(text, active = false) {
  if (!waveformStatus) return;
  waveformStatus.textContent = text;
  waveformStatus.classList.toggle("active", active);
}

startWaveAnimation("idle");

// ── 2b. Speech text cleanup ────────────────────────────────────
// Strips markdown syntax, emoji, and the "Source: ..." attribution
// line before text is sent to TTS, so the voice never reads out
// symbols like "asterisk" or "thought balloon" — it only speaks the
// actual sentence content. The chat bubble still renders the full
// original markdown; only the TTS request uses this cleaned copy.

function stripForSpeech(text) {
  if (!text) return "";
  let clean = text;

  // Drop a trailing "Source: ..." attribution line (with or without emoji)
  clean = clean.replace(/^[^\S\r\n]*\p{Emoji_Presentation}?\s*Source:.*$/gim, "");

  // Markdown: images/links -> keep just the visible label
  clean = clean.replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1");
  clean = clean.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");

  // Markdown: bold/italic/strikethrough/inline code/headers/blockquote/hr
  clean = clean.replace(/(\*\*\*|___)(.*?)\1/g, "$2");
  clean = clean.replace(/(\*\*|__)(.*?)\1/g, "$2");
  clean = clean.replace(/(\*|_)(.*?)\1/g, "$2");
  clean = clean.replace(/~~(.*?)~~/g, "$1");
  clean = clean.replace(/`{1,3}([^`]*)`{1,3}/g, "$1");
  clean = clean.replace(/^#{1,6}\s+/gm, "");
  clean = clean.replace(/^>\s?/gm, "");
  clean = clean.replace(/^\s*([-*_]){3,}\s*$/gm, "");
  clean = clean.replace(/^\s*[-*+]\s+/gm, "");
  clean = clean.replace(/^\s*\d+\.\s+/gm, "");

  // Strip remaining emoji / pictographs
  clean = clean.replace(/\p{Extended_Pictographic}/gu, "");

  return clean.replace(/\n{2,}/g, "\n").trim();
}

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

  if (role === "bot") {
    const speakBtn = document.createElement("button");
    speakBtn.classList.add("speak-btn");
    speakBtn.innerHTML = "🔊";
    speakBtn.title     = "Speak (TTS)";
    speakBtn.onclick   = () => playTTS(stripForSpeech(text), speakBtn, true); // true = manual click
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

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = inputEl.scrollHeight + "px";
});

// ── 4. TTS (Text-to-Speech) ───────────────────────────────────

let ttsSourceNode = null;

async function playTTS(text, btnElement, isManual = false) {
  // If muted and it's not a manual click, skip auto-play
  if (isMuted && !isManual) return;
  
  if (btnElement) {
    var originalHtml = btnElement.innerHTML;
    btnElement.innerHTML = "⏳";
    btnElement.disabled  = true;
  }
  setWaveStatus("Generating speech…", true);

  // If another audio is playing, stop it
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
  }
  
  chatAbortController = new AbortController();
  setInterruptActive(true);

  try {
    const response = await fetch("/api/tts", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
      signal:  chatAbortController.signal
    });

    if (!response.ok) throw new Error("TTS failed");

    const blob     = await response.blob();
    const audioUrl = URL.createObjectURL(blob);
    currentAudio   = new Audio(audioUrl);

    ensureAudioCtx();

    if (ttsSourceNode) {
      try { ttsSourceNode.disconnect(); } catch (_) {}
      ttsSourceNode = null;
    }

    const sourceNode = audioCtx.createMediaElementSource(currentAudio);
    ttsSourceNode    = sourceNode;
    sourceNode.connect(analyser);
    sourceNode.connect(audioCtx.destination);

    currentAudio.addEventListener("play",  () => {
      startWaveAnimation("tts");
      setWaveStatus("Speaking…", true);
    });

    currentAudio.addEventListener("ended", () => {
      stopWaveAnimation();
      setWaveStatus("Tap the mic to speak…", false);
      if (btnElement) {
        btnElement.innerHTML = originalHtml;
        btnElement.disabled  = false;
      }
      URL.revokeObjectURL(audioUrl);
      currentAudio = null;
      setInterruptActive(false);
    });

    if (audioCtx.state === "suspended") await audioCtx.resume();
    await currentAudio.play();

  } catch (err) {
    if (err.name === 'AbortError') {
      console.log('TTS aborted');
    } else {
      console.error(err);
      stopWaveAnimation();
      setWaveStatus("Voice unavailable", false);
      if (btnElement) {
        btnElement.innerHTML = originalHtml;
        btnElement.disabled  = false;
      }
    }
  } finally {
    if (!currentAudio) {
      setInterruptActive(false);
    }
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
  
  chatAbortController = new AbortController();
  setInterruptActive(true);

  try {
    const response = await fetch("/api/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text, session_id: sessionId }),
      signal:  chatAbortController.signal
    });

    loadingEl.remove();
    const data = await response.json();

    if (!response.ok || data.error) {
      appendMessage(data.error || "Something went wrong. Please try again.", "error");
      setInterruptActive(false);
    } else {
      if (data.session_id) {
        sessionId = data.session_id;
        localStorage.setItem("chatbot_session_id", sessionId);
      }
      const botMsg = appendMessage(data.reply, "bot");
      
      const botSpeakBtn = botMsg.querySelector(".speak-btn");
      
      // Auto-play TTS if not muted
      if (!isMuted) {
        // playTTS will handle setting setInterruptActive(false) when done
        playTTS(stripForSpeech(data.reply), botSpeakBtn, false);
      } else {
        setWaveStatus(data.reply.slice(0, 60) + (data.reply.length > 60 ? "…" : ""), false);
        setInterruptActive(false);
      }
    }

  } catch (err) {
    loadingEl.remove();
    if (err.name === 'AbortError') {
      appendMessage("Request interrupted.", "error");
    } else {
      appendMessage("Could not reach the server. Is the FastAPI app running?", "error");
    }
    setInterruptActive(false);
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

// ── 7. STT (Speech-to-Text) ───────────────────────────────────

let mediaRecorder    = null;
let audioChunks      = [];
let isRecording      = false;
let micSourceNode    = null;

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

    ensureAudioCtx();
    if (audioCtx.state === "suspended") await audioCtx.resume();

    micSourceNode = audioCtx.createMediaStreamSource(stream);
    micSourceNode.connect(analyser);

    mediaRecorder.addEventListener("dataavailable", e => audioChunks.push(e.data));

    mediaRecorder.addEventListener("stop", async () => {
      if (micSourceNode) {
        try { micSourceNode.disconnect(); } catch (_) {}
        micSourceNode = null;
      }
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      stream.getTracks().forEach(t => t.stop());
      
      // Convert to WAV so the Python backend (scipy/VAD) can read it
      ensureAudioCtx();
      try {
        const arrayBuffer = await audioBlob.arrayBuffer();
        const decodedData = await audioCtx.decodeAudioData(arrayBuffer);
        
        // Resample to 16kHz using OfflineAudioContext for the backend VAD
        const targetSampleRate = 16000;
        const offlineCtx = new OfflineAudioContext(1, Math.ceil(decodedData.duration * targetSampleRate), targetSampleRate);
        const source = offlineCtx.createBufferSource();
        source.buffer = decodedData;
        source.connect(offlineCtx.destination);
        source.start();
        
        const resampledData = await offlineCtx.startRendering();
        const wavBlob = audioBufferToWav(resampledData);
        await transcribeAudio(wavBlob);
      } catch (err) {
        console.error("Failed to convert audio to WAV", err);
        // Fallback to webm if conversion fails
        await transcribeAudio(audioBlob);
      }
    });

    mediaRecorder.start();
    isRecording = true;
    startWaveAnimation("mic");
    setWaveStatus("Listening…", true);
    micBtn.innerHTML = "⏹";
    micBtn.classList.add("recording");
    
    // Enable Hold button
    setHoldActive(true);

  } catch (err) {
    console.error("Mic access denied or error:", err);
    alert("Microphone access is required for voice input.");
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    isRecording = false;
    stopWaveAnimation();
    setWaveStatus("Transcribing…", true);
    micBtn.innerHTML = "⏳";
    micBtn.classList.remove("recording");
    setHoldActive(false);
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

    if (data.vad === "no_speech" || !data.text) {
      setWaveStatus("No speech detected — try again", false);
      return;
    }

    const cur = inputEl.value.trim();
    inputEl.value = cur ? cur + " " + data.text : data.text;
    inputEl.style.height = "auto";
    inputEl.style.height = inputEl.scrollHeight + "px";
    setWaveStatus(data.text, false);
    
    // Auto send the message
    sendMessage();

  } catch (err) {
    console.error(err);
    setWaveStatus("Transcription failed", false);
    alert("Voice feature unavailable right now");
  } finally {
    micBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"/>
      <path d="M19 10a1 1 0 0 0-2 0 5 5 0 0 1-10 0 1 1 0 0 0-2 0 7 7 0 0 0 6 6.93V19H9a1 1 0 0 0 0 2h6a1 1 0 0 0 0-2h-2v-2.07A7 7 0 0 0 19 10z"/>
    </svg>`;
    micBtn.disabled = false;
  }
}

// ── 8. WAV Encoder Helper ────────────────────────────────────

function audioBufferToWav(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  
  let samples;
  if (numChannels === 1) {
    samples = buffer.getChannelData(0);
  } else {
    // Mix down to mono
    const left = buffer.getChannelData(0);
    const right = buffer.getChannelData(1);
    samples = new Float32Array(left.length);
    for (let i = 0; i < left.length; i++) {
      samples[i] = (left[i] + right[i]) / 2;
    }
  }

  const bytesPerSample = bitDepth / 8;
  const blockAlign = 1 * bytesPerSample; // Mono
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const bufferLength = 44 + dataSize;
  const arrayBuffer = new ArrayBuffer(bufferLength);
  const view = new DataView(arrayBuffer);

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7FFF;
    view.setInt16(offset, s, true);
    offset += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
}
