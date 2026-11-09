"""
main_streamlit.py — Academic Document RAG Chatbot Interface.

Design follows the campus academic portal aesthetic: light blue background,
clean white bordered cards, cyan group headers, and crisp responsive components.
Style definitions are maintained in `app/theme.py`.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
import queue
import re
import sys
import threading
import uuid

import pandas as pd
import requests
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answer_guard import diagnose_answer
from app.citations import (
    extract_cited_pages,
    strip_inline_source_lines,
)
from app.config import (
    DEFAULT_RETRIEVAL_CONFIG,
    EMBEDDING_MODEL,
    LLM_MODEL,
    UTILITY_MODEL,
    RetrievalConfig,
)
from app.llm_client import LLMConfigError, call_llm, call_llm_stream, effective_model
from app.pdf_viewer import render_pdf_viewer, render_source_references
from app.prompt_builder import build_no_context_response, build_prompt, detect_query_language
from app.public_config import (
    app_mode,
    is_operator_mode,
    public_rate_limit,
    public_rate_window_seconds,
)
from app.rate_limit import SessionRateLimiter
from app.retrieval import (
    DocumentRetriever,
    RetrievalOutcome,
    build_source_references,
    format_sources,
)
from app.theme import (
    inject_theme,
    nav_label,
    render_brand,
    render_footer,
    render_lead,
    render_status_badge,
    render_welcome,
)

LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Academic RAG Chatbot | Documents",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()


# ──────────────────────────────────────────────
# Cache & Load Resources
# ──────────────────────────────────────────────

@st.cache_resource
def load_retriever() -> tuple[DocumentRetriever | None, str | None]:
    try:
        retriever = DocumentRetriever()
        return retriever, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error loading index: {e}"


@st.cache_data
def load_metadata_df() -> pd.DataFrame:
    metadata_path = Path(__file__).resolve().parent.parent / "index" / "metadata.json"
    if not metadata_path.exists():
        return pd.DataFrame()
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


@st.cache_data
def load_test_questions() -> list[dict]:
    eval_path = Path(__file__).resolve().parent.parent / "eval" / "test_questions.json"
    if not eval_path.exists():
        return []
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def fetch_tts_audio(text: str, voice: str = "Female") -> bytes | None:
    """Fetch synthesized speech from local GPU TTS microservice."""
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/synthesize",
            json={"text": text, "voice": voice},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        LOGGER.debug("TTS microservice call failed: %s", e)
    return None


def transcribe_audio_via_stt(audio_bytes: bytes, language: str = "auto") -> dict:
    """Send recorded speech bytes to local GPU STT microservice on port 8001."""
    try:
        resp = requests.post(
            f"http://127.0.0.1:8001/transcribe?language={language}",
            data=audio_bytes,
            headers={"Content-Type": "audio/wav"},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"STT server error {resp.status_code}: {resp.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "STT server on port 8001 is unreachable. Please ensure it is running."}
    except Exception as e:
        LOGGER.warning("STT microservice call failed: %s", e)
        return {"error": str(e)}


def render_streaming_tts_player(session_id: str, direct_audio_b64: Optional[str] = None) -> None:
    """Embed Web Audio API real-time queue player connected via WebSocket to local GPU TTS server.
    Includes sequential Promise decoding, buffer time tracking, and direct audio fallback.
    """
    safe_b64 = direct_audio_b64 or ""
    html_code = f"""
    <div id="tts-player-box" style="
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-radius: 8px;
      padding: 6px 12px;
      margin: 6px 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 13px;
      color: #166534;
      display: flex;
      align-items: center;
      justify-content: space-between;
    ">
      <div style="display: flex; align-items: center; gap: 8px;">
        <span id="tts-icon" style="font-size: 16px;">🔊</span>
        <span id="tts-status" style="font-weight: 500;">Streaming voice on GPU...</span>
      </div>
      <button id="tts-btn" onclick="toggleAudio()" style="
        background: #16a34a;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 4px 10px;
        cursor: pointer;
        font-size: 12px;
      ">🔊 Playing</button>
    </div>

    <script>
    (function() {{
      const sessionId = "{session_id}";
      const directB64 = "{safe_b64}";
      const wsUrl = "ws://127.0.0.1:8000/ws_audio/" + sessionId;
      let nextStartTime = 0;
      let ws = null;
      let sentenceCount = 0;
      let playQueue = Promise.resolve();
      let doneScheduled = false;
      let audioReceived = false;

      function getAudioContext() {{
        const parentWin = window.parent || window;
        const AudioContextClass = parentWin.AudioContext || parentWin.webkitAudioContext || window.AudioContext;
        if (!parentWin.__iiitnr_tts_audio_ctx__ || parentWin.__iiitnr_tts_audio_ctx__.state === 'closed') {{
          if (AudioContextClass) {{
            parentWin.__iiitnr_tts_audio_ctx__ = new AudioContextClass();
          }}
        }}
        const ctx = parentWin.__iiitnr_tts_audio_ctx__;
        if (ctx && ctx.state === 'suspended') {{
          ctx.resume();
        }}
        return ctx;
      }}

      getAudioContext();

      window.toggleAudio = function() {{
        const ctx = getAudioContext();
        if (!ctx) return;
        const btn = document.getElementById('tts-btn');
        if (ctx.state === 'running') {{
          ctx.suspend();
          if (btn) {{
            btn.innerText = "▶️ Resume";
            btn.style.background = "#ca8a04";
          }}
        }} else {{
          ctx.resume();
          if (btn) {{
            btn.innerText = "🔊 Playing";
            btn.style.background = "#16a34a";
          }}
        }}
      }};

      function scheduleAudioChunk(arrayBuffer) {{
        audioReceived = true;
        playQueue = playQueue.then(async () => {{
          try {{
            const ctx = getAudioContext();
            if (ctx && ctx.state === 'suspended') {{
              await ctx.resume();
            }}
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);

            const startTime = Math.max(ctx.currentTime, nextStartTime);
            source.start(startTime);
            nextStartTime = startTime + audioBuffer.duration;
            console.log("[TTS] Audio chunk scheduled: dur=" + audioBuffer.duration.toFixed(2) + "s, finishAt=" + nextStartTime.toFixed(2) + "s");
          }} catch (err) {{
            console.error("[TTS] Audio decode error:", err);
          }}
        }});
      }}

      function handlePlaybackDone() {{
        if (doneScheduled) return;
        doneScheduled = true;
        const parentWin = window.parent || window;
        const synth = parentWin.speechSynthesis || window.speechSynthesis;

        function finish() {{
          const el = document.getElementById('tts-status');
          if (el) el.innerText = 'Voice narration complete (' + sentenceCount + ' sentences)';
          const btn = document.getElementById('tts-btn');
          if (btn) {{
            btn.innerText = "✓ Done";
            btn.style.background = "#64748b";
          }}
          if (parentWin.__iiitnr_vad_controller__ && typeof parentWin.__iiitnr_vad_controller__.onBotSpeechCompleted === 'function') {{
            console.log("[TTS] Firing onBotSpeechCompleted()");
            parentWin.__iiitnr_vad_controller__.onBotSpeechCompleted();
          }}
        }}

        if (synth && synth.speaking) {{
          const checkSpeaking = setInterval(() => {{
            if (!synth.speaking) {{
              clearInterval(checkSpeaking);
              finish();
            }}
          }}, 200);
        }} else {{
          playQueue.then(() => {{
            const ctx = getAudioContext();
            const remainingSec = ctx ? Math.max(0, nextStartTime - ctx.currentTime) : 0.5;
            const delayMs = Math.round(remainingSec * 1000) + 200;
            console.log("[TTS] Playback completion scheduled in " + delayMs + "ms");
            setTimeout(finish, delayMs);
          }});
        }}
      }}

      // Fallback: If WebSocket does not deliver chunks within 1200ms and direct base64 audio is available
      if (directB64) {{
        setTimeout(async () => {{
          if (!audioReceived) {{
            console.log("[TTS] WebSocket timed out or empty; playing direct audio fallback...");
            try {{
              const ctx = getAudioContext();
              if (ctx && ctx.state === 'suspended') await ctx.resume();
              const binaryStr = atob(directB64);
              const bytes = new Uint8Array(binaryStr.length);
              for (let i = 0; i < binaryStr.length; i++) {{
                bytes[i] = binaryStr.charCodeAt(i);
              }}
              sentenceCount = 1;
              scheduleAudioChunk(bytes.buffer.slice(0));
              handlePlaybackDone();
            }} catch (e) {{
              console.error("[TTS] Direct fallback error:", e);
            }}
          }}
        }}, 1200);
      }}

      try {{
        ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = function() {{
          const el = document.getElementById('tts-status');
          if (el) el.innerText = "Connected to live voice stream...";
        }};

        ws.onmessage = async function(e) {{
          if (typeof e.data === 'string') {{
            try {{
              const meta = JSON.parse(e.data);
              if (meta.type === 'chunk_meta') {{
                sentenceCount++;
                const el = document.getElementById('tts-status');
                if (el) {{
                  const snippet = meta.text.length > 45 ? meta.text.substring(0, 45) + '...' : meta.text;
                  el.innerText = 'Sentence ' + sentenceCount + ': ' + snippet;
                }}
                if (/[a-zA-Z]{2,}/.test(meta.text) && !/[\u0900-\u097F]/.test(meta.text)) {{
                  // Fallback: only trigger browser Web Speech if server does not stream binary audio chunks
                  setTimeout(() => {{
                    if (!audioReceived) {{
                      try {{
                        const parentWin = window.parent || window;
                        const synth = parentWin.speechSynthesis || window.speechSynthesis;
                        if (synth) {{
                          const utter = new parentWin.SpeechSynthesisUtterance(meta.text);
                          utter.lang = 'en-US';
                          utter.rate = 1.0;
                          synth.speak(utter);
                        }}
                      }} catch (synthErr) {{
                        console.warn('[TTS] SpeechSynthesis error:', synthErr);
                      }}
                    }}
                  }}, 400);
                }}
              }} else if (meta.type === 'done') {{
                handlePlaybackDone();
              }}
            }} catch (err) {{}}
          }} else if (e.data instanceof ArrayBuffer) {{
            scheduleAudioChunk(e.data);
          }}
        }};

        ws.onerror = function(err) {{
          console.warn("[TTS] WebSocket offline or closed:", err);
          if (directB64 && !audioReceived) {{
            try {{
              const ctx = getAudioContext();
              if (ctx && ctx.state === 'suspended') ctx.resume();
              const binaryStr = atob(directB64);
              const bytes = new Uint8Array(binaryStr.length);
              for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
              sentenceCount = 1;
              scheduleAudioChunk(bytes.buffer.slice(0));
              handlePlaybackDone();
            }} catch (e) {{}}
          }}
        }};
      }} catch (err) {{
        console.warn("WebSocket init error:", err);
      }}
    }})();
    </script>
    """
    st.components.v1.html(html_code, height=52)


# ──────────────────────────────────────────────
# Auto-Stop Voice Activity Detection (VAD) Controller
# ──────────────────────────────────────────────

VAD_CONTROLLER_HTML_TEMPLATE = """
<body style="margin: 0; padding: 0; overflow: hidden; background: transparent;">
  <div id="vad-badge" style="
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 12px;
    color: #334155;
    box-sizing: border-box;
    height: 38px;
  ">
    <div style="display: flex; align-items: center; gap: 8px; min-width: 0; flex: 1;">
      <span id="vad-dot" style="
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
        transition: all 0.2s ease;
        flex-shrink: 0;
      "></span>
      <span id="vad-text" style="
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      ">
        ⚡ <b>Auto-Stop VAD:</b> Ready • Press Record, recording auto-stops on silence (__SILENCE_SEC__s)
      </span>
    </div>
    <div id="vad-meter-wrap" style="
      display: none;
      align-items: center;
      gap: 8px;
      width: 130px;
      flex-shrink: 0;
      margin-left: 8px;
    ">
      <div style="
        flex: 1;
        height: 6px;
        background: #e2e8f0;
        border-radius: 3px;
        overflow: hidden;
      ">
        <div id="vad-meter-bar" style="
          width: 0%;
          height: 100%;
          background: #10b981;
          border-radius: 3px;
          transition: width 0.05s ease;
        "></div>
      </div>
      <span id="vad-countdown" style="
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
        min-width: 32px;
        text-align: right;
      "></span>
    </div>
  </div>

  <script>
  (function() {
    const currentEnabled = __ENABLED__;
    const currentSilenceSec = __SILENCE_SEC__;
    const currentCallMode = __CALL_MODE__;
    const parentWin = window.parent || window;

    // 1. Inject Master VAD Script into TOP WINDOW
    if (parentWin.document && parentWin.document.head) {
      try {
        const existing = parentWin.document.getElementById('iiitnr-vad-master-script');
        if (existing) {
          try { existing.remove(); } catch (e) {}
        }
        const s = parentWin.document.createElement('script');
        s.id = 'iiitnr-vad-master-script';
        s.textContent = `
          (function() {
            if (window.__iiitnr_vad_controller__) {
              try {
                if (window.__iiitnr_vad_controller__.pollTimer) clearInterval(window.__iiitnr_vad_controller__.pollTimer);
                if (window.__iiitnr_vad_controller__.analysisTimer) clearInterval(window.__iiitnr_vad_controller__.analysisTimer);
              } catch (e) {}
            }

            const vad = {
              enabled: true,
              silenceSec: 0.9,
              callMode: false,
              state: 'idle',
              audioCtx: null,
              stream: null,
              sourceNode: null,
              analyser: null,
              pollTimer: null,
              analysisTimer: null,
              speechStarted: false,
              speechOnsetCount: 0,
              recordStartTime: 0,
              lastSpeechTime: 0,
              stoppedAt: 0,
              noiseFloor: 0.008,
              onUpdateCb: null,

              configure: function(opts) {
                if (opts.enabled !== undefined) this.enabled = opts.enabled;
                if (opts.silenceSec !== undefined) this.silenceSec = opts.silenceSec;
                if (opts.callMode !== undefined) this.callMode = opts.callMode;
                if (opts.onUpdate !== undefined) this.onUpdateCb = opts.onUpdate;
                this.notify();
              },

              notify: function(extra) {
                if (typeof this.onUpdateCb === 'function') {
                  try {
                    this.onUpdateCb({
                      state: this.state,
                      enabled: this.enabled,
                      silenceSec: this.silenceSec,
                      callMode: this.callMode,
                      speechStarted: this.speechStarted,
                      ...(extra || {})
                    });
                  } catch (e) {
                    this.onUpdateCb = null;
                  }
                }
              },

              getAudioContext: function() {
                const AudioCtx = window.AudioContext || window.webkitAudioContext;
                if (!this.audioCtx || this.audioCtx.state === 'closed') {
                  this.audioCtx = new AudioCtx();
                }
                if (this.audioCtx.state === 'suspended' || this.audioCtx.state === 'interrupted') {
                  this.audioCtx.resume();
                }
                return this.audioCtx;
              },

              getStopButton: function() {
                try {
                  let btn = document.querySelector('[data-testid="stAudioInputActionButton"][aria-label*="Stop"]');
                  if (btn) return btn;
                  btn = document.querySelector('[data-testid="stAudioInput"] button[aria-label*="Stop"]');
                  if (btn) return btn;
                  const allBtns = document.querySelectorAll('[data-testid="stAudioInput"] button');
                  for (let i = 0; i < allBtns.length; i++) {
                    const b = allBtns[i];
                    const l = (b.getAttribute('aria-label') || b.getAttribute('title') || b.innerText || '').toLowerCase();
                    if (l.includes('stop')) return b;
                  }
                } catch (e) {}
                return null;
              },

              getRecordButton: function() {
                try {
                  // Check if there is an existing recorded audio Clear/Delete button and click it to reset
                  const clearBtn = document.querySelector(
                    '[data-testid="stAudioInput"] button[aria-label*="Clear"], ' +
                    '[data-testid="stAudioInput"] button[aria-label*="Delete"], ' +
                    '[data-testid="stAudioInput"] button[aria-label*="Remove"]'
                  );
                  if (clearBtn) {
                    console.log("[IIITNR-VAD] Resetting audio input widget via Clear button...");
                    try { clearBtn.click(); } catch (e) {}
                    return null;
                  }

                  // Exact Record button - strictly exclude any button with 'stop'
                  let btn = document.querySelector('[data-testid="stAudioInputActionButton"][aria-label="Record"]');
                  if (btn) return btn;
                  btn = document.querySelector('[data-testid="stAudioInput"] button[aria-label="Record"]');
                  if (btn) return btn;

                  const allBtns = document.querySelectorAll('[data-testid="stAudioInput"] button');
                  for (let i = 0; i < allBtns.length; i++) {
                    const b = allBtns[i];
                    const l = (b.getAttribute('aria-label') || b.getAttribute('title') || b.innerText || '').toLowerCase();
                    if (l.includes('record') && !l.includes('stop') && !l.includes('pause') && !l.includes('play')) {
                      return b;
                    }
                  }
                } catch (e) {}
                return null;
              },

              clickRecord: function(recordBtn) {
                if (this.state === 'listening' || this.state === 'speaking' || this.state === 'stopping') return;
                const l = (recordBtn.getAttribute('aria-label') || '').toLowerCase();
                if (l.includes('stop')) return; // Safety: never trigger Stop as Record!

                console.log("[IIITNR-VAD] Activating microphone for user turn...");
                try {
                  recordBtn.click();
                } catch (e) {
                  console.warn("[IIITNR-VAD] recordBtn.click error:", e);
                }
              },

              onBotSpeechCompleted: function() {
                if (!this.callMode) return;
                console.log("[IIITNR-VAD] Bot finished speaking in Call Mode. Handing turn to user in 600ms...");
                this.notify({ state: 'call_transition', message: 'Bot finished speaking • Opening microphone...' });

                setTimeout(() => {
                  if (!this.callMode) return;
                  const recordBtn = this.getRecordButton();
                  if (recordBtn) {
                    this.clickRecord(recordBtn);
                  } else {
                    // Retry in 350ms if Streamlit was clearing or remounting widget
                    setTimeout(() => {
                      if (!this.callMode) return;
                      const retryBtn = this.getRecordButton();
                      if (retryBtn) this.clickRecord(retryBtn);
                    }, 350);
                  }
                }, 600);
              },

              start: function() {
                if (this.pollTimer) clearInterval(this.pollTimer);
                this.pollTimer = setInterval(() => this.poll(), 100);
                console.log("[IIITNR-VAD] Top-window master controller initialized and polling.");
              },

              poll: function() {
                const stopBtn = this.getStopButton();

                if (this.state === 'stopping') {
                  if (!stopBtn || (Date.now() - this.stoppedAt > 1500)) {
                    this.endSession();
                  }
                  return;
                }

                if (stopBtn) {
                  if (this.state === 'idle') {
                    if (this.enabled) {
                      this.startSession(stopBtn);
                    } else {
                      this.notify({ state: 'manual' });
                    }
                  }
                } else {
                  if (this.state !== 'idle') {
                    this.endSession();
                  }
                }
              },

              startSession: async function(stopBtn) {
                if (this.state === 'listening' || this.state === 'speaking' || this.state === 'silence') return;

                this.state = 'listening';
                this.speechStarted = false;
                this.speechOnsetCount = 0;
                this.recordStartTime = Date.now();
                this.lastSpeechTime = 0;
                this.noiseFloor = 0.008;
                this.notify({ state: 'listening' });

                try {
                  let stream = this.stream;
                  let isStreamLive = stream && stream.active && stream.getAudioTracks().some(t => t.readyState === 'live');

                  if (!isStreamLive) {
                    stream = await navigator.mediaDevices.getUserMedia({
                      audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                      }
                    });
                    this.stream = stream;
                  }

                  const ctx = this.getAudioContext();
                  if (ctx.state !== 'running') {
                    await ctx.resume();
                  }

                  if (this.sourceNode) {
                    try { this.sourceNode.disconnect(); } catch (e) {}
                  }
                  if (this.analyser) {
                    try { this.analyser.disconnect(); } catch (e) {}
                  }

                  this.sourceNode = ctx.createMediaStreamSource(this.stream);
                  this.analyser = ctx.createAnalyser();
                  this.analyser.fftSize = 512;
                  this.analyser.smoothingTimeConstant = 0.2;

                  this.sourceNode.connect(this.analyser);

                  this.startAnalysis(stopBtn);
                } catch (err) {
                  console.warn("[IIITNR-VAD] startSession error:", err);
                  this.cleanup();
                  this.notify({ state: 'error', message: 'Microphone access notice' });
                }
              },

              startAnalysis: function(stopBtn) {
                if (this.analysisTimer) clearInterval(this.analysisTimer);

                const analyser = this.analyser;
                if (!analyser) return;

                const buffer = new Float32Array(analyser.fftSize);
                let frame = 0;

                this.analysisTimer = setInterval(() => {
                  const currentBtn = this.getStopButton();
                  if (!currentBtn || this.state === 'stopping' || this.state === 'idle') {
                    this.endSession();
                    return;
                  }

                  if (!this.analyser) return;

                  try {
                    this.analyser.getFloatTimeDomainData(buffer);
                  } catch (e) {
                    return;
                  }

                  let sum = 0;
                  for (let i = 0; i < buffer.length; i++) {
                    sum += buffer[i] * buffer[i];
                  }
                  const rms = Math.sqrt(sum / buffer.length);
                  const now = Date.now();
                  const elapsed = now - this.recordStartTime;

                  frame++;
                  if (frame < 15 && !this.speechStarted) {
                    this.noiseFloor = Math.max(0.003, Math.min(0.020, (this.noiseFloor * 0.8) + (rms * 0.2)));
                  }

                  const speechThresh = Math.max(0.022, this.noiseFloor * 2.2);

                  if (elapsed >= 60000) {
                    this.triggerStop(currentBtn);
                    return;
                  }

                  if (rms >= speechThresh) {
                    this.speechOnsetCount++;
                    if (this.speechOnsetCount >= 2) {
                      this.speechStarted = true;
                      this.lastSpeechTime = now;
                      this.state = 'speaking';
                      this.notify({ state: 'speaking', rms: rms });
                    }
                  } else {
                    this.speechOnsetCount = 0;
                    if (this.speechStarted) {
                      const silenceMs = now - this.lastSpeechTime;
                      const targetMs = this.silenceSec * 1000;
                      const remaining = Math.max(0, (targetMs - silenceMs) / 1000).toFixed(1);

                      this.state = 'silence';
                      this.notify({
                        state: 'silence',
                        rms: rms,
                        silenceMs: silenceMs,
                        remainingSec: remaining
                      });

                      // Minimum 1000ms total recording before stopping on silence
                      // to ensure MediaRecorder produces a valid, non-empty WebM Opus container
                      if (silenceMs >= targetMs && elapsed >= 1000) {
                        console.log("[IIITNR-VAD] Silence limit reached (" + this.silenceSec + "s after " + elapsed + "ms). Auto-stopping!");
                        this.triggerStop(currentBtn);
                        return;
                      }
                    } else {
                      this.state = 'listening';
                      if (elapsed >= 15000) {
                        this.triggerStop(currentBtn);
                        return;
                      }
                      this.notify({ state: 'listening', rms: rms });
                    }
                  }
                }, 40);
              },

              triggerStop: function(btn) {
                if (this.state === 'stopping') return;
                const now = Date.now();
                if (this.recordStartTime && (now - this.recordStartTime < 900)) {
                  console.log("[IIITNR-VAD] Waiting for minimum audio length before stop...");
                  return;
                }

                this.state = 'stopping';
                this.stoppedAt = now;
                this.notify({ state: 'stopping' });

                console.log("[IIITNR-VAD] Auto-stopping recording button...");
                try {
                  btn.click();
                } catch (e) {
                  console.warn("[IIITNR-VAD] btn.click error:", e);
                }

                if (this.analysisTimer) {
                  clearInterval(this.analysisTimer);
                  this.analysisTimer = null;
                }
                if (this.sourceNode) {
                  try { this.sourceNode.disconnect(); } catch (e) {}
                  this.sourceNode = null;
                }
                if (this.analyser) {
                  try { this.analyser.disconnect(); } catch (e) {}
                  this.analyser = null;
                }
              },

              cleanup: function() {
                if (this.analysisTimer) {
                  clearInterval(this.analysisTimer);
                  this.analysisTimer = null;
                }
                if (this.sourceNode) {
                  try { this.sourceNode.disconnect(); } catch (e) {}
                  this.sourceNode = null;
                }
                if (this.analyser) {
                  try { this.analyser.disconnect(); } catch (e) {}
                  this.analyser = null;
                }
              },

              endSession: function() {
                this.cleanup();
                this.state = 'idle';
                this.speechStarted = false;
                this.speechOnsetCount = 0;
                this.notify({ state: 'idle' });
              }
            };

            window.__iiitnr_vad_controller__ = vad;
            vad.start();

            // Auto-unlock AudioContext on user gesture anywhere in top window
            const unlockAudio = function() {
              const AudioCtx = window.AudioContext || window.webkitAudioContext;
              if (!window.__iiitnr_tts_audio_ctx__ || window.__iiitnr_tts_audio_ctx__.state === 'closed') {
                if (AudioCtx) window.__iiitnr_tts_audio_ctx__ = new AudioCtx();
              }
              if (window.__iiitnr_tts_audio_ctx__ && window.__iiitnr_tts_audio_ctx__.state === 'suspended') {
                window.__iiitnr_tts_audio_ctx__.resume();
              }
            };
            ['click', 'touchstart', 'keydown'].forEach(evt => {
              window.addEventListener(evt, unlockAudio, { passive: true });
            });
          })();
        `;
        parentWin.document.head.appendChild(s);
      } catch (err) {
        console.warn("[IIITNR-VAD] Master injection notice:", err);
      }
    }

    function updateWidgetUI(data) {
      const dot = document.getElementById('vad-dot');
      const text = document.getElementById('vad-text');
      const meterWrap = document.getElementById('vad-meter-wrap');
      const meterBar = document.getElementById('vad-meter-bar');
      const countdown = document.getElementById('vad-countdown');
      if (!dot || !text) return;

      if (!data.enabled) {
        dot.style.background = '#94a3b8';
        dot.style.boxShadow = 'none';
        text.innerHTML = '⚪ <b>Manual Mode:</b> Auto-Stop is off • Press Stop button when done';
        if (meterWrap) meterWrap.style.display = 'none';
        return;
      }

      if (data.callMode) {
        if (data.state === 'call_transition') {
          dot.style.background = '#8b5cf6';
          dot.style.boxShadow = '0 0 0 4px rgba(139, 92, 246, 0.4)';
          text.innerHTML = '⏳ <b>Call Mode:</b> She finished speaking • Opening microphone...';
          if (meterWrap) meterWrap.style.display = 'none';
          return;
        } else if (data.state === 'listening') {
          dot.style.background = '#22c55e';
          dot.style.boxShadow = '0 0 0 4px rgba(34, 197, 94, 0.4)';
          text.innerHTML = '🎙️ <b>Call Active: Your Turn</b> • Speak your question now, she is listening...';
          if (meterWrap) {
            meterWrap.style.display = 'flex';
            const pct = Math.min(100, Math.round((data.rms || 0) * 800));
            if (meterBar) {
              meterBar.style.width = pct + '%';
              meterBar.style.background = '#22c55e';
            }
            if (countdown) countdown.innerText = 'CALL';
          }
          return;
        }
      }

      switch (data.state) {
        case 'idle':
          dot.style.background = '#10b981';
          dot.style.boxShadow = '0 0 0 2px rgba(16, 185, 129, 0.2)';
          text.innerHTML = (data.callMode ? '📞 <b>Call Mode:</b> Connected • Ready' : '⚡ <b>Auto-Stop VAD:</b> Ready • Press Record, stops on silence (' + data.silenceSec + 's)');
          if (meterWrap) meterWrap.style.display = 'none';
          break;

        case 'listening':
          dot.style.background = '#3b82f6';
          dot.style.boxShadow = '0 0 0 4px rgba(59, 130, 246, 0.3)';
          text.innerHTML = '🎙️ <b>Listening...</b> Speak your question now';
          if (meterWrap) {
            meterWrap.style.display = 'flex';
            const pct = Math.min(100, Math.round((data.rms || 0) * 800));
            if (meterBar) {
              meterBar.style.width = pct + '%';
              meterBar.style.background = '#3b82f6';
            }
            if (countdown) countdown.innerText = 'WAIT';
          }
          break;

        case 'speaking':
          dot.style.background = '#22c55e';
          dot.style.boxShadow = '0 0 0 4px rgba(34, 197, 94, 0.4)';
          text.innerHTML = '🗣️ <b>Voice detected:</b> Speaking...';
          if (meterWrap) {
            meterWrap.style.display = 'flex';
            const pct = Math.min(100, Math.round((data.rms || 0) * 700));
            if (meterBar) {
              meterBar.style.width = Math.max(15, pct) + '%';
              meterBar.style.background = '#22c55e';
            }
            if (countdown) countdown.innerText = 'MIC';
          }
          break;

        case 'silence':
          dot.style.background = '#f59e0b';
          dot.style.boxShadow = '0 0 0 4px rgba(245, 158, 11, 0.3)';
          text.innerHTML = '⏳ <b>Speech paused:</b> Stopping in ' + data.remainingSec + 's...';
          if (meterWrap) {
            meterWrap.style.display = 'flex';
            if (meterBar) {
              const targetMs = (data.silenceSec || 1.4) * 1000;
              const progress = Math.min(100, Math.round(((data.silenceMs || 0) / targetMs) * 100));
              meterBar.style.width = progress + '%';
              meterBar.style.background = '#f59e0b';
            }
            if (countdown) countdown.innerText = data.remainingSec + 's';
          }
          break;

        case 'stopping':
          dot.style.background = '#8b5cf6';
          dot.style.boxShadow = '0 0 0 4px rgba(139, 92, 246, 0.4)';
          text.innerHTML = '⏹️ <b>Auto-stopped!</b> Submitting question to AI...';
          if (meterWrap) {
            meterWrap.style.display = 'flex';
            if (meterBar) {
              meterBar.style.width = '100%';
              meterBar.style.background = '#8b5cf6';
            }
            if (countdown) countdown.innerText = 'DONE';
          }
          break;

        case 'error':
          dot.style.background = '#ef4444';
          dot.style.boxShadow = '0 0 0 2px rgba(239, 68, 68, 0.2)';
          text.innerHTML = '⚠️ <b>Auto-Stop:</b> ' + (data.message || 'Microphone access notice');
          if (meterWrap) meterWrap.style.display = 'none';
          break;
      }
    }

    // 2. Configure master controller with current session options and local UI updater
    if (parentWin.__iiitnr_vad_controller__) {
      parentWin.__iiitnr_vad_controller__.configure({
        enabled: currentEnabled,
        silenceSec: currentSilenceSec,
        callMode: currentCallMode,
        onUpdate: updateWidgetUI
      });
    } else {
      updateWidgetUI({ enabled: currentEnabled, state: 'idle', silenceSec: currentSilenceSec, callMode: currentCallMode });
    }
  })();
  </script>
</body>
"""


def render_vad_controller(enabled: bool = True, silence_sec: float = 0.9, call_mode: bool = False) -> None:
    """Embed client-side Voice Activity Detection (VAD) Auto-Stop and Call Mode controller.

    Watches the Streamlit audio_input widget. When speech begins and is followed by
    `silence_sec` seconds of quiet, it automatically triggers the stop-recording button
    without requiring the user to manually click 'Stop recording'.
    When call_mode is True, it automatically triggers recording when the bot finishes speaking.
    """
    html_code = (
        VAD_CONTROLLER_HTML_TEMPLATE.replace("__ENABLED__", "true" if enabled else "false")
        .replace("__SILENCE_SEC__", str(round(float(silence_sec), 1)))
        .replace("__CALL_MODE__", "true" if call_mode else "false")
    )
    st.components.v1.html(html_code, height=44)



# ──────────────────────────────────────────────
# Retrieval Debug Panel
# ──────────────────────────────────────────────

def outcome_to_debug(outcome: RetrievalOutcome) -> dict:
    """Summarize retrieval outcome into session-safe dictionary."""
    used = {candidate.chunk_index for candidate in outcome.contexts}

    many_documents = len({
        c.source_document for c in outcome.candidates if c.source_document
    }) > 1

    rows = []
    for candidate in outcome.candidates:
        rows.append({
            "Used": "✅" if candidate.chunk_index in used else "",
            "Chunk": candidate.chunk_index,
            **({"Document": candidate.document_label[:24]} if many_documents else {}),
            "Page": candidate.page_label,
            "Section": candidate.heading_path_label or "-",
            "Retrieved by": candidate.retrieved_by,
            "Dense (cosine)": round(candidate.dense_score, 3),
            "BM25": round(candidate.lexical_score, 2),
            "RRF": round(candidate.fusion_score, 5),
            "Rerank (0-10)": (
                round(candidate.rerank_score, 1)
                if candidate.rerank_score is not None else None
            ),
            "Term coverage": f"{candidate.lexical_coverage:.0%}",
            "Matched terms": ", ".join(candidate.matched_terms) or "-",
        })

    return {
        "rows": rows,
        "gate": dict(outcome.gate),
        "timings_ms": dict(outcome.timings_ms),
        "effective_query": outcome.effective_query,
        "was_rewritten": outcome.was_rewritten,
        "refused": outcome.refused,
        "refusal_reason": outcome.refusal_reason,
        "rerank_failed": outcome.rerank_failed,
    }


def render_retrieval_debug(debug: dict) -> None:
    """Render retrieval metrics and score breakdown across pipeline stages."""
    if not debug:
        return

    rows = debug.get("rows") or []
    label = f"🔬 Retrieval Details ({len(rows)} candidates)"

    with st.expander(label, expanded=False):
        timings = debug.get("timings_ms") or {}
        if timings:
            stage_names = {
                "rewrite_ms": "Query rewriting",
                "dense_ms": "Dense (embed + FAISS)",
                "lexical_ms": "BM25",
                "fusion_ms": "RRF Fusion",
                "rerank_ms": "LLM Reranker",
            }
            cols = st.columns(len(timings) + 1)
            for col, (key, value) in zip(cols, timings.items()):
                col.metric(stage_names.get(key, key), f"{value:.0f} ms")
            cols[-1].metric("Total", f"{sum(timings.values()):.0f} ms")

        gate = debug.get("gate") or {}
        if gate:
            st.caption(
                "**Relevance Gate** — "
                f"max similarity {gate.get('max_dense_score', 0):.3f} "
                f"(threshold {gate.get('dense_threshold', 0):.2f}) · "
                f"max term coverage {gate.get('max_lexical_coverage', 0):.0%} "
                f"(threshold {gate.get('lexical_coverage_threshold', 0):.0%}) · "
                f"dense branch {'passed' if gate.get('dense_pass') else 'failed'}, "
                f"lexical branch {'passed' if gate.get('lexical_pass') else 'failed'}"
                + (
                    f", reranker {'passed' if gate.get('rerank_pass') else 'rejected'}"
                    if "rerank_pass" in gate else ""
                )
            )

        if debug.get("rerank_failed"):
            st.error(
                "Reranker failed to execute; fusion rank order was used and "
                "precision gate was skipped for this question. "
                "This usually happens when the API is busy — try again.",
                icon="🔥",
            )

        if debug.get("refused"):
            st.warning(f"Refused: {debug.get('refusal_reason', '-')}")

        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("No candidates found in either search branch.")


def render_index_freshness(retriever: DocumentRetriever | None) -> None:
    """Warn when the index is out of sync with source documents."""
    if not retriever:
        return

    freshness = retriever.freshness()
    if not freshness.has_info or not freshness.is_stale:
        return

    pesan = "\n\n".join(f"- {m}" for m in freshness.messages())
    st.warning(
        f"**Index may be outdated.**\n\n{pesan}\n\n"
        "Run `python ingestion/build_index.py --yes` to rebuild.",
        icon="🕒",
    )


def render_hero() -> None:
    """Hero welcome card matching campus academic portal styling."""
    render_welcome(
        "Welcome to the IIIT-NR Academic Chatbot",
        "Ask questions about <strong>IIIT-NR Academic Guidelines, Faculty Profiles, Admissions, Curriculum, and Campus Information</strong>. "
        "Every answer is constructed strictly from official documents and includes "
        "page numbers and section titles so you can verify each detail directly.",
    )


# ──────────────────────────────────────────────
# View 1: Q&A Chat
# ──────────────────────────────────────────────

def render_chat_view(
    retriever: DocumentRetriever | None,
    config: RetrievalConfig,
    operator_mode: bool = False,
    rate_limiter: SessionRateLimiter | None = None,
) -> None:
    st.header("Academic Q&A")
    render_lead(
        "Ask questions in English about academic policies, credit requirements, "
        "thesis guidelines, leave of absence, or graduation."
    )

    if not retriever:
        if operator_mode:
            st.error(
                "Document index not found. Run "
                "`python ingestion/build_index.py` first."
            )
        else:
            st.error(
                "Chatbot service is being prepared. Please try again later "
                "or contact the administrator."
            )
        return

    # Preset Questions Picker
    preset_col, action_col = st.columns([3, 1])
    sample_queries = [
        "-- Select a sample question --",
        "Who is Dr. Abhishek Sharma and what is his research area?",
        "What are the admission requirements for B.Tech at IIIT-NR?",
        "What are the placement statistics and top recruiters?",
        "What is the curriculum and credit structure for B.Tech CSE?",
        "What campus facilities and research laboratories are available?",
    ]
    with preset_col:
        selected_sample = st.selectbox(
            "Sample questions",
            options=sample_queries,
            help="Select a common question to test the chatbot workflow directly.",
        )
    with action_col:
        st.write("")
        st.write("")
        if st.button("Send question", type="primary", disabled=(selected_sample == sample_queries[0])):
            st.session_state["queued_query"] = selected_sample
            st.rerun()

    call_mode_active = st.session_state.get("call_mode_active", False)

    # 📞 Two-Way Voice Call Banner, Assistant Language Mode & Controller
    call_col1, call_col2, call_col3 = st.columns([2.5, 1.4, 1.1])
    with call_col1:
        if call_mode_active:
            st.info("📞 **Interactive Voice Call Active** • She speaks → You speak → She answers → Repeat.", icon="🟢")
        else:
            st.caption("💡 **Tip:** Start **Voice Call Mode** for a phone-call experience or ask questions below.")
    with call_col2:
        lang_mode_options = [
            "🌐 Auto-Detect (Sticky)",
            "🌾 Chhattisgarhi (Always)",
            "🇮🇳 Hindi (Always)",
            "🇬🇧 English (Always)",
        ]
        lang_mode_keys = {
            "🌐 Auto-Detect (Sticky)": "auto",
            "🌾 Chhattisgarhi (Always)": "chhattisgarhi",
            "🇮🇳 Hindi (Always)": "hindi",
            "🇬🇧 English (Always)": "english",
        }
        current_lock = st.session_state.get("assistant_language_lock", "auto")
        inv_map = {v: k for k, v in lang_mode_keys.items()}
        current_label = inv_map.get(current_lock, "🌐 Auto-Detect (Sticky)")
        sel_label = st.selectbox(
            "🗣️ Assistant Language",
            options=lang_mode_options,
            index=lang_mode_options.index(current_label) if current_label in lang_mode_options else 0,
            key="assistant_lang_mode_widget",
            help="Lock the assistant's voice and text language. 'Auto-Detect' maintains conversation language stickiness across turns.",
        )
        st.session_state["assistant_language_lock"] = lang_mode_keys[sel_label]
    with call_col3:
        if not call_mode_active:
            if st.button("📞 Start Call", type="primary", use_container_width=True):
                st.session_state["call_mode_active"] = True
                st.session_state["queued_call_start"] = True
                st.session_state["vad_auto_stop"] = True
                st.session_state["audio_input_counter"] = st.session_state.get("audio_input_counter", 0) + 1
                st.rerun()
        else:
            if st.button("🔴 End Call", type="secondary", use_container_width=True):
                st.session_state["call_mode_active"] = False
                st.session_state["queued_call_start"] = False
                st.rerun()

    # Session messages initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # If call just started, she speaks first in the selected language!
    if st.session_state.pop("queued_call_start", False):
        active_lang = st.session_state.get("assistant_language_lock", "auto")
        if active_lang == "chhattisgarhi":
            greeting_text = (
                "जय जोहार! मैं आईआईआईटी नया रायपुर के अकादमिक सहायिका आंव। "
                "आप मन मोले प्रवेश, पाठ्यक्रम या परिसर के बारे म कुछु भी पूछ सकथो। "
                "आप मन का जानना चाहथो?"
            )
            st.session_state["active_conversation_language"] = "chhattisgarhi"
        elif active_lang == "english":
            greeting_text = (
                "Hello! I am the academic assistant for IIIT Naya Raipur. "
                "You can ask me anything about admissions, courses, curriculum, or campus life. "
                "How may I help you today?"
            )
            st.session_state["active_conversation_language"] = "english"
        else:
            greeting_text = (
                "नमस्ते! मैं ट्रिपल आई टी नया रायपुर की अकादमिक सहायिका हूँ। "
                "आप मुझसे प्रवेश, पाठ्यक्रम या परिसर के बारे में कुछ भी पूछ सकते हैं। "
                "आप क्या जानना चाहते हैं?"
            )
            st.session_state["active_conversation_language"] = "hindi"

        greeting_session_id = uuid.uuid4().hex
        voice_gender = st.session_state.get("voice_gender", "Female")

        try:
            requests.post(
                "http://127.0.0.1:8000/stream_sentence",
                json={"session_id": greeting_session_id, "text": greeting_text, "voice": voice_gender},
                timeout=5,
            )
            res = requests.post(
                "http://127.0.0.1:8000/stream_finish",
                json={"session_id": greeting_session_id},
                timeout=10,
            )
            audio_data = None
            if res.status_code == 200:
                b64_audio = res.json().get("audio_base64")
                if b64_audio:
                    audio_data = base64.b64decode(b64_audio)
        except Exception as e:
            LOGGER.warning("TTS greeting streaming failed: %s", e)
            audio_data = None

        st.session_state.messages.append({
            "role": "assistant",
            "content": greeting_text,
            "audio": audio_data,
            "streaming_session_id": greeting_session_id,
        })
        st.rerun()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("streaming_session_id"):
                    b64_audio = base64.b64encode(msg["audio"]).decode("utf-8") if msg.get("audio") else None
                    render_streaming_tts_player(msg.pop("streaming_session_id"), direct_audio_b64=b64_audio)
                elif msg.get("audio"):
                    st.audio(msg["audio"], format="audio/wav")
                if msg.get("source_refs"):
                    render_source_references(
                        msg["source_refs"],
                        key_prefix=f"history_{msg.get('message_id', id(msg))}",
                        public=not operator_mode,
                    )
                elif msg.get("sources"):
                    st.markdown(
                        f'<div class="source-box"><strong>📚 Document Sources:</strong><br>{msg["sources"]}</div>',
                        unsafe_allow_html=True,
                    )
                if operator_mode and msg.get("debug"):
                    render_retrieval_debug(msg["debug"])

    # 🎙️ Voice Input (Chhattisgarhi / Hindi / English)
    with st.expander("🎙️ Speak your question (Voice Input)", expanded=call_mode_active):
        v_col1, v_col2 = st.columns([2, 1])
        with v_col2:
            current_assistant_lock = st.session_state.get("assistant_language_lock", "auto")
            lock_to_stt_idx = {
                "auto": 0,
                "chhattisgarhi": 1,
                "hindi": 2,
                "english": 3,
            }
            default_stt_idx = lock_to_stt_idx.get(current_assistant_lock, 0)
            stt_lang_choice = st.selectbox(
                "Voice Language",
                options=[
                    "🌐 Auto (English / Hindi / Hinglish)",
                    "🌾 Chhattisgarhi (छत्तीसगढ़ी - MMS)",
                    "🇮🇳 Hindi (हिन्दी - Whisper)",
                    "🇬🇧 English (Whisper)",
                ],
                index=default_stt_idx,
                help="Auto automatically detects English or Hindi. Choose Chhattisgarhi for native Meta MMS 1B model.",
            )
            lang_code_map = {
                "🌐 Auto (English / Hindi / Hinglish)": "auto",
                "🌾 Chhattisgarhi (छत्तीसगढ़ी - MMS)": "hne",
                "🇮🇳 Hindi (हिन्दी - Whisper)": "hi",
                "🇬🇧 English (Whisper)": "en",
            }
            target_lang = lang_code_map[stt_lang_choice]

            auto_stop_enabled = st.toggle(
                "⚡ Auto-Stop on Silence",
                value=st.session_state.get("vad_auto_stop", True),
                key="vad_auto_stop_toggle",
                help="Automatically stop recording when speech finishes (VAD auto-cutoff). No need to press Stop!",
            )
            st.session_state["vad_auto_stop"] = auto_stop_enabled

            if auto_stop_enabled:
                silence_cutoff = st.select_slider(
                    "Silence Delay",
                    options=[0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 2.0],
                    value=st.session_state.get("vad_silence_delay", 0.9),
                    key="vad_silence_slider",
                    format_func=lambda x: f"{x}s pause",
                    help="Time to wait after speaking before automatically stopping.",
                )
                st.session_state["vad_silence_delay"] = silence_cutoff
            else:
                silence_cutoff = 0.9

        with v_col1:
            mic_counter = st.session_state.get("audio_input_counter", 0)
            recorded_audio = st.audio_input(
                "Record your question from microphone:",
                key=f"mic_input_{mic_counter}",
            )
            render_vad_controller(
                enabled=auto_stop_enabled,
                silence_sec=silence_cutoff,
                call_mode=call_mode_active,
            )

        if recorded_audio is not None:
            raw_audio = recorded_audio.getvalue()
            audio_hash = hash(raw_audio)
            if st.session_state.get("last_processed_audio_hash") != audio_hash:
                st.session_state["last_processed_audio_hash"] = audio_hash
                with st.spinner("🎧 Transcribing speech on GPU..."):
                    result = transcribe_audio_via_stt(raw_audio, language=target_lang)
                    if result and result.get("error"):
                        st.error(f"❌ {result['error']}")
                    elif result and result.get("text"):
                        recognized_text = result["text"].strip()
                        det_lang = result.get("detected_language", target_lang)
                        inf_ms = result.get("inference_ms", 0)
                        st.session_state["last_audio_debug"] = {
                            "raw_audio": raw_audio,
                            "processed_b64": result.get("processed_audio_b64", ""),
                            "text": recognized_text,
                            "lang": det_lang,
                            "inf_ms": inf_ms,
                            "log_id": result.get("log_id", ""),
                            "raw_path": result.get("raw_audio_path", ""),
                            "proc_path": result.get("processed_audio_path", ""),
                            "hardware_gate_warning": result.get("hardware_gate_warning", ""),
                        }
                        st.toast(f"🗣️ Transcribed [{det_lang.upper()} in {inf_ms}ms]: {recognized_text}", icon="🎙️")
                        st.session_state["queued_query"] = recognized_text
                        st.session_state["last_audio_language"] = target_lang if target_lang != "auto" else det_lang
                        st.session_state["audio_input_counter"] = mic_counter + 1
                        st.rerun()
                    else:
                        st.warning("⚠️ No clear speech was detected. Please make sure your microphone is allowed/unmuted and speak closer to the mic.")

        # Audio Inspection Player (Compare Raw vs. Processed Audio)
        if "last_audio_debug" in st.session_state:
            dbg = st.session_state["last_audio_debug"]
            st.markdown("---")
            st.markdown("🎧 **Voice Inspector: Compare Raw vs. Processed Audio**")
            col_raw, col_proc = st.columns(2)
            with col_raw:
                st.caption("🎙️ **Raw Microphone Audio** (Direct from browser)")
                st.audio(dbg["raw_audio"], format="audio/wav")
            with col_proc:
                st.caption("🎛️ **Processed Audio** (16kHz, Normalized, Click-filtered)")
                if dbg.get("processed_b64"):
                    st.audio(base64.b64decode(dbg["processed_b64"]), format="audio/wav")
                else:
                    st.caption("*(Processed audio unavailable)*")
            if dbg.get("hardware_gate_warning"):
                st.warning(f"🔇 **Hardware/Browser Cutoff Detected:** {dbg['hardware_gate_warning']}\n\n"
                           f"👉 **Fix:** Turn off *Audio Enhancements* in Windows Sound Settings or disable *Brave Shields* for localhost.")
            if dbg.get("log_id"):
                st.caption(f"💾 **Logged to Disk:** `{dbg['log_id']}` in `STT/stt-service/logs/audio/`")

    # Process queued query from preset if any
    input_query = None
    if "queued_query" in st.session_state:
        input_query = st.session_state.pop("queued_query")

    chat_input = st.chat_input("Type your question here...")
    query = input_query or chat_input

    if query:
        _handle_query(
            query,
            retriever,
            config,
            operator_mode=operator_mode,
            rate_limiter=rate_limiter,
        )

    render_pdf_viewer()


def _handle_query(
    query: str,
    retriever: DocumentRetriever,
    config: RetrievalConfig,
    operator_mode: bool = False,
    rate_limiter: SessionRateLimiter | None = None,
) -> None:
    if rate_limiter is not None:
        limit_result = rate_limiter.check_and_consume(st.session_state)
        if not limit_result.allowed:
            minutes = max(1, (limit_result.retry_after_seconds + 59) // 60)
            st.warning(
                f"Question rate limit reached "
                f"({limit_result.limit} questions per hour). "
                f"Please try again in about {minutes} minute(s).",
                icon="⏳",
            )
            return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(query)

    # Chat history excluding latest question
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant", avatar="🎓"):
        from app.query_rewriter import needs_rewrite
        spinner_text = (
            "Translating/Rewriting question and searching documents..."
            if config.use_query_rewrite and needs_rewrite(query, chat_history)
            else "Searching relevant document excerpts..."
        )

        try:
            with st.spinner(spinner_text):
                outcome = retriever.retrieve(
                    query,
                    config=config,
                    chat_history=chat_history,
                )
        except LLMConfigError as error:
            if operator_mode:
                st.error(f"⚠️ {error}")
            else:
                st.error(
                    "Search service is not ready. Please contact administrator.",
                    icon="⚠️",
                )
            st.session_state.messages.pop()
            return
        except Exception as error:
            if operator_mode:
                st.error(f"⚠️ Search failed: {error}")
            else:
                st.error(
                    "Search service encountered an error. "
                    "Please try again later.",
                    icon="⚠️",
                )
            st.session_state.messages.pop()
            return

        debug = outcome_to_debug(outcome)

        if outcome.was_rewritten and operator_mode:
            st.caption(
                f"🔁 Question rewritten for search: "
                f"*{outcome.effective_query}*"
            )

        audio_lang = st.session_state.pop("last_audio_language", None)
        lang_lock = st.session_state.get("assistant_language_lock", "auto")
        effective_hint = lang_lock if lang_lock != "auto" else audio_lang
        session_lang = st.session_state.get("active_conversation_language")
        detected_lang = detect_query_language(
            outcome.original_query,
            hint_lang=effective_hint,
            session_lang=session_lang,
        )
        st.session_state["active_conversation_language"] = detected_lang

        if operator_mode:
            st.caption(f"🗣️ Detected query language: **{detected_lang.capitalize()}**")

        if outcome.refused:
            response = build_no_context_response(language=detected_lang)
            st.markdown(response)
            if operator_mode:
                render_retrieval_debug(debug)

            enable_tts = st.session_state.get("enable_tts", True)
            voice_gender = st.session_state.get("voice_gender", "Female")
            refusal_session_id = uuid.uuid4().hex if enable_tts else None
            audio_data = None
            b64_audio = None

            if enable_tts and refusal_session_id:
                try:
                    requests.post(
                        "http://127.0.0.1:8000/stream_sentence",
                        json={"session_id": refusal_session_id, "text": response, "voice": voice_gender},
                        timeout=5,
                    )
                    res = requests.post(
                        "http://127.0.0.1:8000/stream_finish",
                        json={"session_id": refusal_session_id},
                        timeout=10,
                    )
                    if res.status_code == 200:
                        b64_audio = res.json().get("audio_base64")
                        if b64_audio:
                            audio_data = base64.b64decode(b64_audio)
                except Exception:
                    pass

                render_streaming_tts_player(refusal_session_id, direct_audio_b64=b64_audio)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "audio": audio_data,
                "debug": debug,
            })
            return

        system_prompt, user_prompt = build_prompt(
            outcome.original_query, outcome.contexts, chat_history, language=detected_lang
        )

        enable_tts = st.session_state.get("enable_tts", True)
        voice_gender = st.session_state.get("voice_gender", "Female")
        session_id = uuid.uuid4().hex if enable_tts else None

        # Render live streaming audio player widget
        audio_placeholder = st.empty()
        if enable_tts and session_id:
            with audio_placeholder:
                render_streaming_tts_player(session_id)

        # Background sentence sender queue to ensure 0ms blocking on LLM token stream
        tts_sentence_queue = queue.Queue()
        tts_sentinel = object()

        def background_tts_sender():
            while True:
                item = tts_sentence_queue.get()
                if item is tts_sentinel:
                    break
                try:
                    requests.post(
                        "http://127.0.0.1:8000/stream_sentence",
                        json={"session_id": session_id, "text": item, "voice": voice_gender},
                        timeout=5,
                    )
                except Exception:
                    pass
                tts_sentence_queue.task_done()

        if enable_tts and session_id:
            sender_thread = threading.Thread(target=background_tts_sender, daemon=True)
            sender_thread.start()

        message_placeholder = st.empty()
        full_response = ""
        sentence_buffer = ""
        # Split on sentence endings in Hindi / Chhattisgarhi (Danda ।, ?, !, \n) or English (. followed by space)
        sentence_delimiter = re.compile(r'([।?!\n]|\.\s+)')

        try:
            for chunk in call_llm_stream(user_prompt, system_instruction=system_prompt):
                full_response += chunk
                live_text = strip_inline_source_lines(full_response)
                message_placeholder.markdown(live_text + " ▌")

                # Concurrently push completed sentences to background queue
                if enable_tts and session_id:
                    sentence_buffer += chunk
                    parts = sentence_delimiter.split(sentence_buffer)
                    if len(parts) > 1:
                        completed_sentence = "".join(parts[:-1]).strip()
                        if completed_sentence and len(completed_sentence.split()) >= 3:
                            tts_sentence_queue.put(completed_sentence)
                            sentence_buffer = parts[-1]

            # Flush any remaining sentence buffer
            if enable_tts and session_id and sentence_buffer.strip():
                tts_sentence_queue.put(sentence_buffer.strip())

        except Exception as error:
            LOGGER.warning("Streaming error, fallback: %s", error)
            if not full_response:
                full_response = call_llm(user_prompt, system_instruction=system_prompt)

        if enable_tts and session_id:
            tts_sentence_queue.put(tts_sentinel)

        response = full_response

        # Finish streaming TTS session and retrieve full audio for history replay
        audio_data = None
        if enable_tts and session_id:
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/stream_finish",
                    json={"session_id": session_id},
                    timeout=10,
                )
                if res.status_code == 200:
                    b64_audio = res.json().get("audio_base64")
                    if b64_audio:
                        audio_data = base64.b64decode(b64_audio)
            except Exception:
                pass

        context_pages = {
            page for context in outcome.contexts for page in context.page_numbers
        }
        diagnostic = diagnose_answer(response, context_pages)
        if diagnostic is not None:
            diagnostic_code, diagnostic_detail = diagnostic
            LOGGER.warning(
                "answer_guard_rejected code=%s detail=%s",
                diagnostic_code,
                diagnostic_detail,
            )
            if operator_mode:
                st.error(
                    "Chat provider response failed validation (page citation missing "
                    "or outside retrieval context). Review provider output."
                )
                st.caption(f"Diagnosis ({diagnostic_code}): {diagnostic_detail}")
            else:
                st.warning(
                    "The AI service could not return a verified response. "
                    "Please try again later or contact the administrator.",
                    icon="⚠️",
                )
            response = (
                "I apologize, but the AI service could not produce a verified answer "
                "at this time. Please try again later."
            )

        cited_pages = set(extract_cited_pages(response))
        source_refs = build_source_references(
            outcome.contexts,
            cited_pages=cited_pages,
        )
        display_response = strip_inline_source_lines(response)
        message_placeholder.markdown(display_response)

        # Fallback if streaming audio was not captured
        if enable_tts and not audio_data and display_response:
            audio_data = fetch_tts_audio(display_response, voice=voice_gender)

        sources_text = format_sources(
            outcome.contexts,
            cited_pages=cited_pages,
            include_scores=operator_mode,
        )
        if source_refs:
            render_source_references(
                source_refs,
                key_prefix=f"current_{len(st.session_state.messages)}",
                public=not operator_mode,
            )

        if operator_mode:
            render_retrieval_debug(debug)

        st.session_state.messages.append({
            "role": "assistant",
            "content": display_response,
            "audio": audio_data,
            "sources": sources_text,
            "source_refs": source_refs,
            "message_id": len(st.session_state.messages),
            "debug": debug,
        })


# ──────────────────────────────────────────────
# View 2: Document Explorer
# ──────────────────────────────────────────────

def render_document_explorer(retriever: DocumentRetriever | None) -> None:
    st.header("Document & Index Explorer")
    render_lead(
        "Inspect text structure, tokenized chunks, and page distributions "
        "across indexed source documents."
    )

    if not retriever:
        st.warning("Index is not available yet.")
        return

    info = retriever.document_info
    df = load_metadata_df()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunks", f"{info.get('total_chunks', 0):,}")
    col2.metric("Total Pages", f"{info.get('total_pages', 0)}")
    col3.metric("Page Range", f"p. {info.get('page_range', '-')}")
    col4.metric(
        "Pages per Chunk",
        f"{info.get('avg_pages_per_chunk', 0)} avg",
        help="Values closer to 1 ensure higher citation precision. "
             f"Maximum page span: {info.get('max_pages_per_chunk', 0)} pages.",
    )

    documents = info.get("documents") or []
    if len(documents) > 1:
        st.subheader("Documents in Index")
        st.dataframe(
            pd.DataFrame([
                {
                    "Document": doc["name"],
                    "Chunks": doc["chunks"],
                    "Pages": doc["pages"],
                }
                for doc in documents
            ]),
            hide_index=True,
            width="stretch",
        )

    lexical = info.get("lexical") or {}
    if lexical:
        st.caption(
            f"**BM25 Lexical Index** — {lexical.get('documents', 0)} documents · "
            f"{lexical.get('vocabulary', 0):,} unique terms · "
            f"average {lexical.get('avg_tokens_per_doc', 0)} tokens per chunk · "
            f"k1={lexical.get('k1')}, b={lexical.get('b')}. "
            "Constructed in-memory at application startup."
        )

    st.subheader("Search Document Content")
    search_term = st.text_input(
        "Filter chunk text",
        placeholder="Example: thesis, leave, curriculum, honors...",
    )

    display_df = df.copy()
    if search_term and not display_df.empty:
        haystack = display_df["text"].fillna("")
        if "section_title" in display_df.columns:
            haystack = haystack + " " + display_df["section_title"].fillna("")
        display_df = display_df[haystack.str.contains(search_term, case=False, na=False)]

    if not display_df.empty:
        st.caption(f"Displaying {min(len(display_df), 50)} of {len(display_df)} matching chunks")
        table_data = []
        for _, row in display_df.head(50).iterrows():
            pages = row.get("page_numbers") or []
            page_label = "-"
            if len(pages):
                first, last = min(pages), max(pages)
                page_label = str(first) if first == last else f"{first}-{last}"

            heading_path = row.get("heading_path")
            if isinstance(heading_path, list) and heading_path:
                heading = " › ".join(heading_path)
            else:
                heading = row.get("section_title") or "-"

            entri = {
                "Index": row.get("chunk_index"),
                "Page": page_label,
                "Section / Title": heading,
                "Characters": row.get("char_count"),
                "Text Excerpt": (row.get("text") or "")[:140] + "...",
            }
            if len(documents) > 1:
                name = row.get("source_document") or "-"
                entri = {"Document": name.rsplit(".", 1)[0][:28], **entri}
            table_data.append(entri)
        st.dataframe(pd.DataFrame(table_data), hide_index=True, width="stretch")
    else:
        st.info("No chunks match the search filter.")


# ──────────────────────────────────────────────
# View 3: Evaluation
# ──────────────────────────────────────────────

def render_evaluation(retriever: DocumentRetriever | None) -> None:
    st.header("RAG Quality Evaluation")
    render_lead(
        "Evaluate retrieval precision and out-of-scope question refusal."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Target Recall@4", "≥ 80.0%")
    col2.metric("Refusal Accuracy", "100.0%")
    col3.metric("Target Latency", "< 5.0 s")

    questions = load_test_questions()
    if not questions:
        st.info("File `eval/test_questions.json` not found.")
        return

    in_scope = [q for q in questions if q.get("category") == "in_scope"]
    out_scope = [q for q in questions if q.get("category") == "out_of_scope"]

    tab1, tab2 = st.tabs([f"In-Scope ({len(in_scope)} Questions)", f"Out-of-Scope ({len(out_scope)} Questions)"])

    with tab1:
        st.caption("Questions whose answers exist within the academic guidelines:")
        in_data = [
            {
                "ID": q["id"],
                "Question": q["question"],
                "Expected Keywords": ", ".join(q.get("expected_answer_contains", [])),
                "Category": "In-Scope",
            }
            for q in in_scope
        ]
        st.dataframe(pd.DataFrame(in_data), hide_index=True)

    with tab2:
        st.caption("Out-of-scope questions that must be politely declined (anti-hallucination):")
        out_data = [
            {
                "ID": q["id"],
                "Question": q["question"],
                "Expected Outcome": "Refused / Information not found",
                "Category": "Out-of-Scope",
            }
            for q in out_scope
        ]
        st.dataframe(pd.DataFrame(out_data), hide_index=True)


# ──────────────────────────────────────────────
# View 4: About
# ──────────────────────────────────────────────

def render_about(retriever: DocumentRetriever | None) -> None:
    st.header("About the Project")
    st.markdown(
        """
        This project implements a **Retrieval-Augmented Generation (RAG)** architecture to answer questions regarding campus academic policies and guidelines accurately and without hallucination.

        **Terms of Use & Disclaimer**
        This application serves as an educational reference assistant. Information provided is for academic guidance. For official administrative decisions, students must consult official faculty announcements and academic administration.
        """
    )

    st.subheader("Retrieval Pipeline")
    st.markdown(
        """
        Every user question passes through the following pipeline:

        1. **Query rewriting** — follow-up questions like *"what are the requirements?"* are converted
           into standalone queries using conversation history, **before** search runs.
        2. **Dual-branch retrieval** — *dense search* (Gemini embeddings + FAISS cosine similarity) captures
           semantic meaning, while *BM25 Okapi* captures exact keyword matches such as "GPA 3.50", "144 credits", or "Article 12".
        3. **Reciprocal Rank Fusion (RRF)** — ranks are merged by rank position rather than raw scores,
           seamlessly joining bounded cosine (0–1) and unbounded BM25 scores.
        4. **Reranking** — the cross-encoder LLM evaluates query and document excerpts together,
           scoring *answerability* rather than mere lexical similarity.
        5. **Multi-layer relevance gating** — similarity threshold, lexical term coverage, reranker score,
           and strict system instructions ensure out-of-scope questions are politely refused.
        """
    )

    st.subheader("Configuration & Technical Stack")
    info = retriever.document_info if retriever else {}
    index_info = info.get("index_info") or {}
    embedding_info = index_info.get("embedding") or {}
    lexical = info.get("lexical") or {}

    from app.index_status import indexed_documents

    sources = indexed_documents(index_info)
    if len(sources) > 1:
        label_documents = f"{len(sources)} documents: " + ", ".join(
            d.get("name", "?").rsplit(".", 1)[0][:32] for d in sources
        )
    elif sources:
        label_documents = sources[0].get("name", "-")
    else:
        label_documents = "Academic Guidelines 2025-2026"

    dimension = embedding_info.get("dimension")
    embedding_label = f"Google {EMBEDDING_MODEL}"
    if dimension:
        embedding_label += f" (dimension {dimension})"

    rows = [
        ("Institution", "IIIT Naya Raipur (Dr. SPM IIIT-NR)"),
        ("Source Documents", label_documents),
        ("LLM Generation Model", f"Google Gemini · {effective_model(LLM_MODEL)} (via API)"),
        ("Rerank & Rewrite Model", f"Google Gemini · {effective_model(UTILITY_MODEL)} (via API)"),
        ("Embedding Model", embedding_label),
        ("Embedding Task Type", embedding_info.get("task_type") or "generic"),
        ("Vector Store", "FAISS IndexFlatIP (cosine similarity)"),
        ("Lexical Index", f"BM25 Okapi in-memory, {lexical.get('vocabulary', 0):,} unique terms"),
        ("Total Indexed Chunks", f"{info.get('total_chunks', 0)} chunks"),
        ("Total Source Pages", f"{info.get('total_pages', 0)} pages"),
        (
            "Page Citation Precision",
            f"average {info.get('avg_pages_per_chunk', 0)} pages per chunk "
            f"(maximum {info.get('max_pages_per_chunk', 0)})",
        ),
        ("UI Framework", "Streamlit"),
        (
            "Hallucination Guardrails",
            "Similarity threshold + lexical coverage + reranker gate + strict system prompt",
        ),
    ]

    if index_info.get("built_at"):
        rows.append(("Index Built At", index_info["built_at"]))

    st.dataframe(
        pd.DataFrame(rows, columns=["Component", "Specification / Value"]),
        hide_index=True,
        width="stretch",
    )


# ──────────────────────────────────────────────
# Main Entry Point & Sidebar
# ──────────────────────────────────────────────

def _operator_main_legacy() -> None:
    render_hero()

    retriever, error = load_retriever()
    render_index_freshness(retriever)

    with st.sidebar:
        render_brand("IIIT-NR Chatbot", "Academic & Campus Assistant")

        nav_label("Navigation")
        page = st.radio(
            "Navigation",
            ["Q&A Chat", "Document Explorer", "Evaluation", "About"],
            label_visibility="collapsed",
        )

        st.divider()

        nav_label("System Status")
        if retriever:
            render_status_badge(
                "Vector Store",
                f"● Active Index · {retriever.total_chunks} Chunks",
                online=True,
            )
        else:
            render_status_badge("Vector Store", "● Index Not Ready", online=False)

        # Settings
        selected_documents: tuple[str, ...] = ()
        if retriever:
            available = retriever.source_documents
            if len(available) > 1:
                nav_label("Document Scope")
                chosen = st.multiselect(
                    "Search only in selected documents",
                    options=available,
                    default=available,
                    format_func=lambda name: name.rsplit(".", 1)[0][:42],
                    help="Leave empty to search all documents.",
                )
                if chosen and len(chosen) < len(available):
                    selected_documents = tuple(chosen)

        nav_label("Retrieval Pipeline")
        use_hybrid = st.toggle(
            "Hybrid search (BM25 + dense)",
            value=DEFAULT_RETRIEVAL_CONFIG.use_hybrid,
            help="Combines semantic embedding search with exact keyword matching (BM25) using Reciprocal Rank Fusion.",
        )
        use_rerank = st.toggle(
            "LLM Reranker",
            value=DEFAULT_RETRIEVAL_CONFIG.use_rerank,
            help="Evaluates candidate excerpts against the question simultaneously using cross-encoder scoring.",
        )
        if not use_rerank:
            st.warning(
                "Without the reranker, out-of-scope question refusal drops at retrieval level. "
                "Answers are still protected by system prompt, but irrelevant context may be included.",
                icon="⚠️",
            )
        use_rewrite = st.toggle(
            "Query rewriting",
            value=DEFAULT_RETRIEVAL_CONFIG.use_query_rewrite,
            help="Rewrites follow-up questions into standalone queries before retrieval.",
        )

        nav_label("Retrieval Parameters")
        top_k = st.slider(
            "Top-K Context Chunks", min_value=1, max_value=8,
            value=DEFAULT_RETRIEVAL_CONFIG.top_k,
            help="Number of document chunks sent to LLM as context",
        )
        threshold = st.slider(
            "Relevance Threshold", min_value=0.1, max_value=0.8,
            value=DEFAULT_RETRIEVAL_CONFIG.dense_threshold, step=0.05,
            help="Minimum cosine similarity for dense retrieval branch",
        )

        config = DEFAULT_RETRIEVAL_CONFIG.with_overrides(
            top_k=top_k,
            dense_threshold=threshold,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            use_query_rewrite=use_rewrite,
            document_filter=selected_documents,
        )

        nav_label("Voice Engine (GPU)")
        enable_tts = st.toggle("🔊 Voice Output", value=True, help="Synthesizes Hindi speech via local GPU VITS TTS")
        voice_gender = st.selectbox("Voice", ["Female", "Male"], index=0, key="operator_voice")
        st.session_state["enable_tts"] = enable_tts
        st.session_state["voice_gender"] = voice_gender
        st.caption("🎙️ STT: Port 8001 (Whisper / MMS)")
        st.caption("🔊 TTS: Port 8000 (Chhattisgarhi VITS)")

        st.divider()
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.pop("active_conversation_language", None)
            st.rerun()

        st.caption(
            f"Generation: {effective_model(LLM_MODEL)} · "
            f"Rerank/rewrite: {effective_model(UTILITY_MODEL)}"
        )

    if page == "Q&A Chat":
        render_chat_view(retriever, config, operator_mode=True)
    elif page == "Document Explorer":
        render_document_explorer(retriever)
    elif page == "Evaluation":
        render_evaluation(retriever)
    else:
        render_about(retriever)

    st.divider()
    render_footer(
        "IIIT-NR Academic & Campus RAG Chatbot"
    )


def main() -> None:
    """Run application in public or operator mode."""
    operator_mode = is_operator_mode()

    if operator_mode:
        _operator_main_legacy()
        return

    render_hero()
    retriever, error = load_retriever()

    limiter = SessionRateLimiter(
        limit=public_rate_limit(),
        window_seconds=public_rate_window_seconds(),
    )
    config = DEFAULT_RETRIEVAL_CONFIG

    with st.sidebar:
        render_brand("IIIT-NR Chatbot", "Academic & Campus Services")
        nav_label("Student Services")
        st.caption(
            "Ask academic, faculty, curriculum, and campus questions based on official IIIT-NR documents."
        )
        nav_label("Voice Engine (GPU)")
        enable_tts = st.toggle("🔊 Voice Output", value=True, help="Synthesizes Hindi speech via local GPU VITS TTS")
        voice_gender = st.selectbox("Voice", ["Female", "Male"], index=0, key="public_voice")
        st.session_state["enable_tts"] = enable_tts
        st.session_state["voice_gender"] = voice_gender
        st.caption("🎙️ STT: Port 8001 (Whisper / MMS)")
        st.caption("🔊 TTS: Port 8000 (Chhattisgarhi VITS)")

        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("active_conversation_language", None)
            st.rerun()
        st.caption("Use this information as guidance. For official administrative decisions, please consult academic administration.")

    render_chat_view(
        retriever,
        config,
        operator_mode=False,
        rate_limiter=limiter,
    )

    st.divider()
    render_footer(
        "IIIT-NR Academic & Campus RAG Chatbot"
    )


if __name__ == "__main__":
    main()
