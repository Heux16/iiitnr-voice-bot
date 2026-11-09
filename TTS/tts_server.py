"""
FastAPI Microservice for Ultra Low-Latency Streaming Chhattisgarhi / Hindi TTS.
Uses decoupled async queuing: HTTP requests acknowledge in <1ms while the RTX 4050 GPU
synthesizes and pushes audio chunks concurrently over WebSockets.
"""

import os
import sys
import io
import re
import time
import json
import base64
import asyncio
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import scipy.io.wavfile as wavfile

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure Windows stdout encoding supports UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from streaming_tts import StreamingChhattisgarhiTTS

app = FastAPI(title="Ultra Low-Latency Streaming TTS Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_engines: Dict[str, StreamingChhattisgarhiTTS] = {}
audio_cache: Dict[str, bytes] = {}


# ──────────────────────────────────────────────
# Hindi Number to Words Normalizer
# ──────────────────────────────────────────────

_HINDI_NUMS = {
    0: 'शून्य', 1: 'एक', 2: 'दो', 3: 'तीन', 4: 'चार', 5: 'पाँच', 6: 'छह', 7: 'सात', 8: 'आठ', 9: 'नौ',
    10: 'दस', 11: 'ग्यारह', 12: 'बारह', 13: 'तेरह', 14: 'चौदह', 15: 'पंद्रह', 16: 'सोलह', 17: 'सत्रह', 18: 'अठारह', 19: 'उन्नीस',
    20: 'बीस', 21: 'इक्कीस', 22: 'बाईस', 23: 'तेईस', 24: 'चौबीस', 25: 'पच्चीस', 26: 'छब्बीस', 27: 'सत्ताईस', 28: 'अट्ठाईस', 29: 'उनतीस',
    30: 'तीस', 31: 'इकतीस', 32: 'बत्तीस', 33: 'तैंतीस', 34: 'चौंतीस', 35: 'पैंतीस', 36: 'छत्तीस', 37: 'सैंतीस', 38: 'अड़तीस', 39: 'उनतालीस',
    40: 'चालीस', 41: 'इकतालीस', 42: 'बयालीस', 43: 'तैंतालीस', 44: 'चौवालीस', 45: 'पैंतालीस', 46: 'छियालीस', 47: 'सैंतालीस', 48: 'अड़तालीस', 49: 'उनचास',
    50: 'पचास', 51: 'इक्यावन', 52: 'बावन', 53: 'तिरपन', 54: 'चौवन', 55: 'पचपन', 56: 'छप्पन', 57: 'सत्तावन', 58: 'अट्ठावन', 59: 'उनसठ',
    60: 'साठ', 61: 'इकसठ', 62: 'बासठ', 63: 'तिरसठ', 64: 'चौंसठ', 65: 'पैंसठ', 66: 'छियासठ', 67: 'सरसठ', 68: 'अड़सठ', 69: 'उनहत्तर',
    70: 'सत्तर', 71: 'इकहत्तर', 72: 'बहत्तर', 73: 'तिहत्तर', 74: 'चौहत्तर', 75: 'पचहत्तर', 76: 'छियाहत्तर', 77: 'सतहत्तर', 78: 'अठहत्तर', 79: 'उन्नासी',
    80: 'अस्सी', 81: 'इक्यासी', 82: 'बयासी', 83: 'तिरासी', 84: 'चौरासी', 85: 'पचासी', 86: 'छियासी', 87: 'सत्तासी', 88: 'अट्ठासी', 89: 'नवासी',
    90: 'नब्बे', 91: 'इक्यानवे', 92: 'बानवे', 93: 'तिरानवे', 94: 'चौरानवे', 95: 'पचानवे', 96: 'छियानवे', 97: 'सत्तानवे', 98: 'अट्ठानवे', 99: 'निन्यानवे'
}

def num_to_hindi(n: int) -> str:
    if n < 0:
        return 'माइनस ' + num_to_hindi(-n)
    if n in _HINDI_NUMS:
        return _HINDI_NUMS[n]
    if n < 1000:
        hundreds = n // 100
        rem = n % 100
        res = _HINDI_NUMS.get(hundreds, '') + ' सौ'
        if rem > 0:
            res += ' ' + num_to_hindi(rem)
        return res
    if n < 100000:
        thousands = n // 1000
        rem = n % 1000
        res = num_to_hindi(thousands) + ' हज़ार'
        if rem > 0:
            res += ' ' + num_to_hindi(rem)
        return res
    if n < 10000000:
        lakhs = n // 100000
        rem = n % 100000
        res = num_to_hindi(lakhs) + ' लाख'
        if rem > 0:
            res += ' ' + num_to_hindi(rem)
        return res
    crores = n // 10000000
    rem = n % 10000000
    res = num_to_hindi(crores) + ' करोड़'
    if rem > 0:
        res += ' ' + num_to_hindi(rem)
    return res

def convert_numbers_in_text(text: str) -> str:
    """Converts digits, currency symbols, and percentages to Hindi words."""
    # Currency: ₹90,000 or Rs. 90000 -> नब्बे हज़ार रुपये
    def curr_repl(m):
        raw = m.group(1).replace(',', '')
        try:
            return num_to_hindi(int(raw)) + ' रुपये'
        except ValueError:
            return m.group(0)
    text = re.sub(r'(?:₹|Rs\.?|INR)\s*([0-9,]+)', curr_repl, text, flags=re.IGNORECASE)

    # Percentage: 75% -> पचहत्तर प्रतिशत
    def pct_repl(m):
        raw = m.group(1).replace(',', '')
        try:
            return num_to_hindi(int(raw)) + ' प्रतिशत'
        except ValueError:
            return m.group(0)
    text = re.sub(r'([0-9,]+)\s*%', pct_repl, text)

    # Standalone numbers: 2024 -> दो हज़ार चौबीस, 4 -> चार
    def num_repl(m):
        raw = m.group(0).replace(',', '')
        try:
            return num_to_hindi(int(raw))
        except ValueError:
            return m.group(0)
    text = re.sub(r'\b[0-9,]+\b', num_repl, text)
    return text


def clean_text_for_tts(text: str) -> str:
    """Strip markdown, citations, symbols, and normalize numbers to words."""
    # Remove source lines (📄 Source: Page N ...)
    text = re.sub(r'(?:📄|\ud83d\udcc4)?\s*(?:Source|Page|स्रोत).*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Check if text contains Devanagari characters
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    if has_devanagari:
        # Convert digits/amounts/percentages to Hindi words only for Devanagari text
        text = convert_numbers_in_text(text)
    # Remove markdown headers and formatting
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove bullets and list markers
    text = re.sub(r'^\s*[-*•\d\.]+\s*', '', text, flags=re.MULTILINE)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def synthesize_english_sapi(text: str, voice: str = "Female") -> Tuple[Optional[np.ndarray], Optional[bytes], float]:
    """Synthesize English text using Windows SAPI (Zira for Female, David for Male) at 22050Hz."""
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        speaker = None
        stream = None
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            target_name = "Zira" if voice.lower() == "female" else "David"
            for v in speaker.GetVoices():
                if target_name.lower() in v.GetDescription().lower():
                    speaker.Voice = v
                    break
            stream = win32com.client.Dispatch("SAPI.SpMemoryStream")
            stream.Format.Type = 22  # SAFT22kHz16BitMono
            speaker.AudioOutputStream = stream
            speaker.Rate = 0  # 1.0X normal speed
            speaker.Speak(text)
            raw_pcm = bytes(stream.GetData())
            pcm_arr = np.frombuffer(raw_pcm, dtype=np.int16)
            if len(pcm_arr) == 0:
                return None, None, 0.0

            buf = io.BytesIO()
            wavfile.write(buf, 22050, pcm_arr)
            wav_bytes = buf.getvalue()

            dur = len(pcm_arr) / 22050.0
            wav_arr = pcm_arr.astype(np.float32) / 32767.0
            return wav_arr, wav_bytes, dur
        finally:
            del speaker
            del stream
            pythoncom.CoUninitialize()
    except Exception as e:
        print(f"[SAPI Warning] English synthesis error: {e}")
        return None, None, 0.0


def get_engine(voice: str = "Female") -> StreamingChhattisgarhiTTS:
    voice_key = "Male" if voice.lower() == "male" else "Female"
    if voice_key not in tts_engines:
        print(f"[TTS Server] Initializing {voice_key} voice on GPU (1.25x speed)...")
        tts_engines[voice_key] = StreamingChhattisgarhiTTS(
            voice=voice_key,
            use_cuda=True,
            length_scale=0.80
        )
    return tts_engines[voice_key]


# ──────────────────────────────────────────────
# Non-Blocking Streaming Session Management
# ──────────────────────────────────────────────

class AudioSession:
    def __init__(self, session_id: str, voice: str = "Female"):
        self.session_id = session_id
        self.voice = voice
        self.websocket: Optional[WebSocket] = None
        self.text_queue: asyncio.Queue = asyncio.Queue()
        self.chunks: List[np.ndarray] = []
        self.worker_task: Optional[asyncio.Task] = None
        self.created_at = time.time()
        self.is_done = False
        self.packet_history: List[Tuple[str, Any]] = []
        self.packet_event: asyncio.Event = asyncio.Event()


active_sessions: Dict[str, AudioSession] = {}


async def session_gpu_worker(session: AudioSession):
    """
    Dedicated GPU worker task running concurrently in the background.
    Synthesizes sentences and buffers packets so any WebSocket client (connected
    before, during, or after synthesis) receives the full, ordered audio stream.
    """
    engine = get_engine(session.voice)
    while True:
        text_item = await session.text_queue.get()
        if text_item is None:
            # End of session sentinel
            session.is_done = True
            done_meta = json.dumps({"type": "done"})
            session.packet_history.append(("text", done_meta))
            session.packet_event.set()
            session.text_queue.task_done()
            break

        clean_text = clean_text_for_tts(text_item)
        if clean_text:
            has_devanagari = bool(re.search(r"[\u0900-\u097F]", clean_text))
            if not has_devanagari:
                # English sentence: synthesize via native Windows SAPI at 22050Hz
                wav_arr, wav_bytes, dur = await asyncio.to_thread(synthesize_english_sapi, clean_text, session.voice)
                if wav_bytes and wav_arr is not None:
                    session.chunks.append(wav_arr)
                    meta_json = json.dumps({
                        "type": "chunk_meta",
                        "text": clean_text,
                        "duration": round(dur, 2)
                    })
                    session.packet_history.append(("text", meta_json))
                    session.packet_history.append(("bytes", wav_bytes))
                    session.packet_event.set()
                    print(f"[Worker] Synthesized English {dur:.2f}s via SAPI. Packets: {len(session.packet_history)}")
                else:
                    # Fallback to browser Web Speech API if SAPI is unavailable
                    dur = round(max(0.8, len(clean_text.split()) * 0.35), 2)
                    meta_json = json.dumps({
                        "type": "chunk_meta",
                        "text": clean_text,
                        "duration": dur
                    })
                    session.packet_history.append(("text", meta_json))
                    session.packet_event.set()
                    print(f"[Worker] Delegated English sentence to client Web Speech: '{clean_text}'")
            else:
                def run_synthesis():
                    t0 = time.time()
                    wav = engine.synthesizer.tts(text=clean_text)
                    wav_arr = np.array(wav, dtype=np.float32)
                    dur = len(wav_arr) / engine.sample_rate

                    max_val = np.max(np.abs(wav_arr))
                    norm_audio = (wav_arr / max_val * 32767).astype(np.int16) if max_val > 0 else (wav_arr * 32767).astype(np.int16)

                    buf = io.BytesIO()
                    wavfile.write(buf, engine.sample_rate, norm_audio)
                    return wav_arr, buf.getvalue(), dur, time.time() - t0

                try:
                    wav_arr, wav_bytes, dur, infer_time = await asyncio.to_thread(run_synthesis)
                    session.chunks.append(wav_arr)

                    meta_json = json.dumps({
                        "type": "chunk_meta",
                        "text": clean_text,
                        "duration": round(dur, 2)
                    })
                    session.packet_history.append(("text", meta_json))
                    session.packet_history.append(("bytes", wav_bytes))
                    session.packet_event.set()
                    print(f"[Worker] Synthesized {dur:.2f}s in {infer_time:.3f}s on GPU. Packets: {len(session.packet_history)}")
                except Exception as e:
                    print(f"[Worker Error] Synthesis failed on '{clean_text}': {e}")

        session.text_queue.task_done()


def get_or_create_session(session_id: str, voice: str = "Female") -> AudioSession:
    if session_id not in active_sessions:
        session = AudioSession(session_id, voice)
        session.worker_task = asyncio.create_task(session_gpu_worker(session))
        active_sessions[session_id] = session
    return active_sessions[session_id]


@app.on_event("startup")
def startup_event():
    try:
        get_engine("Female")
        print("[TTS Server] Startup complete. NVIDIA RTX 4050 GPU ready!")
    except Exception as e:
        print(f"[TTS Server Startup Error] {e}")


@app.get("/health")
def health():
    return {
        "status": "ready",
        "loaded_voices": list(tts_engines.keys()),
        "cuda": any(e.use_cuda for e in tts_engines.values()),
        "active_streaming_sessions": len(active_sessions)
    }


# ──────────────────────────────────────────────
# 1. WebSocket Endpoint for Browser Real-Time Audio
# ──────────────────────────────────────────────

@app.websocket("/ws_audio/{session_id}")
async def ws_audio(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = get_or_create_session(session_id)
    session.websocket = websocket
    print(f"[WebSocket] Client connected: {session_id}")

    try:
        sent_index = 0
        while True:
            # Deliver any pending packets from history in sequence
            while sent_index < len(session.packet_history):
                pkt_type, pkt_data = session.packet_history[sent_index]
                if pkt_type == "text":
                    await websocket.send_text(pkt_data)
                elif pkt_type == "bytes":
                    await websocket.send_bytes(pkt_data)
                sent_index += 1

            if session.is_done and sent_index >= len(session.packet_history):
                print(f"[WebSocket] All packets delivered for session: {session_id}")
                break

            session.packet_event.clear()
            try:
                await asyncio.wait_for(session.packet_event.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                pass

        # Give client 1.5 seconds grace to consume all TCP frames before closing
        await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected: {session_id}")
    except Exception as e:
        print(f"[WebSocket Error] {session_id}: {e}")
    finally:
        session.websocket = None


# ──────────────────────────────────────────────
# 2. Instant Non-Blocking Push Endpoint (<1ms latency)
# ──────────────────────────────────────────────

class StreamSentenceRequest(BaseModel):
    session_id: str
    text: str
    voice: Optional[str] = "Female"


@app.post("/stream_sentence")
async def stream_sentence(req: StreamSentenceRequest):
    """
    Sub-millisecond instant return: puts text into background GPU worker queue.
    Never blocks the LLM streaming thread!
    """
    if not req.text.strip():
        return {"status": "skipped"}

    session = get_or_create_session(req.session_id, req.voice or "Female")
    await session.text_queue.put(req.text)
    return {"status": "queued"}


class StreamFinishRequest(BaseModel):
    session_id: str


@app.post("/stream_finish")
async def stream_finish(req: StreamFinishRequest):
    session = active_sessions.get(req.session_id)
    if not session:
        return {"status": "done", "audio_base64": None}

    # Signal worker to finish and wait for remaining sentences
    await session.text_queue.put(None)
    if session.worker_task:
        await session.worker_task

    full_b64 = None
    total_dur = 0.0
    if session.chunks:
        engine = get_engine(session.voice)
        full_wav = np.concatenate(session.chunks)
        total_dur = round(len(full_wav) / engine.sample_rate, 2)

        max_val = np.max(np.abs(full_wav))
        norm_audio = (full_wav / max_val * 32767).astype(np.int16) if max_val > 0 else (full_wav * 32767).astype(np.int16)

        buf = io.BytesIO()
        wavfile.write(buf, engine.sample_rate, norm_audio)
        full_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Cleanup session after delay
    async def cleanup():
        await asyncio.sleep(60)
        active_sessions.pop(req.session_id, None)

    asyncio.create_task(cleanup())

    return {
        "status": "done",
        "audio_base64": full_b64,
        "total_duration": total_dur
    }


# ──────────────────────────────────────────────
# 3. Synchronous Synthesize Endpoint (Fallback)
# ──────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "Female"


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    clean_text = clean_text_for_tts(req.text)
    if not clean_text:
        raise HTTPException(status_code=400, detail="No readable text after cleaning")

    cache_key = f"{req.voice}_{clean_text[:120]}"
    if cache_key in audio_cache:
        return Response(content=audio_cache[cache_key], media_type="audio/wav")

    engine = get_engine(req.voice or "Female")
    try:
        wav = engine.synthesizer.tts(text=clean_text)
        wav_arr = np.array(wav, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}")

    max_val = np.max(np.abs(wav_arr))
    norm_audio = (wav_arr / max_val * 32767).astype(np.int16) if max_val > 0 else (wav_arr * 32767).astype(np.int16)

    buffer = io.BytesIO()
    wavfile.write(buffer, engine.sample_rate, norm_audio)
    audio_bytes = buffer.getvalue()

    if len(audio_cache) > 50:
        audio_cache.pop(next(iter(audio_cache)))
    audio_cache[cache_key] = audio_bytes

    return Response(content=audio_bytes, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
