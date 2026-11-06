"""
WebSocket and REST server exposing the STT pipeline.

Client protocol:
  1. WebSocket: Connect to ws://<host>:<port>/ws/stt/{session_id}
     Stream binary WebSocket frames of raw 16-bit PCM, mono, 16kHz audio.
     Receive JSON text frames (STTEvent): session_started, speech_started, partial, final.
  2. REST: POST to http://<host>:<port>/transcribe?language=auto
     Send raw audio bytes (WAV, PCM) in request body.
     Receive JSON: {"text": "...", "detected_language": "...", "inference_ms": ..., "confidence": ...}
"""
import asyncio
import base64
from datetime import datetime
import io
import json
import logging
from pathlib import Path
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

from src.asr import get_engine, logprob_to_confidence
from src.config import settings
from src.schemas import ClientControlMessage, EventType, STTEvent
from src.session import SttSession

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("stt-server")

_capacity = asyncio.Semaphore(settings.MAX_CONCURRENT_SESSIONS)
_active_sessions = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading ASR model '{settings.ASR_MODEL_SIZE}' on {settings.ASR_DEVICE} ({settings.ASR_COMPUTE_TYPE})...")
    get_engine()  # warm the default singleton
    if not settings.API_KEY:
        logger.warning("STT_API_KEY is not set — server will accept unauthenticated connections.")
    logger.info(f"Model ready. Capacity: {settings.MAX_CONCURRENT_SESSIONS} concurrent calls. Server port: {settings.PORT}.")
    yield
    logger.info("Shutting down STT server.")


app = FastAPI(title="Local Multilingual STT Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "model": settings.ASR_MODEL_SIZE,
        "device": settings.ASR_DEVICE,
        "active_sessions": _active_sessions,
        "max_sessions": settings.MAX_CONCURRENT_SESSIONS,
        "at_capacity": _active_sessions >= settings.MAX_CONCURRENT_SESSIONS,
        "supported_languages": ["auto", "hne", "hi", "en"],
    })


def save_audio_log(
    raw_bytes: bytes,
    processed_data: np.ndarray,
    sr: int,
    meta: dict,
) -> tuple[str, str, str]:
    """
    Persist raw audio, processed audio, and diagnostic metadata to disk.
    Returns (log_id, raw_path, processed_path).
    """
    log_dir = Path(settings.AUDIO_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    log_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"

    raw_path = log_dir / f"{log_id}_raw.wav"
    proc_path = log_dir / f"{log_id}_processed.wav"
    meta_path = log_dir / f"{log_id}_meta.json"

    try:
        raw_path.write_bytes(raw_bytes)
    except Exception as e:
        logger.warning(f"Failed to write raw audio log: {e}")

    try:
        buf = io.BytesIO()
        sf.write(buf, processed_data, sr, format="WAV", subtype="PCM_16")
        proc_path.write_bytes(buf.getvalue())
    except Exception as e:
        logger.warning(f"Failed to write processed audio log: {e}")

    try:
        meta_dict = {
            "log_id": log_id,
            "timestamp": datetime.now().isoformat(),
            **meta,
        }
        meta_path.write_text(json.dumps(meta_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write meta log: {e}")

    return log_id, str(raw_path), str(proc_path)


@app.post("/transcribe")
async def transcribe_endpoint(
    request: Request,
    language: str = Query("auto", description="Target language: 'auto', 'hne', 'hi', or 'en'")
):
    """
    HTTP endpoint to transcribe recorded audio bytes directly.
    Accepts raw audio bytes (WAV, PCM, etc.) in the request body.
    Supports Chhattisgarhi ('hne'), Hindi ('hi'), English ('en'), or 'auto'.
    Returns transcribed text, diagnostics, processed audio base64, and log metadata.
    """
    audio_bytes = await request.body()
    if not audio_bytes:
        return JSONResponse({"error": "Empty audio payload"}, status_code=400)

    try:
        # Decode audio using soundfile
        with io.BytesIO(audio_bytes) as f:
            data, sr = sf.read(f)
        sr_orig = sr

        if data.ndim > 1:
            # Multi-channel: pick channel with highest RMS energy (avoids halving volume on asymmetric mics)
            rms_per_ch = [float(np.sqrt(np.mean(data[:, c] ** 2))) for c in range(data.shape[1])]
            best_ch = int(np.argmax(rms_per_ch))
            data = data[:, best_ch]

        data = data.astype(np.float32)

        # 1. Resample to 16,000 Hz FIRST so all downstream processing operates at uniform rate
        if sr != settings.SAMPLE_RATE:
            tensor = torch.from_numpy(data).float()
            resampler = T.Resample(sr, settings.SAMPLE_RATE)
            data = resampler(tensor).numpy()
            sr = settings.SAMPLE_RATE

        # 2. Subtract DC offset
        data = data - np.mean(data)

        # 3. Robust Voice Normalization:
        # Ignore mouse clicks / button-release impulsive spikes by checking the 99th percentile of active audio
        abs_data = np.abs(data)
        non_silent = abs_data[abs_data > 0.005]
        if len(non_silent) > 0:
            effective_peak = float(np.percentile(non_silent, 99))
        else:
            effective_peak = float(np.max(abs_data)) if len(abs_data) > 0 else 0.0

        duration_s = len(data) / float(sr) if sr > 0 else 0
        logger.info(f"Transcribe request: dur={duration_s:.2f}s, effective_peak={effective_peak:.4f}, lang={language}")

        # Check if browser/driver hardware noise gate clamped trailing audio to pure zero
        hardware_gate_warning = ""
        if duration_s > 2.5:
            tail_samples = abs_data[int(len(abs_data) * 0.5):]
            zero_ratio = float(np.mean(tail_samples < 0.00001))
            if zero_ratio > 0.80:
                hardware_gate_warning = "Microphone audio was cut off and clamped to 0.0 by your browser (Brave Shields) or Windows/ASUS AI Noise-Canceling Microphone."
                logger.warning("Hardware/Browser clamp detected: %.1f%% of trailing audio is pure zero", zero_ratio * 100)

        # Check for near-absolute silence / muted microphone
        if effective_peak < 0.001 or duration_s < 0.2:
            logger.info("Audio is silent or too short (peak=%.4f, dur=%.2fs)", effective_peak, duration_s)
            return JSONResponse({
                "text": "",
                "detected_language": "unknown",
                "inference_ms": 0.0,
                "confidence": 0.0,
                "is_silent": True,
                "processed_audio_b64": "",
                "log_id": "",
                "raw_audio_path": "",
                "processed_audio_path": "",
            })

        # Normalize quiet laptop mic audio up to 0.85 so Silero VAD reliably picks up speech
        if effective_peak < 0.80:
            gain = min(0.85 / effective_peak, 25.0)  # Up to 25x gain
            data = np.clip(data * gain, -0.99, 0.99)

        # Generate 16kHz PCM WAV bytes of normalized/processed audio for client playback
        buf_proc = io.BytesIO()
        sf.write(buf_proc, data, settings.SAMPLE_RATE, format="WAV", subtype="PCM_16")
        proc_wav_bytes = buf_proc.getvalue()
        proc_b64 = base64.b64encode(proc_wav_bytes).decode("ascii")

        engine = get_engine(language)
        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(
            None,
            lambda: engine.transcribe(data, fast=False, language=language)
        )

        logger.info(f"Transcribed output: text={transcript.text!r}, lang={transcript.detected_language}, ms={transcript.inference_ms:.1f}")

        # Save audio logs to disk
        log_id, raw_path, proc_path = "", "", ""
        if settings.AUDIO_LOG_ENABLED:
            meta = {
                "language_requested": language,
                "detected_language": transcript.detected_language,
                "duration_seconds": round(duration_s, 2),
                "original_sample_rate": sr_orig,
                "effective_peak": round(effective_peak, 4),
                "inference_ms": round(transcript.inference_ms, 2),
                "confidence": round(float(logprob_to_confidence(transcript.avg_logprob)), 3),
                "transcribed_text": transcript.text,
            }
            log_id, raw_path, proc_path = save_audio_log(audio_bytes, data, settings.SAMPLE_RATE, meta)

        return JSONResponse({
            "text": transcript.text,
            "detected_language": transcript.detected_language,
            "inference_ms": round(transcript.inference_ms, 2),
            "confidence": round(float(logprob_to_confidence(transcript.avg_logprob)), 3),
            "is_silent": False,
            "processed_audio_b64": proc_b64,
            "log_id": log_id,
            "raw_audio_path": raw_path,
            "processed_audio_path": proc_path,
            "hardware_gate_warning": hardware_gate_warning,
        })
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws/stt/{session_id}")
async def stt_websocket(websocket: WebSocket, session_id: str = None):
    global _active_sessions

    if settings.API_KEY and websocket.query_params.get("api_key") != settings.API_KEY:
        await websocket.close(code=4401)
        logger.warning(f"[{session_id}] rejected: bad or missing api_key")
        return

    if _capacity.locked():
        await websocket.accept()
        await websocket.send_text(STTEvent(
            type=EventType.SESSION_REJECTED,
            session_id=session_id or "unknown",
            error_message="Server at capacity. Please try again shortly.",
        ).model_dump_json())
        await websocket.close(code=4503)
        logger.warning(f"[{session_id}] rejected: at capacity ({_active_sessions}/{settings.MAX_CONCURRENT_SESSIONS})")
        return

    await _capacity.acquire()
    _active_sessions += 1
    await websocket.accept()

    session_id = session_id or str(uuid.uuid4())
    session = SttSession(session_id)
    call_start = time.monotonic()
    logger.info(f"[{session_id}] connected ({_active_sessions}/{settings.MAX_CONCURRENT_SESSIONS} active)")

    try:
        await websocket.send_text((await session.start()).model_dump_json())

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=settings.IDLE_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning(f"[{session_id}] idle timeout after {settings.IDLE_TIMEOUT_S}s, closing")
                break

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                events = await session.process_chunk(message["bytes"])
                for event in events:
                    await websocket.send_text(event.model_dump_json())

            elif "text" in message and message["text"] is not None:
                try:
                    ctrl = ClientControlMessage.model_validate(json.loads(message["text"]))
                except Exception:
                    continue
                if ctrl.action == "stop":
                    final_event = await session.flush()
                    if final_event:
                        await websocket.send_text(final_event.model_dump_json())
                    break

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] disconnected")
    finally:
        try:
            final_event = await session.flush()
            if final_event:
                await websocket.send_text(final_event.model_dump_json())
        except Exception:
            pass
        _active_sessions -= 1
        _capacity.release()
        call_duration_s = time.monotonic() - call_start
        logger.info(f"[{session_id}] session closed — duration={call_duration_s:.1f}s, "
                     f"utterances={session._utterance_counter}, ({_active_sessions}/{settings.MAX_CONCURRENT_SESSIONS} active)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        ws_ping_interval=settings.WS_PING_INTERVAL_S,
        ws_ping_timeout=settings.WS_PING_TIMEOUT_S,
    )
