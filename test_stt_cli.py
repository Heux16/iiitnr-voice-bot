"""
test_stt_cli.py — Terminal Debugger & Tester for Multilingual STT

Allows developers and AI agents to test and debug the STT engine directly from the
terminal without needing a browser or Streamlit UI.

Usage:
  1. Interactive Menu:
     python test_stt_cli.py

  2. Record from mic for N seconds:
     python test_stt_cli.py --record 4 --lang auto
     python test_stt_cli.py --record 5 --lang hne   # Chhattisgarhi
     python test_stt_cli.py --record 4 --lang hi    # Hindi
     python test_stt_cli.py --record 3 --lang en    # English

  3. Test with an audio file:
     python test_stt_cli.py --file path/to/audio.wav --lang hne

  4. Test TTS -> STT roundtrip (synthesizes speech on port 8000 and transcribes on 8001):
     python test_stt_cli.py --synth "नवा रायपुर मा फीस कतेक हे" --lang hne
     python test_stt_cli.py --synth "कंप्यूटर साइंस में कितनी सीटें हैं?" --lang auto

  5. Check server health & GPU status:
     python test_stt_cli.py --status
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

# Force UTF-8 stdout for proper Devanagari rendering on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

STT_URL = os.getenv("STT_URL", "http://127.0.0.1:8001")
TTS_URL = os.getenv("TTS_URL", "http://127.0.0.1:8000")
SAMPLE_RATE = 16000


def check_server_health(verbose: bool = True) -> dict | None:
    """Check if the STT microservice on port 8001 is active."""
    try:
        resp = requests.get(f"{STT_URL}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if verbose:
                print(f"✅ STT Server is ONLINE on {STT_URL}")
                print(f"   Model:  {data.get('model', 'unknown')}")
                print(f"   Device: {data.get('device', 'unknown')}")
                print(f"   Langs:  {', '.join(data.get('supported_languages', []))}")
            return data
    except Exception as e:
        if verbose:
            print(f"❌ STT Server is OFFLINE on {STT_URL} ({e})")
            print(f"   Start it with: python STT/stt-service/run.py")
    return None


def draw_volume_meter(peak: float, length: int = 20) -> str:
    """Generate a visual ASCII volume level meter."""
    filled = min(length, int(peak * length * 1.5))
    bar = "█" * filled + "░" * (length - filled)
    return f"[{bar}] {peak:.3f}"


def record_mic_fixed(duration_s: float = 4.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio for a fixed number of seconds with a live VU meter."""
    print(f"\n🎙️  Recording for {duration_s:.1f} seconds... Speak into your microphone now!")
    chunk_samples = int(sample_rate * 0.1)  # 100ms chunks for live meter
    total_chunks = int(duration_s / 0.1)
    frames = []

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        for i in range(total_chunks):
            chunk, _ = stream.read(chunk_samples)
            frames.append(chunk)
            peak = float(np.max(np.abs(chunk)))
            elapsed = (i + 1) * 0.1
            meter = draw_volume_meter(peak)
            sys.stdout.write(f"\r  🎙️  Level: {meter} | Time: {elapsed:.1f}s / {duration_s:.1f}s")
            sys.stdout.flush()

    sys.stdout.write("\n\n")
    audio = np.concatenate(frames, axis=0).flatten()
    return audio


def record_mic_interactive(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Start recording on Enter, stop on Enter."""
    input("\n👉 Press [ENTER] to START recording...")
    print("🎙️  Recording started! Speak into your microphone. Press [ENTER] to STOP.")

    frames = []
    stop_event = threading.Event()

    def _rec_worker():
        chunk_samples = int(sample_rate * 0.1)
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            while not stop_event.is_set():
                chunk, _ = stream.read(chunk_samples)
                frames.append(chunk)
                peak = float(np.max(np.abs(chunk)))
                meter = draw_volume_meter(peak)
                sys.stdout.write(f"\r  🎙️  Level: {meter} | Press [ENTER] to finish...")
                sys.stdout.flush()

    t = threading.Thread(target=_rec_worker, daemon=True)
    t.start()

    input()
    stop_event.set()
    t.join(timeout=1.0)
    sys.stdout.write("\n\n")

    if not frames:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(frames, axis=0).flatten()


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert float32 numpy array to 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def send_transcribe_request(wav_bytes: bytes, language: str = "auto") -> dict:
    """Send audio bytes to STT server /transcribe endpoint."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{STT_URL}/transcribe?language={language}",
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"},
            timeout=30,
        )
        total_time_ms = (time.monotonic() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            data["client_roundtrip_ms"] = round(total_time_ms, 2)
            return data
        return {"error": f"Server HTTP {resp.status_code}: {resp.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to STT server at {STT_URL}. Is it running?"}
    except Exception as e:
        return {"error": str(e)}


def transcribe_direct(audio: np.ndarray, language: str = "auto") -> dict:
    """In-process transcription fallback when server is not running."""
    project_root = Path(__file__).resolve().parent
    stt_src = project_root / "STT" / "stt-service"
    if str(stt_src) not in sys.path:
        sys.path.insert(0, str(stt_src))

    from src.asr import get_engine, logprob_to_confidence

    engine = get_engine(language)
    transcript = engine.transcribe(audio, fast=False, language=language)
    return {
        "text": transcript.text,
        "detected_language": transcript.detected_language,
        "inference_ms": round(transcript.inference_ms, 2),
        "confidence": round(float(logprob_to_confidence(transcript.avg_logprob)), 3),
        "mode": "in-process direct",
    }


def print_stt_result(result: dict, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Pretty print transcription results and acoustic diagnostics."""
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    duration_s = len(audio) / float(sr) if sr > 0 else 0.0

    print("=" * 64)
    print("📊 ACOUSTIC DIAGNOSTICS:")
    print(f"   ⏱️  Audio Duration:  {duration_s:.2f} seconds ({len(audio)} samples @ {sr} Hz)")
    print(f"   🔊 Peak Amplitude:  {peak:.4f} " + ("⚠️ (Low volume, speak closer)" if peak < 0.03 else "✅ (Good level)"))
    print("-" * 64)

    if result.get("error"):
        print(f"❌ TRANSCRIPTION FAILED: {result['error']}")
        print("=" * 64)
        return

    text = result.get("text", "")
    det_lang = result.get("detected_language", "unknown")
    inf_ms = result.get("inference_ms", 0.0)
    confidence = result.get("confidence", 0.0)
    roundtrip = result.get("client_roundtrip_ms")

    if not text:
        print("⚠️  NO SPEECH RECOGNIZED:")
        print("   Whisper/MMS detected silence or background noise only.")
        print("   Tips: Ensure your mic is selected as default and speak clearly.")
    else:
        print("🎯 RECOGNITION RESULT:")
        print(f"   📝 Text:             {text}")
        print(f"   🌐 Language:         {det_lang.upper()}")
        print(f"   ⚡ GPU Inference:    {inf_ms:.1f} ms" + (f" (HTTP roundtrip: {roundtrip:.1f} ms)" if roundtrip else ""))
        print(f"   📊 Confidence:       {confidence * 100:.1f}%")
        if result.get("log_id"):
            print(f"   💾 Audio Log ID:     {result.get('log_id')}")
            print(f"      - Raw:           {result.get('raw_audio_path')}")
            print(f"      - Processed:     {result.get('processed_audio_path')}")

        # Try query rewriting if available to demonstrate academic retrieval query
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from app.query_rewriter import rewrite_query
            en_query = rewrite_query(text, [])
            if en_query != text:
                print(f"   🔄 Rewritten for RAG: {en_query}")
        except Exception:
            pass

    print("=" * 64)


def test_mic(duration: float = 4.0, language: str = "auto", interactive: bool = False) -> None:
    """Test mic recording and transcribe."""
    server_info = check_server_health(verbose=False)

    if interactive:
        audio = record_mic_interactive()
    else:
        audio = record_mic_fixed(duration_s=duration)

    if len(audio) == 0:
        print("⚠️ No audio recorded.")
        return

    wav_bytes = audio_to_wav_bytes(audio)

    print("⏳ Transcribing on GPU...")
    if server_info:
        result = send_transcribe_request(wav_bytes, language=language)
    else:
        print("ℹ️ Server offline, falling back to direct in-process engine...")
        result = transcribe_direct(audio, language=language)

    print_stt_result(result, audio)


def test_file(file_path: str, language: str = "auto") -> None:
    """Test an existing audio file."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"\n📁 Loading file: {path.name}")
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        import torch
        import torchaudio.transforms as T
        data = T.Resample(sr, SAMPLE_RATE)(torch.from_numpy(data)).numpy()
        sr = SAMPLE_RATE

    wav_bytes = audio_to_wav_bytes(data, sample_rate=sr)
    server_info = check_server_health(verbose=False)

    print("⏳ Transcribing on GPU...")
    if server_info:
        result = send_transcribe_request(wav_bytes, language=language)
    else:
        result = transcribe_direct(data, language=language)

    print_stt_result(result, data, sr=sr)


def test_synth_roundtrip(sentence: str, language: str = "auto") -> None:
    """Synthesize speech on TTS server (port 8000) and transcribe on STT server (port 8001)."""
    print(f"\n🔄 Running TTS -> STT Roundtrip Test")
    print(f"   Input Sentence: '{sentence}'")
    print(f"   Target Language: {language}")

    # 1. Synthesize via TTS
    try:
        tts_resp = requests.post(
            f"{TTS_URL}/synthesize",
            json={"text": sentence, "voice": "Female"},
            timeout=15,
        )
        if tts_resp.status_code != 200:
            print(f"❌ TTS synthesis failed on {TTS_URL}: HTTP {tts_resp.status_code}")
            return
        wav_bytes = tts_resp.content
        print(f"   ✅ TTS Synthesized: {len(wav_bytes)} bytes WAV")
    except Exception as e:
        print(f"❌ Cannot connect to TTS server on {TTS_URL}: {e}")
        return

    # 2. Transcribe via STT
    data, sr = sf.read(io.BytesIO(wav_bytes))
    result = send_transcribe_request(wav_bytes, language=language)
    print_stt_result(result, data, sr=sr)


def interactive_menu() -> None:
    """Interactive terminal menu for debugging STT."""
    while True:
        print("\n" + "=" * 60)
        print("🎙️  IIIT-NR MULTILINGUAL STT TERMINAL DEBUGGER")
        print("=" * 60)
        check_server_health(verbose=True)
        print("\nSelect an action:")
        print("  [1] ⏱️  Quick Record from Mic (Fixed Duration, e.g. 4s)")
        print("  [2] 🎙️  Interactive Record (Press Enter to start & stop)")
        print("  [3] 📁 Transcribe an Audio File (WAV / MP3)")
        print("  [4] 🔄 Test TTS -> STT Roundtrip (Synthesize & Transcribe)")
        print("  [5] 📊 Detailed Server & GPU Diagnostics")
        print("  [0] ❌ Exit")

        choice = input("\nEnter choice [0-5]: ").strip()

        if choice == "0":
            print("Exiting. Bye!")
            break

        if choice in ("1", "2", "3", "4"):
            print("\nSelect Language Mode:")
            print("  [1] 🌐 auto  (Multilingual Whisper: English / Hindi / Hinglish)")
            print("  [2] 🌾 hne   (Meta MMS 1B: Chhattisgarhi)")
            print("  [3] 🇮🇳 hi    (Whisper: Hindi)")
            print("  [4] 🇬🇧 en    (Whisper: English)")
            lang_choice = input("Enter language [1-4, default=1]: ").strip()
            lang_map = {"1": "auto", "2": "hne", "3": "hi", "4": "en", "": "auto"}
            selected_lang = lang_map.get(lang_choice, "auto")

            if choice == "1":
                dur_input = input("Recording duration in seconds [default=4]: ").strip()
                dur = float(dur_input) if dur_input else 4.0
                test_mic(duration=dur, language=selected_lang, interactive=False)
            elif choice == "2":
                test_mic(language=selected_lang, interactive=True)
            elif choice == "3":
                fpath = input("Enter audio file path: ").strip().strip('"').strip("'")
                test_file(fpath, language=selected_lang)
            elif choice == "4":
                sample = input("Enter sentence to synthesize [default='नवा रायपुर मा फीस कतेक हे']: ").strip()
                if not sample:
                    sample = "नवा रायपुर मा फीस कतेक हे"
                test_synth_roundtrip(sample, language=selected_lang)

        elif choice == "5":
            check_server_health(verbose=True)
            try:
                import torch
                print(f"\nGPU Details:")
                print(f"   CUDA Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"   Device Name:    {torch.cuda.get_device_name(0)}")
                    print(f"   Allocated VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
                    print(f"   Reserved VRAM:  {torch.cuda.memory_reserved(0)/(1024**2):.1f} MB")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="IIIT-NR Multilingual STT Terminal Debugger")
    parser.add_argument("--record", type=float, help="Record from mic for N seconds and transcribe")
    parser.add_argument("--mic", action="store_true", help="Interactive press-Enter start/stop mic recording")
    parser.add_argument("--file", type=str, help="Path to WAV audio file to transcribe")
    parser.add_argument("--synth", type=str, help="Sentence to synthesize via TTS and transcribe")
    parser.add_argument("--lang", type=str, default="auto", choices=["auto", "hne", "hi", "en"], help="Target language (default: auto)")
    parser.add_argument("--status", action="store_true", help="Check STT server health and exit")

    args = parser.parse_args()

    if args.status:
        check_server_health(verbose=True)
        return

    if args.record:
        test_mic(duration=args.record, language=args.lang, interactive=False)
    elif args.mic:
        test_mic(language=args.lang, interactive=True)
    elif args.file:
        test_file(args.file, language=args.lang)
    elif args.synth:
        test_synth_roundtrip(args.synth, language=args.lang)
    else:
        # No arguments given -> launch interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()
