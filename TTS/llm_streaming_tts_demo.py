"""
End-to-End Demo: LLM Streaming Generation + Simultaneous GPU TTS Synthesis.
- LLM: Google Gemini 3.5 Flash Lite (streams text tokens in real time)
- TTS: Coqui VITS Chhattisgarhi Model on NVIDIA RTX 4050 GPU (CUDA)
- Simultaneous: Pipeline generates audio chunks as tokens arrive without waiting for the full response.
"""

import os
import sys
import time
import queue
import threading
import re
import numpy as np
import scipy.io.wavfile as wavfile

# Fix Windows stdout encoding for Hindi / Chhattisgarhi Devanagari text
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from google import genai
from streaming_tts import StreamingChhattisgarhiTTS

def load_gemini_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    candidates = [
        os.path.join(parent_dir, ".streamlit", "secrets.toml"),
        os.path.join(parent_dir, "uajy-academic-rag-chatbot", ".streamlit", "secrets.toml")
    ]
    secrets_path = next((p for p in candidates if os.path.exists(p)), None)
    if secrets_path and os.path.exists(secrets_path):
        with open(secrets_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("GEMINI_API_KEY"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        return parts[1].strip().strip('"').strip("'")
    return None

gemini_api_key = load_gemini_key()

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment or secrets.toml!")

client = genai.Client(api_key=gemini_api_key)


def main():
    print("=" * 70)
    print("🚀 LLM STREAMING TO CHHATTISGARHI TTS (GPU-ACCELERATED)")
    print("=" * 70)

    # Initialize Streaming TTS Engine on GPU
    tts_engine = StreamingChhattisgarhiTTS(
        voice="Female",
        use_cuda=True,
        length_scale=1.15  # Natural pacing
    )

    user_prompt = "छत्तीसबढ़ के बारे म 2-3 लाइन म सुंदर बात बतावा"
    system_instruction = (
        "You are an assistant speaking in Chhattisgarhi written in Devanagari script. "
        "Provide a concise, poetic 2-3 sentence description in pure Chhattisgarhi."
    )

    print(f"\nUser Prompt: {user_prompt}")
    print("\n[1] Starting Gemini Token Stream...")
    
    t_start = time.time()
    response_stream = client.models.generate_content_stream(
        model="gemini-3.5-flash-lite",
        contents=f"{system_instruction}\n\nPrompt: {user_prompt}"
    )

    # Queue to feed streaming tokens into TTS while printing simultaneously
    token_queue = queue.Queue()
    sentinel = object()

    def token_generator():
        while True:
            t = token_queue.get()
            if t is sentinel:
                break
            yield t

    # Start TTS pipeline in background thread
    synthesized_audio_chunks = []
    first_audio_time = None

    def tts_consumer():
        nonlocal first_audio_time
        for sentence_text, audio_arr in tts_engine.synthesize_stream(token_generator()):
            now = time.time()
            if first_audio_time is None:
                first_audio_time = now - t_start
                print(f"\n\n⚡ [TTFA - Time To First Audio]: {first_audio_time:.3f}s from stream start!")
            
            print(f"\n🎙️ [Synthesized Chunk]: \"{sentence_text}\" ({len(audio_arr)/22050:.2f}s audio)")
            synthesized_audio_chunks.append(audio_arr)

    tts_thread = threading.Thread(target=tts_consumer, daemon=True)
    tts_thread.start()

    # Stream tokens from Gemini
    print("\n📝 [Live LLM Output]: ", end="", flush=True)
    full_text = ""
    for chunk in response_stream:
        if chunk.text:
            text_piece = chunk.text
            full_text += text_piece
            print(text_piece, end="", flush=True)
            token_queue.put(text_piece)

    token_queue.put(sentinel)
    tts_thread.join()

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("📊 [BENCHMARK & PERFORMANCE SUMMARY]")
    print(f"Total Pipeline Time: {total_time:.2f}s")
    if first_audio_time:
        print(f"Time to First Playable Audio: {first_audio_time:.3f}s")

    if synthesized_audio_chunks:
        full_audio = np.concatenate(synthesized_audio_chunks)
        audio_dur = len(full_audio) / 22050.0
        print(f"Total Speech Audio Duration: {audio_dur:.2f}s")
        print(f"Synthesis Speed: {audio_dur / total_time:.1f}x real-time!")

        output_wav_path = os.path.join(current_dir, "output_streamed.wav")
        # Normalize to 16-bit PCM for universal audio player compatibility
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            norm_audio = (full_audio / max_val * 32767).astype(np.int16)
        else:
            norm_audio = (full_audio * 32767).astype(np.int16)
        
        wavfile.write(output_wav_path, 22050, norm_audio)
        print(f"💾 Combined Audio Saved To: {output_wav_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
