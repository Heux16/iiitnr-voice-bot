import os
import sys
import time

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

candidates = [
    os.path.join(parent_dir, ".streamlit", "secrets.toml"),
    os.path.join(parent_dir, "uajy-academic-rag-chatbot", ".streamlit", "secrets.toml")
]
secrets_path = next((p for p in candidates if os.path.exists(p)), None)
gemini_key = None
if secrets_path and os.path.exists(secrets_path):
    with open(secrets_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip().startswith("GEMINI_API_KEY"):
            gemini_key = line.split("=")[1].strip().strip('"').strip("'")

client = genai.Client(api_key=gemini_key)

english_answer = (
    "Dr. Abhishek Sharma is an Assistant Professor in the Electronics and Communication "
    "Engineering discipline at IIIT-NR. He earned his Ph.D. in Signal Processing from IIT Guwahati."
)

print("[1] Translating English answer to Chhattisgarhi Devanagari...")
t0 = time.time()
prompt = (
    "You are an expert in the Chhattisgarhi language. "
    "Translate this academic explanation into clear, natural Chhattisgarhi written in Devanagari script. "
    "Keep it concise (1-2 sentences). Do not use any English words or Latin script:\n\n"
    f"{english_answer}"
)
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)
chhattisgarhi_text = response.text.strip()
print(f"Translated in {time.time() - t0:.2f}s:\n-> {chhattisgarhi_text}")

print("\n[2] Synthesizing speech on NVIDIA RTX 4050 GPU...")
tts = StreamingChhattisgarhiTTS(voice="Female", use_cuda=True, length_scale=1.15)

t1 = time.time()
wav = tts.synthesizer.tts(text=chhattisgarhi_text)
tts_time = time.time() - t1
audio_len = len(wav) / 22050.0
print(f"Synthesized {audio_len:.2f}s of audio in {tts_time:.3f}s (Speedup: {audio_len/tts_time:.1f}x real-time)")

tts.synthesizer.save_wav(wav, os.path.join(current_dir, "test_academic_voice.wav"))
print("Saved to test_academic_voice.wav")

