# Real-Time LLM Streaming to Chhattisgarhi TTS (GPU-Accelerated)

This module connects any streaming LLM (e.g. Google Gemini 3.5 Flash Lite) to the local **Coqui VITS Chhattisgarhi TTS** model running on the **NVIDIA GeForce RTX 4050 GPU**.

---

## ⚡ Performance Benchmark (Verified on NVIDIA RTX 4050)

- **Synthesis Speed:** **7.0x – 27.3x real-time!**
- **Inference Latency:** Only **0.14s – 0.33s** per full sentence.
- **Time To First Audio (TTFA):** **~1.4 seconds** from the moment the user sends the prompt.
- **Audio Format:** 22,050 Hz, 16-bit PCM WAV.

---

## 🛠️ Environment & Prerequisites

- **Conda Environment:** `minor1` (Python 3.10 with PyTorch 2.5.1 + CUDA 12.1 and Coqui TTS).
- **Interpreter Path:** `C:\Users\ASUS\anaconda3\envs\minor1\python.exe`

---

## 🚀 How to Run the End-to-End Demo

Run the end-to-end streaming demo from your terminal:

```powershell
C:\Users\ASUS\anaconda3\envs\minor1\python.exe "d:\iiit nr\sem 5\New folder (2)\TTS\llm_streaming_tts_demo.py"
```

What it does:

1. Queries Gemini (`gemini-3.5-flash-lite`) using your existing API key.
2. Streams tokens live into the terminal.
3. Concurrently synthesizes audio sentences on the RTX 4050 GPU without waiting for Gemini to finish.
4. Saves the complete synchronized audio to `output_streamed.wav`.

---

## 💻 Python Code Usage (Direct Integration)

```python
from streaming_tts import StreamingChhattisgarhiTTS
from google import genai

# 1. Initialize engine on GPU (weights stay hot in VRAM)
tts = StreamingChhattisgarhiTTS(
    voice="Female",     # Or "Male"
    use_cuda=True,      # Uses RTX 4050 GPU
    length_scale=1.15   # Natural speaking speed
)

# 2. Start LLM stream
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
stream = client.models.generate_content_stream(
    model="gemini-3.5-flash-lite",
    contents="छत्तीसबढ़ के बारे म बतावा"
)

def token_gen():
    for chunk in stream:
        if chunk.text:
            yield chunk.text

# 3. Stream text and receive audio chunks simultaneously
for sentence_text, audio_array in tts.synthesize_stream(token_gen()):
    print(f"Synthesized: {sentence_text} (Duration: {len(audio_array)/22050:.2f}s)")
    # Play or stream audio_array immediately!
```

---

## 🌐 FastAPI Microservice (Optional)

If your frontend runs in another process (e.g. Streamlit or React), start the FastAPI server:

```powershell
C:\Users\ASUS\anaconda3\envs\minor1\python.exe "d:\iiit nr\sem 5\New folder (2)\TTS\tts_server.py"
```

Endpoint available at: `http://127.0.0.1:8000/synthesize_chunk`
