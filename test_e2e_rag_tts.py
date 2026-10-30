import sys
import os
import time

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

app_dir = r"d:\iiit nr\sem 5\New folder (2)\uajy-academic-rag-chatbot"
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

import requests
from app.retrieval import DocumentRetriever
from app.prompt_builder import build_prompt
from app.llm_client import call_llm
from app.citations import strip_inline_source_lines

print("=" * 60)
print("Testing RAG + GPU TTS End-to-End Pipeline")
print("=" * 60)

query = "who is abhishek sharma"
print(f"User Query: {query}")

t0 = time.time()
retriever = DocumentRetriever()
outcome = retriever.retrieve(query)
print(f"[1] Retrieved {len(outcome.contexts)} contexts in {time.time() - t0:.2f}s")
if outcome.contexts:
    print(f"    Top Source: {outcome.contexts[0].source_document} - {outcome.contexts[0].heading_label}")

t1 = time.time()
sys_prompt, user_prompt = build_prompt(query, outcome.contexts)
print(f"[2] Calling Gemini for Hindi response...")
answer = call_llm(user_prompt, system_instruction=sys_prompt)
print(f"    Response generated in {time.time() - t1:.2f}s:")
print(f"    --> {answer}")

display_answer = strip_inline_source_lines(answer)

t2 = time.time()
print(f"\n[3] Calling GPU TTS Server (http://127.0.0.1:8000/synthesize)...")
res = requests.post(
    "http://127.0.0.1:8000/synthesize",
    json={"text": display_answer, "voice": "Female"},
    timeout=15
)
print(f"    TTS Status Code: {res.status_code}")
if res.status_code == 200:
    wav_bytes = res.content
    print(f"    TTS Audio generated in {time.time() - t2:.3f}s! Size: {len(wav_bytes)} bytes")
    out_file = os.path.join(app_dir, "TTS", "rag_live_answer.wav")
    with open(out_file, "wb") as f:
        f.write(wav_bytes)
    print(f"    Saved audio to: {out_file}")
else:
    print(f"    TTS Failed: {res.text}")

print("=" * 60)
print(f"Total Pipeline Duration: {time.time() - t0:.2f}s")

