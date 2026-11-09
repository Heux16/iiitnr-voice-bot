"""
benchmark_25_test.py — Full-stack automated benchmark across 25 questions in 3 languages.
Tests STT, RAG (Retrieval + Prompting + LLM), and TTS pipelines.
"""

import os
import sys
import time
import json
import uuid
import re
import requests
import io
import wave
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.retrieval import DocumentRetriever
from app.config import RetrievalConfig
from app.prompt_builder import detect_query_language, build_prompt
from app.llm_client import call_llm
from app.answer_guard import diagnose_answer

STT_URL = "http://127.0.0.1:8001/transcribe"
TTS_SENTENCE_URL = "http://127.0.0.1:8000/stream_sentence"
TTS_FINISH_URL = "http://127.0.0.1:8000/stream_finish"

# ──────────────────────────────────────────────
# 25 Benchmark Questions across 3 Languages
# ──────────────────────────────────────────────

BENCHMARK_QUESTIONS = [
    # 🌾 Chhattisgarhi (8 questions)
    {
        "id": "CG-01",
        "lang": "chhattisgarhi",
        "question": "छात्रावास के फीस कतेक हे अउ मेस चार्ज कितना लगही?",
        "topic": "Hostel & Mess Fee",
    },
    {
        "id": "CG-02",
        "lang": "chhattisgarhi",
        "question": "एमटेक बर हॉस्टल के फीस कतेक हावे?",
        "topic": "M.Tech Hostel Fee",
    },
    {
        "id": "CG-03",
        "lang": "chhattisgarhi",
        "question": "जय जोहार, सीएसई म एडमिशन कइसे मिलही?",
        "topic": "CSE Admission Procedure",
    },
    {
        "id": "CG-04",
        "lang": "chhattisgarhi",
        "question": "IIIT नया रायपुर म बीटेक बर का का ब्रांच हावे?",
        "topic": "B.Tech Branches Available",
    },
    {
        "id": "CG-05",
        "lang": "chhattisgarhi",
        "question": "स्कॉलरशिप बर कतेक नंबर लाना पड़ही संगी?",
        "topic": "Scholarship Eligibility",
    },
    {
        "id": "CG-06",
        "lang": "chhattisgarhi",
        "question": "ब्रांच बदले बर कतेक सीपीआई जरूरी हे?",
        "topic": "Branch Sliding CPI",
    },
    {
        "id": "CG-07",
        "lang": "chhattisgarhi",
        "question": "हॉस्टल म मेस के सुविधा कइसन हे अउ खाना के व्यवस्था कइसे रहिथे?",
        "topic": "Mess Facilities",
    },
    {
        "id": "CG-08",
        "lang": "chhattisgarhi",
        "question": "hostel ke fees katek he aau mess charge kitna laghi",
        "topic": "Hostel Fees (Romanized Chhattisgarhi)",
    },

    # 🇮🇳 Hindi (8 questions)
    {
        "id": "HI-01",
        "lang": "hindi",
        "question": "एमटेक (डीएसएआई) के लिए प्रति सेमेस्टर छात्रावास का शुल्क कितना है?",
        "topic": "M.Tech Hostel Fee",
    },
    {
        "id": "HI-02",
        "lang": "hindi",
        "question": "आईआईआईटी नया रायपुर में बीटेक सीएसई में प्रवेश कैसे मिलता है?",
        "topic": "B.Tech CSE Admission",
    },
    {
        "id": "HI-03",
        "lang": "hindi",
        "question": "शाखा परिवर्तन (ब्रांच स्लाइडिंग) के लिए न्यूनतम सीपीआई की आवश्यकता क्या है?",
        "topic": "Branch Sliding Requirement",
    },
    {
        "id": "HI-04",
        "lang": "hindi",
        "question": "छात्रवृत्ति और सीएमआईटी फेलोशिप के नियम क्या हैं?",
        "topic": "Fellowship & Scholarship Rules",
    },
    {
        "id": "HI-05",
        "lang": "hindi",
        "question": "अकादमिक अवकाश (स्टडी लीव) लेने की प्रक्रिया और अधिकतम अवधि क्या है?",
        "topic": "Study Leave Duration",
    },
    {
        "id": "HI-06",
        "lang": "hindi",
        "question": "परीक्षा में बैठने के लिए न्यूनतम उपस्थिति (अटेंडेंस) कितनी अनिवार्य है?",
        "topic": "Attendance Rules",
    },
    {
        "id": "HI-07",
        "lang": "hindi",
        "question": "संस्थान का परिसर कहाँ स्थित है और इसका पता क्या है?",
        "topic": "Institute Location",
    },
    {
        "id": "HI-08",
        "lang": "hindi",
        "question": "btech admission ke liye minimum eligibility criteria kya hai aur josaa counselling kab hoti hai",
        "topic": "Admission Eligibility (Hinglish)",
    },

    # 🇬🇧 English (9 questions)
    {
        "id": "EN-01",
        "lang": "english",
        "question": "What is the fee for the hostel accommodation and mess charges?",
        "topic": "Hostel & Mess Fees",
    },
    {
        "id": "EN-02",
        "lang": "english",
        "question": "What is the procedure and maximum duration for academic leave of absence?",
        "topic": "Leave of Absence Policy",
    },
    {
        "id": "EN-03",
        "lang": "english",
        "question": "What are the eligibility criteria and fellowship amount for M.Tech DSAI?",
        "topic": "M.Tech DSAI Fellowship",
    },
    {
        "id": "EN-04",
        "lang": "english",
        "question": "What is the minimum CPI required for branch change after the first year?",
        "topic": "Branch Change Minimum CPI",
    },
    {
        "id": "EN-05",
        "lang": "english",
        "question": "What are the available academic programs and branches offered at IIIT-NR?",
        "topic": "Academic Programs",
    },
    {
        "id": "EN-06",
        "lang": "english",
        "question": "What are the attendance requirements to appear in the end semester examination?",
        "topic": "Attendance Requirement",
    },
    {
        "id": "EN-07",
        "lang": "english",
        "question": "Where is IIIT Naya Raipur campus located and how to reach there?",
        "topic": "Campus Location & Reach",
    },
    {
        "id": "EN-08",
        "lang": "english",
        "question": "What is the grading policy and how is Semester Performance Index calculated?",
        "topic": "Grading & SPI Calculation",
    },
    {
        "id": "EN-09",
        "lang": "english",
        "question": "Tell me about the vocational training programme hostel accommodation and charges.",
        "topic": "Vocational Training Hostel",
    },
]


def generate_dummy_speech_wav(duration_s: float = 1.0, freq: float = 220.0) -> bytes:
    """Generate simple WAV audio bytes for STT testing."""
    sample_rate = 16000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    # Apply fade in / fade out to avoid clicks
    fade = int(sample_rate * 0.05)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    int16_audio = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())
    return buf.getvalue()


def test_stt_layer(audio_bytes: bytes, target_lang: str) -> dict:
    """Test STT endpoint on port 8001."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{STT_URL}?language={target_lang}",
            data=audio_bytes,
            headers={"Content-Type": "audio/wav"},
            timeout=15,
        )
        latency_ms = round((time.time() - t0) * 1000, 1)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "status": "PASS",
                "latency_ms": latency_ms,
                "server_ms": data.get("inference_ms", 0),
                "detected_lang": data.get("detected_language", target_lang),
                "text": data.get("text", "").strip(),
            }
        return {
            "status": "FAIL",
            "latency_ms": latency_ms,
            "error": f"HTTP {resp.status_code}: {resp.text}",
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "error": str(e),
        }


def test_rag_layer(question: str, retriever: DocumentRetriever, config: RetrievalConfig) -> dict:
    """Test full RAG pipeline: retrieval, language detection, prompt building, LLM generation, guard evaluation."""
    t0 = time.time()
    outcome = retriever.retrieve(question, config=config)
    retrieval_time = round(time.time() - t0, 3)

    detected_lang = detect_query_language(outcome.original_query)
    sys_prompt, user_prompt = build_prompt(
        outcome.original_query, outcome.contexts, language=detected_lang
    )

    t_llm = time.time()
    response_text = call_llm(user_prompt, system_instruction=sys_prompt)
    llm_time = round(time.time() - t_llm, 3)

    context_pages = {p for c in outcome.contexts for p in c.page_numbers}
    guard_diag = diagnose_answer(response_text, context_pages)

    # Check citation present
    has_source = bool(re.search(r"📄\s*Source:\s*Page\s*\d+", response_text, re.IGNORECASE))

    # Top source excerpt
    top_source = "None"
    if outcome.contexts:
        top_source = f"Page {outcome.contexts[0].page_label} ({outcome.contexts[0].heading_label})"

    return {
        "status": "PASS" if (response_text and guard_diag is None) else "GUARD_FLAG",
        "retrieval_sec": retrieval_time,
        "llm_sec": llm_time,
        "total_rag_sec": round(retrieval_time + llm_time, 3),
        "contexts_count": len(outcome.contexts),
        "was_rewritten": outcome.was_rewritten,
        "effective_query": outcome.effective_query,
        "top_source": top_source,
        "detected_lang": detected_lang,
        "response_text": response_text.strip(),
        "has_citation": has_source,
        "guard_diagnostic": guard_diag,
    }


def test_tts_layer(response_text: str, session_id: str) -> dict:
    """Test TTS synthesis on port 8000."""
    t0 = time.time()
    # Clean text to sentences
    sentences = [s.strip() for s in re.split(r'([।?!\n]|\.\s+)', response_text) if s.strip() and len(s.strip()) > 3]
    if not sentences:
        sentences = [response_text.strip()]

    chunks_sent = 0
    try:
        for s in sentences[:4]:  # Stream up to first 4 sentences for benchmark
            requests.post(
                TTS_SENTENCE_URL,
                json={"session_id": session_id, "text": s, "voice": "Female"},
                timeout=5,
            )
            chunks_sent += 1

        fin_resp = requests.post(
            TTS_FINISH_URL,
            json={"session_id": session_id},
            timeout=10,
        )
        total_ms = round((time.time() - t0) * 1000, 1)

        if fin_resp.status_code == 200:
            fin_data = fin_resp.json()
            return {
                "status": "PASS",
                "chunks_sent": chunks_sent,
                "latency_ms": total_ms,
                "tts_status": fin_data.get("status"),
                "audio_len": len(fin_data.get("audio_base64") or ""),
            }
        return {
            "status": "FAIL",
            "latency_ms": total_ms,
            "error": f"HTTP {fin_resp.status_code}",
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "error": str(e),
        }


def run_benchmark():
    print("=" * 70)
    print("STARTING 25-QUESTION TRILINGUAL BENCHMARK (STT, RAG, TTS)")
    print("=" * 70)

    retriever = DocumentRetriever()
    config = RetrievalConfig()

    results = []

    # Pre-select known real audio files if available for STT
    audio_log_files = []
    log_dir = os.path.join(os.path.dirname(__file__), "STT", "stt-service", "logs", "audio")
    if os.path.exists(log_dir):
        audio_log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith("_processed.wav")]

    for i, item in enumerate(BENCHMARK_QUESTIONS, 1):
        q_id = item["id"]
        lang = item["lang"]
        q_text = item["question"]
        topic = item["topic"]

        print(f"\n[{i:02d}/25] {q_id} [{lang.upper()}] Topic: {topic}")
        print(f"       Question: {q_text[:70]}...")

        # 1. STT Test
        stt_target = "hne" if lang == "chhattisgarhi" else ("hi" if lang == "hindi" else "en")
        # Use real recorded WAV if available, otherwise generate dummy speech audio
        audio_sample = None
        if audio_log_files:
            sample_path = audio_log_files[(i - 1) % len(audio_log_files)]
            try:
                with open(sample_path, "rb") as af:
                    audio_sample = af.read()
            except Exception:
                pass
        if not audio_sample:
            audio_sample = generate_dummy_speech_wav(1.2)

        stt_res = test_stt_layer(audio_sample, stt_target)
        print(f"       🎙️ STT: {stt_res['status']} in {stt_res.get('latency_ms', 0)}ms (detected: {stt_res.get('detected_lang')})")

        # 2. RAG Test
        rag_res = test_rag_layer(q_text, retriever, config)
        print(f"       📚 RAG: {rag_res['status']} in {rag_res['total_rag_sec']}s | Lang: {rag_res['detected_lang']} | Citation: {rag_res['has_citation']}")
        print(f"       📄 Top Source: {rag_res['top_source']}")
        print(f"       💬 Response Excerpt: {rag_res['response_text'][:80]}...")

        # 3. TTS Test
        sess_id = uuid.uuid4().hex
        tts_res = test_tts_layer(rag_res["response_text"], sess_id)
        print(f"       🔊 TTS: {tts_res['status']} in {tts_res.get('latency_ms', 0)}ms | Status: {tts_res.get('tts_status')}")

        results.append({
            "id": q_id,
            "topic": topic,
            "expected_lang": lang,
            "question": q_text,
            "stt": stt_res,
            "rag": rag_res,
            "tts": tts_res,
        })

    # Save complete JSON report
    out_file = os.path.join(os.path.dirname(__file__), "benchmark_25_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE. Saved raw results to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()

