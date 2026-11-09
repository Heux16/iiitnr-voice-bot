"""
benchmark_200_test.py — Automated 200-Question Trilingual Benchmark for IIIT-NR Chatbot.
Tests STT, Language Detection, RAG (Retrieval + Prompting + LLM), Guardrails, and TTS.
Saves results incrementally to benchmark_200_results.json.
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
from app.prompt_builder import detect_query_language, build_prompt, build_no_context_response
from app.llm_client import call_llm
from app.answer_guard import diagnose_answer

STT_URL = "http://127.0.0.1:8001/transcribe"
TTS_SENTENCE_URL = "http://127.0.0.1:8000/stream_sentence"
TTS_FINISH_URL = "http://127.0.0.1:8000/stream_finish"

QUESTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "benchmark_200_questions.json")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_200_results.json")


def generate_dummy_speech_wav(duration_s: float = 1.0, freq: float = 220.0) -> bytes:
    """Generate simple WAV audio bytes for STT testing."""
    sample_rate = 16000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
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
            timeout=10,
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


def test_rag_layer(question: str, retriever: DocumentRetriever, config: RetrievalConfig, expected_behavior: str) -> dict:
    """Test full RAG pipeline: retrieval, language detection, prompt building, LLM generation, guard evaluation."""
    t0 = time.time()
    outcome = retriever.retrieve(question, config=config)
    retrieval_time = round(time.time() - t0, 3)

    detected_lang = detect_query_language(outcome.original_query)

    t_llm = time.time()
    if outcome.refused:
        response_text = build_no_context_response(language=detected_lang)
    else:
        sys_prompt, user_prompt = build_prompt(
            outcome.original_query, outcome.contexts, language=detected_lang
        )
        response_text = call_llm(user_prompt, system_instruction=sys_prompt)
    llm_time = round(time.time() - t_llm, 3)

    context_pages = {p for c in outcome.contexts for p in c.page_numbers}
    guard_diag = diagnose_answer(response_text, context_pages)

    has_source = bool(re.search(r"(?:📄|Source:)\s*(?:Page|पृष्ठ)\s*\d+", response_text, re.IGNORECASE))
    
    # Check if refusal was detected
    is_refusal = any(phrase in response_text for phrase in [
        "केवल उपलब्ध शैक्षणिक", "जानकारी उपलब्ध दस्तावेजों में नहीं", "मोला माफ करहू",
        "नइ मिले हे", "I apologize", "not found in the available", "I can only answer"
    ])

    top_source = "None"
    if outcome.contexts:
        top_source = f"Page {outcome.contexts[0].page_label} ({outcome.contexts[0].heading_label})"

    return {
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
        "is_refusal": is_refusal,
        "guard_diagnostic": guard_diag,
        "refused_at_retrieval": outcome.refused,
    }


def test_tts_layer(response_text: str, session_id: str) -> dict:
    """Test TTS synthesis on port 8000."""
    t0 = time.time()
    sentences = [s.strip() for s in re.split(r'([।?!\n]|\.\s+)', response_text) if s.strip() and len(s.strip()) > 3]
    if not sentences:
        sentences = [response_text.strip()]

    chunks_sent = 0
    try:
        for s in sentences[:3]:  # Stream first 3 sentences for benchmark
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
                "total_dur_sec": fin_data.get("total_duration", 0.0),
                "audio_bytes_len": len(fin_data.get("audio_base64") or ""),
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
    print("=" * 75)
    print("STARTING 200-QUESTION TRILINGUAL BENCHMARK (STT, RAG, TTS)")
    print("=" * 75)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} test questions.")

    retriever = DocumentRetriever()
    config = RetrievalConfig()

    dummy_audio = generate_dummy_speech_wav(duration_s=1.2, freq=240.0)

    # Resume from checkpoint if exists
    results = []
    completed_ids = set()
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as rf:
                results = json.load(rf)
                completed_ids = {r["id"] for r in results}
                print(f"Found existing results: {len(results)} questions already evaluated. Resuming...")
        except Exception as e:
            print(f"Warning reading existing results: {e}. Starting fresh.")
            results = []

    start_total_time = time.time()

    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        if qid in completed_ids:
            continue

        target_lang = q["lang"]
        stt_code = "hne" if target_lang == "chhattisgarhi" else ("hi" if target_lang == "hindi" else "en")
        
        t_start_q = time.time()

        # 1. STT Layer Test
        stt_res = test_stt_layer(dummy_audio, target_lang=stt_code)

        # 2. RAG Layer Test
        rag_res = test_rag_layer(q["question"], retriever, config, q["expected_behavior"])

        # 3. TTS Layer Test
        session_id = uuid.uuid4().hex
        tts_res = test_tts_layer(rag_res["response_text"], session_id=session_id)

        total_q_time = round(time.time() - t_start_q, 2)

        # Language Detection Accuracy Check
        lang_correct = (rag_res["detected_lang"] == target_lang)

        # Verdict Classification
        if q["expected_behavior"] == "refusal":
            if rag_res["is_refusal"] or rag_res["refused_at_retrieval"]:
                verdict = "SAFE_REFUSAL"
            else:
                verdict = "UNGROUNDED_ANSWER_ON_OOS"
        else:
            if rag_res["guard_diagnostic"] is None and rag_res["has_citation"]:
                verdict = "PASS"
            elif rag_res["guard_diagnostic"] is not None:
                diag_code = rag_res["guard_diagnostic"][0] if isinstance(rag_res["guard_diagnostic"], tuple) else str(rag_res["guard_diagnostic"])
                verdict = f"GUARD_FLAG ({diag_code})"
            elif not rag_res["has_citation"]:
                verdict = "MISSING_CITATION"
            else:
                verdict = "RETRIEVAL_MISS"

        res_entry = {
            "id": qid,
            "lang": target_lang,
            "category": q["category"],
            "expected_behavior": q["expected_behavior"],
            "topic": q["topic"],
            "question": q["question"],
            "detected_lang": rag_res["detected_lang"],
            "lang_correct": lang_correct,
            "stt_status": stt_res.get("status", "FAIL"),
            "stt_latency_ms": stt_res.get("latency_ms", 0),
            "retrieval_sec": rag_res["retrieval_sec"],
            "llm_sec": rag_res["llm_sec"],
            "total_rag_sec": rag_res["total_rag_sec"],
            "contexts_count": rag_res["contexts_count"],
            "top_source": rag_res["top_source"],
            "effective_query": rag_res["effective_query"],
            "was_rewritten": rag_res["was_rewritten"],
            "response_text": rag_res["response_text"],
            "has_citation": rag_res["has_citation"],
            "is_refusal": rag_res["is_refusal"],
            "guard_diagnostic": str(rag_res["guard_diagnostic"]),
            "tts_status": tts_res.get("status", "FAIL"),
            "tts_latency_ms": tts_res.get("latency_ms", 0),
            "tts_dur_sec": tts_res.get("total_dur_sec", 0.0),
            "audio_bytes_len": tts_res.get("audio_bytes_len", 0),
            "total_latency_sec": total_q_time,
            "verdict": verdict,
        }

        results.append(res_entry)

        # Print progress
        status_sym = "✅" if verdict in ("PASS", "SAFE_REFUSAL") else "⚠️"
        print(f"[{len(results)}/200] {status_sym} {qid} ({target_lang[:2].upper()}): {verdict} | RAG: {rag_res['total_rag_sec']}s, TTS: {tts_res.get('latency_ms', 0)}ms, Total: {total_q_time}s | Topic: {q['topic']}")

        # Save checkpoint every 5 questions
        if len(results) % 5 == 0 or len(results) == len(questions):
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    total_bench_time = round(time.time() - start_total_time, 1)
    print("=" * 75)
    print(f"BENCHMARK COMPLETED IN {total_bench_time}s")
    print("=" * 75)

    # Summary Statistics
    total_q = len(results)
    pass_count = sum(1 for r in results if r["verdict"] in ("PASS", "SAFE_REFUSAL"))
    refusal_count = sum(1 for r in results if r["verdict"] == "SAFE_REFUSAL")
    lang_acc = sum(1 for r in results if r["lang_correct"]) / total_q * 100
    stt_pass = sum(1 for r in results if r["stt_status"] == "PASS") / total_q * 100
    tts_pass = sum(1 for r in results if r["tts_status"] == "PASS") / total_q * 100

    avg_rag = np.mean([r["total_rag_sec"] for r in results])
    avg_tts = np.mean([r["tts_latency_ms"] for r in results])
    avg_total = np.mean([r["total_latency_sec"] for r in results])

    print(f"Total Questions: {total_q}")
    print(f"Overall Accuracy: {pass_count}/{total_q} ({pass_count/total_q*100:.1f}%)")
    print(f"Safe Refusal Compliance: {refusal_count}/23")
    print(f"Language Detection Accuracy: {lang_acc:.1f}%")
    print(f"STT Service Pass Rate: {stt_pass:.1f}%")
    print(f"TTS Service Pass Rate: {tts_pass:.1f}%")
    print(f"Average Latency: RAG={avg_rag:.2f}s, TTS={avg_tts:.0f}ms, Total={avg_total:.2f}s")


if __name__ == "__main__":
    run_benchmark()
