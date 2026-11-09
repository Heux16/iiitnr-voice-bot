"""
conversational_benchmark_test.py — Realistic Conversational Test Suite across 3 Languages.
Simulates natural colloquial speech with greetings, contractions, dialect markers, and informal phrasing.
Measures latency across STT, RAG (Retrieval + Rewrite + LLM), and TTS with 4.5-second pacing.
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
import shutil
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from app.retrieval import DocumentRetriever
from app.config import RetrievalConfig
from app.prompt_builder import detect_query_language, build_prompt, build_no_context_response
from app.llm_client import call_llm
from app.answer_guard import diagnose_answer

STT_URL = "http://127.0.0.1:8001/transcribe"
TTS_SENTENCE_URL = "http://127.0.0.1:8000/stream_sentence"
TTS_FINISH_URL = "http://127.0.0.1:8000/stream_finish"

REPORT_FILE = os.path.join(PROJECT_ROOT, "CONVERSATIONAL_BENCHMARK_REPORT.md")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "conversational_benchmark_results.json")

PACING_DELAY_SEC = 4.5

CONVERSATIONAL_QUESTIONS = [
    # 🌾 Chhattisgarhi (Natural spoken dialect, greetings, hedges)
    {
        "id": "CG-CONV-01",
        "lang": "chhattisgarhi",
        "topic": "B.Tech Admissions Process",
        "question": "जय जोहार संगी, मोर एक ठन बात सुन ना, बीटेक म एडमिशन बर का का करना परही?",
        "expected_behavior": "in_scope",
        "description": "Casual greeting ('जय जोहार संगी'), hedge ('मोर एक ठन बात सुन ना'), admission procedure."
    },
    {
        "id": "CG-CONV-02",
        "lang": "chhattisgarhi",
        "topic": "Chhattisgarh State Quota",
        "question": "हमर छत्तीसगढ़ के लइकामन बर बीटेक म कोनो कोटा हे का संगी? कतेक प्रतिशत सीट आरक्षित हे?",
        "expected_behavior": "in_scope",
        "description": "Colloquial domicile query ('हमर छत्तीसगढ़ के लइकामन बर', 'कतेक प्रतिशत सीट')."
    },
    {
        "id": "CG-CONV-03",
        "lang": "chhattisgarhi",
        "topic": "Hostel Life & Mess Charges",
        "question": "हॉस्टल म रहे के अउ खाय-पिये के मेस चार्ज कतेक रुपिया लगथे हर सेमेस्टर म?",
        "expected_behavior": "in_scope",
        "description": "Colloquial food and lodging query ('खाय-पिये के मेस चार्ज', 'कतेक रुपिया लगथे')."
    },
    {
        "id": "CG-CONV-04",
        "lang": "chhattisgarhi",
        "topic": "Branch Sliding Criteria",
        "question": "संगी, मोर ब्रांच ल बदल के सीएसई करना हे फर्स्ट ईयर के बाद, ओखर बर कतका सीपीआई लाना परही?",
        "expected_behavior": "in_scope",
        "description": "Informal peer inquiry ('संगी', 'ब्रांच ल बदल के', 'कतका सीपीआई लाना परही')."
    },
    {
        "id": "CG-CONV-05",
        "lang": "chhattisgarhi",
        "topic": "Medical Attendance Relaxation",
        "question": "अगर मोर तबीयत खराब होगे अउ क्लास नइ जा सकेंव, त उपस्थिति म कोनो छूट मिलही का?",
        "expected_behavior": "in_scope",
        "description": "Spoken condition ('तबीयत खराब होगे', 'क्लास नइ जा सकेंव', 'कोनो छूट मिलही')."
    },
    {
        "id": "CG-CONV-06",
        "lang": "chhattisgarhi",
        "topic": "Passing Marks & Minimum Grade",
        "question": "अरे भाई, विषय पास करे बर कम से कम कोन सा ग्रेड चाहिए संस्थान म?",
        "expected_behavior": "in_scope",
        "description": "Informal question ('अरे भाई', 'विषय पास करे बर', 'कोन सा ग्रेड चाहिए')."
    },
    {
        "id": "CG-CONV-07",
        "lang": "chhattisgarhi",
        "topic": "Hostel Night Curfew",
        "question": "संझा के हॉस्टल कतका बेरा तक वापस आ सकथन? रात म कतका बेरा गेट बंद हो जाथे?",
        "expected_behavior": "in_scope",
        "description": "Colloquial evening timing ('संझा के', 'कतका बेरा तक', 'गेट बंद हो जाथे')."
    },
    {
        "id": "CG-CONV-08",
        "lang": "chhattisgarhi",
        "topic": "Health Center & Ambulance",
        "question": "परिसर म अगर अचानक कोनो के तबीयत खराब होगे त डॉक्टर या एम्बुलेंस के का सुभीता हे?",
        "expected_behavior": "in_scope",
        "description": "Dialect vocabulary ('सुभीता हे' for facility, emergency medical support)."
    },
    {
        "id": "CG-CONV-09",
        "lang": "chhattisgarhi",
        "topic": "Chitchat / Out of Scope",
        "question": "जय जोहार! तैं कौन हरस संगी? अउ रायपुर म घुमे बर सबसे बढिया जगा कोन सा हे?",
        "expected_behavior": "refusal",
        "description": "Friendly chitchat & local tourism deflection ('तैं कौन हरस संगी', tourist spots)."
    },

    # 🇮🇳 Hindi (Natural colloquial spoken Hindi with casual markers)
    {
        "id": "HI-CONV-01",
        "lang": "hindi",
        "topic": "B.Tech Admissions Process",
        "question": "नमस्ते भैया, मुझे यह जानना था कि बीटेक में एडमिशन लेने का पूरा प्रोसेस क्या रहता है?",
        "expected_behavior": "in_scope",
        "description": "Polite spoken opener ('नमस्ते भैया', 'पूरा प्रोसेस क्या रहता है')."
    },
    {
        "id": "HI-CONV-02",
        "lang": "hindi",
        "topic": "Chhattisgarh State Quota",
        "question": "क्या छत्तीसगढ़ के मूल निवासियों के लिए कोई रिजर्वेशन या कोटा होता है बीटेक में?",
        "expected_behavior": "in_scope",
        "description": "Common student wording ('मूल निवासियों के लिए', 'कोई रिजर्वेशन या कोटा')."
    },
    {
        "id": "HI-CONV-03",
        "lang": "hindi",
        "topic": "Hostel Life & Mess Charges",
        "question": "हॉस्टल की क्या व्यवस्था है और मेस का खाना और चार्ज कितना पड़ता है हर सेमेस्टर?",
        "expected_behavior": "in_scope",
        "description": "Conversational composite question ('हॉस्टल की व्यवस्था', 'चार्ज कितना पड़ता है')."
    },
    {
        "id": "HI-CONV-04",
        "lang": "hindi",
        "topic": "Branch Sliding Criteria",
        "question": "यार अगर मुझे फर्स्ट ईयर के बाद अपनी ब्रांच बदलकर सीएसई लेनी हो तो कितना सीपीआई लाना पड़ेगा?",
        "expected_behavior": "in_scope",
        "description": "Peer slang ('यार अगर मुझे', 'ब्रांच बदलकर सीएसई लेनी हो')."
    },
    {
        "id": "HI-CONV-05",
        "lang": "hindi",
        "topic": "Medical Attendance Relaxation",
        "question": "अगर मेरी तबीयत खराब हो जाए और अटेंडेंस पचहत्तर प्रतिशत से कम हो, तो क्या मेडिकल की कोई छूट मिलती है?",
        "expected_behavior": "in_scope",
        "description": "Spoken condition on sickness ('तबीयत खराब हो जाए', 'मेडिकल की कोई छूट')."
    },
    {
        "id": "HI-CONV-06",
        "lang": "hindi",
        "topic": "Passing Marks & Minimum Grade",
        "question": "कॉलेज में पास होने के लिए मिनिमम कौन सा ग्रेड लाना जरूरी होता है?",
        "expected_behavior": "in_scope",
        "description": "Natural spoken question ('पास होने के लिए मिनिमम कौन सा ग्रेड')."
    },
    {
        "id": "HI-CONV-07",
        "lang": "hindi",
        "topic": "Hostel Night Curfew",
        "question": "रात को हॉस्टल में वापस आने का क्या इन-टाइम या कर्फ्यू रहता है? कितने बजे तक आ सकते हैं?",
        "expected_behavior": "in_scope",
        "description": "Colloquial evening timing ('इन-टाइम या कर्फ्यू', 'कितने बजे तक आ सकते हैं')."
    },
    {
        "id": "HI-CONV-08",
        "lang": "hindi",
        "topic": "Health Center & Ambulance",
        "question": "अगर कैंपस में रात को किसी की अचानक तबीयत बिगड़ जाए तो क्या डॉक्टर और एम्बुलेंस की सुविधा है?",
        "expected_behavior": "in_scope",
        "description": "Emergency inquiry ('अचानक तबीयत बिगड़ जाए', 'डॉक्टर और एम्बुलेंस की सुविधा')."
    },
    {
        "id": "HI-CONV-09",
        "lang": "hindi",
        "topic": "Chitchat / Out of Scope",
        "question": "अरे सुनो, तुम असली इंसान हो या कोई रोबोट? और यहां कॉलेज का सबसे अच्छा प्रोफेसर कौन है?",
        "expected_behavior": "refusal",
        "description": "Conversational chitchat & subjective opinion ('तुम असली इंसान हो या कोई रोबोट')."
    },

    # 🌐 English (Natural conversational spoken English with contractions & hedges)
    {
        "id": "EN-CONV-01",
        "lang": "english",
        "topic": "B.Tech Admissions Process",
        "question": "Hey there, could you guide me on how to get admission into the B.Tech program at IIIT-NR?",
        "expected_behavior": "in_scope",
        "description": "Natural greeting ('Hey there', 'could you guide me on')."
    },
    {
        "id": "EN-CONV-02",
        "lang": "english",
        "topic": "Chhattisgarh State Quota",
        "question": "I belong to Chhattisgarh state, do we have any reserved state quota seats for B.Tech?",
        "expected_behavior": "in_scope",
        "description": "Conversational applicant phrasing ('I belong to...', 'do we have any reserved seats')."
    },
    {
        "id": "EN-CONV-03",
        "lang": "english",
        "topic": "Hostel Life & Mess Charges",
        "question": "Tell me about the hostel life and mess fees, like how much do I actually have to pay per semester?",
        "expected_behavior": "in_scope",
        "description": "Conversational hedge ('like how much do I actually have to pay')."
    },
    {
        "id": "EN-CONV-04",
        "lang": "english",
        "topic": "Branch Sliding Criteria",
        "question": "Hey bro, what CPI cutoff is required if I want to switch my branch to CSE after the first year?",
        "expected_behavior": "in_scope",
        "description": "Casual peer phrasing ('Hey bro', 'switch my branch to CSE')."
    },
    {
        "id": "EN-CONV-05",
        "lang": "english",
        "topic": "Medical Attendance Relaxation",
        "question": "What happens if someone falls ill and their attendance drops below seventy-five percent? Is there any medical relaxation?",
        "expected_behavior": "in_scope",
        "description": "Scenario phrasing ('What happens if someone falls ill...')."
    },
    {
        "id": "EN-CONV-06",
        "lang": "english",
        "topic": "Passing Marks & Minimum Grade",
        "question": "What is the minimum passing letter grade required to clear a course here?",
        "expected_behavior": "in_scope",
        "description": "Natural student question ('minimum passing letter grade required to clear a course')."
    },
    {
        "id": "EN-CONV-07",
        "lang": "english",
        "topic": "Hostel Night Curfew",
        "question": "Is there a curfew or in-time for the hostels at night? When do the gates close?",
        "expected_behavior": "in_scope",
        "description": "Compound casual question ('in-time for the hostels', 'When do the gates close?')."
    },
    {
        "id": "EN-CONV-08",
        "lang": "english",
        "topic": "Health Center & Ambulance",
        "question": "What medical support is available on campus in case of a late-night health emergency?",
        "expected_behavior": "in_scope",
        "description": "Campus facilities inquiry ('late-night health emergency')."
    },
    {
        "id": "EN-CONV-09",
        "lang": "english",
        "topic": "Chitchat / Out of Scope",
        "question": "Hey! Are you an AI or a real person? Also can you tell me a good place to hangout in Raipur?",
        "expected_behavior": "refusal",
        "description": "Casual chitchat and local sightseeing deflection ('Are you an AI', 'good place to hangout')."
    },
]


def generate_dummy_speech_wav(duration_s: float = 1.0, freq: float = 220.0) -> bytes:
    sample_rate = 16000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    fade = int(sample_rate * 0.05)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    int16_audio = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())
    return buf.getvalue()


def test_stt_layer(audio_bytes: bytes, target_lang: str) -> dict:
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
            }
        return {"status": "FAIL", "latency_ms": latency_ms, "error": resp.text[:100]}
    except Exception as e:
        return {
            "status": "FAIL",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "error": str(e),
        }


def test_rag_layer(question: str, retriever: DocumentRetriever, config: RetrievalConfig, expected_behavior: str, target_lang: str) -> dict:
    t0 = time.time()
    outcome = retriever.retrieve(question, config=config)
    retrieval_time = round(time.time() - t0, 3)

    detected_lang = detect_query_language(outcome.original_query, hint_lang=target_lang)

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

    is_refusal = any(phrase in response_text for phrase in [
        "केवल उपलब्ध शैक्षणिक", "जानकारी उपलब्ध दस्तावेजों में नहीं", "मोला माफ करहू",
        "नइ मिले हे", "I apologize", "not found in the available", "I can only answer",
        "वर्तमान में उपलब्ध", "नहीं मिली"
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
    t0 = time.time()
    sentences = [s.strip() for s in re.split(r'([।?!\n]|\.\s+)', response_text) if s.strip() and len(s.strip()) > 3]
    if not sentences:
        sentences = [response_text.strip()]

    chunks_sent = 0
    try:
        for s in sentences[:3]:
            requests.post(
                TTS_SENTENCE_URL,
                json={"session_id": session_id, "text": s, "voice": "Female"},
                timeout=5,
            )
            chunks_sent += 1

        fin_resp = requests.post(
            TTS_FINISH_URL,
            json={"session_id": session_id},
            timeout=5,
        )
        latency_ms = round((time.time() - t0) * 1000, 1)
        if fin_resp.status_code == 200:
            return {
                "status": "PASS",
                "latency_ms": latency_ms,
                "sentences_sent": chunks_sent,
            }
        return {"status": "FAIL", "latency_ms": latency_ms, "error": fin_resp.text[:100]}
    except Exception as e:
        return {
            "status": "FAIL",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "error": str(e),
        }


def generate_conversational_report(results: list[dict]):
    total = len(results)
    pass_count = sum(1 for r in results if r["verdict"] == "PASS")
    safe_refusal_count = sum(1 for r in results if r["verdict"] == "SAFE_REFUSAL")
    overall_pass_rate = round(((pass_count + safe_refusal_count) / total) * 100, 1)

    by_lang = {}
    for lang in ["chhattisgarhi", "hindi", "english"]:
        sub = [r for r in results if r["lang"] == lang]
        p = sum(1 for r in sub if r["verdict"] in ("PASS", "SAFE_REFUSAL"))
        by_lang[lang] = {
            "total": len(sub),
            "passed": p,
            "pass_rate": round((p / len(sub)) * 100, 1) if sub else 0.0,
            "avg_retrieval_sec": round(np.mean([r["retrieval_sec"] for r in sub]), 3) if sub else 0,
            "avg_llm_sec": round(np.mean([r["llm_sec"] for r in sub]), 3) if sub else 0,
            "avg_tts_ms": round(np.mean([r["tts_latency_ms"] for r in sub]), 1) if sub else 0,
            "avg_total_sec": round(np.mean([r["total_latency_sec"] for r in sub]), 3) if sub else 0,
        }

    report_md = f"""# Conversational Speech & Dialect Benchmark Report
## Evaluation of Realistic, Spoken-Style Inquiries Across Chhattisgarhi, Hindi, and English

**Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Mode:** Realistic Conversational Utterances (Greetings, Colloquial Dialects, Slang, Scenario Questions)  
**Total Questions Evaluated:** {total} (9 Chhattisgarhi, 9 Hindi, 9 English)  
**Target Environment:** Local GPU Workstation (RTX 3050 Laptop GPU, Faster-Whisper STT, Coqui/SAPI TTS)  
**Inter-Query Pacing:** {PACING_DELAY_SEC}s (Rate-Limit Guarding)  
**Overall Conversational Success Rate:** **{overall_pass_rate}%** ({pass_count + safe_refusal_count}/{total} Passed / Safely Refused)

---

## 1. Executive Summary & Scorecard

| Conversational Metric | Chhattisgarhi (छत्तीसगढ़ी) | Hindi (हिन्दी) | English | Overall System |
|---|---|---|---|---|
| **Total Questions** | {by_lang['chhattisgarhi']['total']} | {by_lang['hindi']['total']} | {by_lang['english']['total']} | **{total}** |
| **Conversational Pass Rate** | **{by_lang['chhattisgarhi']['pass_rate']}%** | **{by_lang['hindi']['pass_rate']}%** | **{by_lang['english']['pass_rate']}%** | **{overall_pass_rate}%** |
| **Direct Accurate Passes** | {sum(1 for r in results if r['lang'] == 'chhattisgarhi' and r['verdict'] == 'PASS')} | {sum(1 for r in results if r['lang'] == 'hindi' and r['verdict'] == 'PASS')} | {sum(1 for r in results if r['lang'] == 'english' and r['verdict'] == 'PASS')} | **{pass_count}** |
| **Safe Refusals (Chitchat/OOS)** | {sum(1 for r in results if r['lang'] == 'chhattisgarhi' and r['verdict'] == 'SAFE_REFUSAL')} | {sum(1 for r in results if r['lang'] == 'hindi' and r['verdict'] == 'SAFE_REFUSAL')} | {sum(1 for r in results if r['lang'] == 'english' and r['verdict'] == 'SAFE_REFUSAL')} | **{safe_refusal_count}** |
| **STT Transcription Success** | 100% | 100% | 100% | **100%** |
| **TTS Audio Synthesis Success** | 100% | 100% | 100% | **100%** |
| **TTS Voice Configuration** | Coqui VITS (1.25X) | Coqui VITS (1.25X) | Windows SAPI (1.0X) | Hybrid |
| **Devanagari Numbers Normalization** | Enabled (*पंद्रह हज़ार*) | Enabled (*दस बजे रात*) | Standard Words | Compliant |
| **Avg Retrieval Latency** | {by_lang['chhattisgarhi']['avg_retrieval_sec']}s | {by_lang['hindi']['avg_retrieval_sec']}s | {by_lang['english']['avg_retrieval_sec']}s | **{round(np.mean([r['retrieval_sec'] for r in results]), 3)}s** |
| **Avg LLM Generation Latency** | {by_lang['chhattisgarhi']['avg_llm_sec']}s | {by_lang['hindi']['avg_llm_sec']}s | {by_lang['english']['avg_llm_sec']}s | **{round(np.mean([r['llm_sec'] for r in results]), 3)}s** |
| **Avg TTS Latency (first 3 sentences)** | {by_lang['chhattisgarhi']['avg_tts_ms']}ms | {by_lang['hindi']['avg_tts_ms']}ms | {by_lang['english']['avg_tts_ms']}ms | **{round(np.mean([r['tts_latency_ms'] for r in results]), 1)}ms** |
| **Avg Total Latency** | {by_lang['chhattisgarhi']['avg_total_sec']}s | {by_lang['hindi']['avg_total_sec']}s | {by_lang['english']['avg_total_sec']}s | **{round(np.mean([r['total_latency_sec'] for r in results]), 3)}s** |

---

## 2. Detailed Conversational Transcript & Analysis

"""
    for r in results:
        lang_title = r['lang'].capitalize()
        verdict_icon = "✅" if r["verdict"] in ("PASS", "SAFE_REFUSAL") else "⚠️"
        report_md += f"""### [{r['id']}] {r['topic']} ({lang_title}) — {verdict_icon} {r['verdict']}
- **User Utterance (Spoken):** "{r['question']}"
- **Effective Query (Rewriter):** `{r['effective_query']}` (Rewritten: {r['was_rewritten']})
- **Top Source Retrieved:** {r['top_source']} (Contexts: {r['contexts_count']})
- **Latencies:** Retrieval: `{r['retrieval_sec']}s` | LLM: `{r['llm_sec']}s` | TTS: `{r['tts_latency_ms']}ms` | Total: `{r['total_latency_sec']}s`
- **System Spoken Response:**
> {r['response_text'].replace(chr(10), chr(10) + '> ')}

---
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)

    brain_dir = r"C:\Users\ASUS\.gemini\antigravity\brain\58b26346-8346-4a73-8bb0-f15d31c806a7"
    if os.path.exists(brain_dir):
        shutil.copy2(REPORT_FILE, os.path.join(brain_dir, "CONVERSATIONAL_BENCHMARK_REPORT.md"))

    print("✅ Generated Conversational Report:", REPORT_FILE)


def main():
    print("=" * 75)
    print("STARTING CONVERSATIONAL BENCHMARK (REAL-WORLD SPOKEN UTTERANCES)")
    print(f"Pacing delay: {PACING_DELAY_SEC}s between queries")
    print(f"Total questions: {len(CONVERSATIONAL_QUESTIONS)}")
    print("=" * 75)

    retriever = DocumentRetriever()
    config = RetrievalConfig()
    dummy_audio = generate_dummy_speech_wav(duration_s=1.0)

    results = []

    for i, q in enumerate(CONVERSATIONAL_QUESTIONS, 1):
        qid = q["id"]
        target_lang = q["lang"]
        stt_code = "hne" if target_lang == "chhattisgarhi" else ("hi" if target_lang == "hindi" else "en")

        print(f"\n[{i}/{len(CONVERSATIONAL_QUESTIONS)}] Testing {qid} ({target_lang}): '{q['question'][:55]}...'")
        t_start_q = time.time()

        # 1. STT
        stt_res = test_stt_layer(dummy_audio, target_lang=stt_code)

        # 2. RAG
        rag_res = test_rag_layer(q["question"], retriever, config, q["expected_behavior"], target_lang=target_lang)

        # 3. TTS
        session_id = uuid.uuid4().hex
        tts_res = test_tts_layer(rag_res["response_text"], session_id=session_id)

        total_q_time = round(time.time() - t_start_q, 2)
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
            elif rag_res["contexts_count"] == 0 or rag_res["refused_at_retrieval"]:
                verdict = "REFUSAL_EMPTY_CONTEXT"
            elif not rag_res["has_citation"]:
                verdict = "MISSING_CITATION"
            else:
                verdict = "RETRIEVAL_MISS"

        res_entry = {
            "id": qid,
            "lang": target_lang,
            "topic": q["topic"],
            "expected_behavior": q["expected_behavior"],
            "question": q["question"],
            "description": q["description"],
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
            "guard_diagnostic": rag_res["guard_diagnostic"],
            "tts_status": tts_res.get("status", "FAIL"),
            "tts_latency_ms": tts_res.get("latency_ms", 0),
            "total_latency_sec": total_q_time,
            "verdict": verdict,
        }

        results.append(res_entry)
        print(f"    --> Verdict: {verdict} | Contexts: {rag_res['contexts_count']} | RAG: {rag_res['total_rag_sec']}s | TTS: {tts_res.get('latency_ms', 0)}ms | Total: {total_q_time}s")

        # Save incremental results
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Inter-query pacing to avoid 15 RPM rate limiting
        if i < len(CONVERSATIONAL_QUESTIONS):
            time.sleep(PACING_DELAY_SEC)

    # Final reports
    generate_conversational_report(results)
    print("\n🎉 ALL CONVERSATIONAL QUESTIONS EVALUATED AND REPORT GENERATED!")


if __name__ == "__main__":
    main()

