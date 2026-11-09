# IIIT-NR Multilingual Voice-First Academic AI Assistant

**A full-duplex, multimodal Academic Voice Assistant for Dr. SPM IIIT Naya Raipur (IIIT-NR) featuring Hybrid Retrieval (FAISS + BM25 Okapi + RRF), Google Gemini 2.5/3.5, GPU-Accelerated STT (Faster-Whisper Large-v3-Turbo), and Real-Time Hybrid Voice Synthesis (Coqui VITS GPU + Windows SAPI) across Chhattisgarhi, Hindi, and English.**

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square\&logo=python\&logoColor=white)

![Streamlit](https://img.shields.io/badge/Streamlit-1.49-FF4B4B?style=flat-square\&logo=streamlit\&logoColor=white)

![Google Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20%2F%203.5%20Flash-4285F4?style=flat-square\&logo=google)

![Hybrid Retrieval](https://img.shields.io/badge/Retrieval-Hybrid%20BM25%20%2B%20Dense%20\(RRF\)-8E44AD?style=flat-square)

![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20IndexFlatIP-26A69A?style=flat-square)

![Faster Whisper](https://img.shields.io/badge/STT-Faster--Whisper%20Large--v3--Turbo%20\(CUDA\)-blue?style=flat-square)

![TTS Engine](https://img.shields.io/badge/TTS-Coqui%20VITS%20GPU%20%2B%20Windows%20SAPI-E67E22?style=flat-square)

![CUDA Acceleration](https://img.shields.io/badge/GPU-NVIDIA%20RTX%20GPU%20\(CUDA%2012.8\)-76B900?style=flat-square\&logo=nvidia)

![Tests Passing](https://img.shields.io/badge/tests-489%20passing-2ea44f?style=flat-square)

![Benchmark Success](https://img.shields.io/badge/Benchmark-95.5%25%20Pass%20Rate-brightgreen?style=flat-square)

---

## 📌 Overview

The **IIIT-NR Academic Voice Assistant** provides accurate, grounded, and verified information to students, parents, faculty, and campus visitors based on the official documentation of **Dr. SPM IIIT Naya Raipur** (academic ordinances, B.Tech & M.Tech curricula, fee structures, admissions policies, and campus rules).

To enable natural phone call interactions and regional language accessibility, the system operates as a **full-duplex conversational voice agent**:

- Users speak naturally in **Chhattisgarhi (छत्तीसगढ़ी)**, **Hindi (हिन्दी)**, or **English**.
- **GPU Faster-Whisper Turbo STT** transcribes spoken voice in real time (~50–120ms).
- A **Cross-Lingual Query Rewriter** translates and disambiguates multi-turn follow-ups (_"What is its fees?"_, _"ओमा कतेक सीट हे?"_).
- **Hybrid Retrieval (FAISS + BM25 Okapi + RRF)** retrieves grounded citations from 2,523 indexed academic chunks.
- A **Voice Call Persona** formulates human-sounding answers without robotic document references or repeated greetings.
- A **Hybrid Low-Latency TTS Engine** narrates answers back:
  - **Chhattisgarhi & Hindi**: GPU Coqui VITS (22,050 Hz) at **1.25X pace** with full Devanagari number-to-words normalization (_पंद्रह हज़ार रुपिया / रुपये_).
  - **English**: Native Windows SAPI at **1.0X normal conversational pace** (~40–110ms latency).

---

## ✨ Key Capabilities

### 1. Hybrid Retrieval & Academic Grounding

- 🔀 **Hybrid Retrieval (RRF):** Fuses dense vector similarity from Google Gemini embeddings with exact lexical token matching from BM25 Okapi via Reciprocal Rank Fusion (`dense_threshold = 0.65`).
- 💬 **Cross-Lingual Query Rewriter:** Translates incoming Indic dialects into search queries against English official documents while resolving pronouns and coreferences across conversation history.
- 📑 **Citation Verifier & Answer Guard:** Every answer is backed by exact source pages (`📄 Source: Page N — Section Title`). Built-in circuit breakers block unverified hallucinations and safely deflect out-of-scope inquiries.

### 2. Full-Duplex Two-Way Voice Call Persona

- 📞 **Continuous Hands-Free Voice Loop:** Starts with an audible spoken greeting, automatically opens the mic, auto-stops when the user finishes speaking (client-side VAD), vocalizes the response, and immediately loops for the next question.
- 🗣️ **Natural Phone Advisor Tone:**
  - **No Repetitive Greetings:** Greet only at the start of a call; ongoing follow-up questions jump directly into the answer using natural connectives (_"जी बिल्कुल,"_, _"हाँ,"_, _"Sure,"_).
  - **Zero Robotic Document References:** Never mentions _"उपलब्ध शैक्षणिक दस्तावेजों के अनुसार"_ or _"As per the documents"_. Speaks authoritatively as an institute representative (_"संस्थान के नियमानुसार..."_ or direct statements).
  - **Devanagari Spoken Numbers:** All currency, percentages, years, and semesters are vocalized as spoken words (_नब्बे हज़ार रुपये_, _दस बजे रात_, _पचहत्तर प्रतिशत_).

---

## 📊 Benchmark Scorecards & Empirical Results

The system has undergone rigorous automated testing across three independent evaluation benchmarks:

### 1. Comprehensive 200-Question Trilingual Benchmark (`benchmark_200_test.py`)

| Metric                                  | Chhattisgarhi (छत्तीसगढ़ी) | Hindi (हिन्दी)  | English        | Overall System      |
| --------------------------------------- | -------------------------- | --------------- | -------------- | ------------------- |
| **Total Test Inquiries**                | 67                         | 67              | 66             | **200**             |
| **Pipeline Success Rate**               | **97.0%**                  | **98.5%**       | **90.9%**      | **95.5%** (191/200) |
| **Direct Accurate Answers**             | 57                         | 58              | 53             | **168**             |
| **Intended Safe Refusals (OOS)**        | 8                          | 8               | 7              | **23**              |
| **Missing Citations**                   | 0                          | 0               | 0              | **0**               |
| **STT Transcription Rate**              | 100%                       | 100%            | 100%           | **100%**            |
| **TTS Synthesis Reliability**           | 100%                       | 100%            | 100%           | **100%**            |
| **Avg TTS Latency (first 3 sentences)** | 670.4ms (1.25X)            | 628.2ms (1.25X) | 110.3ms (1.0X) | **471.4ms**         |

### 2. Conversational Spoken Utterances (`conversational_benchmark_test.py`)

- Evaluated **27 colloquial spoken inquiries** with dialect greetings (_"जय जोहार संगी"_, _"नमस्ते भैया"_, _"Hey bro"_), slang, and informal phrasing:
- **Overall Success Rate**: **92.6%** (25/27 passed / safely deflected).
- Clean average RAG retrieval latency: **~3.07 seconds**.

### 3. Multi-Turn Context & Coreference Memory (`multiturn_context_test.py`)

- Evaluated **6 complete dialogues (18 turns total)** testing pronoun coreference (_"What is its fees?"_, _"उसमें कितना वजीफा मिलता है?"_, _"ओमा कतेक सीट आरक्षित हे?"_):
- **Multi-Turn Resolution Rate**: **94.4%** (17/18 turns accurately disambiguated and answered).

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │     IIIT-NR Official Academic Documents      │
                    │   (Regulations, Curricula, Fees, Helplines)  │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │      Ingestion Pipeline (ingestion/)         │
                    │   • Header-aware chunking (1,400 chars)      │
                    │   • Gemini Embeddings (768 dimensions)       │
                    │   • FAISS IndexFlatIP + In-Memory BM25       │
                    │   • 2,523 total indexed chunks               │
                    └──────────────────────┬───────────────────────┘
                                           │
 ──────────────────────────────────────────┼──────────────────────────────────────────
                                           ▼ (Voice & Chat Query Flow)
                               ┌───────────────────────┐
                               │ User Voice or Text    │
                               │ (CG, Hindi, English)  │
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │ Faster-Whisper Turbo  │  STT Port 8001
                               │ (CUDA float16, 50ms)  │  Auto / Hindi / English / CG
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │ Query Rewriter (LLM)  │  Cross-lingual translation +
                               │ + Context Resolution  │  Multi-turn pronoun resolution
                               └───────────┬───────────┘
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
             ┌───────────────────────┐           ┌───────────────────────┐
             │ Dense Search (FAISS)  │           │ Lexical Search (BM25) │
             │ Gemini Embeddings     │           │ Okapi Token Matching  │
             └───────────┬───────────┘           └───────────┬───────────┘
                         └─────────────────┬─────────────────┘
                                           ▼
                               ┌───────────────────────┐
                               │ Reciprocal Rank Fusion│
                               │ (RRF dense_thresh=0.65│
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │ Gemini 2.5 / 3.5 Flash│  Voice Call Persona:
                               │ (Streaming Tokens)    │  • Warm Turn 1 greeting only
                               └───────────┬───────────┘  • Zero document references
                                           │              • Devanagari numerals
                    ┌──────────────────────┴──────────────────────┐
                    │ (Text Tokens)                               │ (Sentences on Boundary)
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │ Streamlit Chat UI     │                     │ Hybrid TTS Server     │
        │ Port 8501             │                     │ Port 8000             │
        │ Live Typing Markdown  │                     │ • EN: Windows SAPI 1.0│
        └───────────────────────┘                     │ • CG/HI: GPU VITS 1.25│
                                                      └───────────┬───────────┘
                                                                  │ (22,050 Hz PCM Audio)
                                                                  ▼
                                                      ┌───────────────────────┐
                                                      │ WebSocket Connection  │
                                                      │ Browser Web Audio API │
                                                      │ (Seamless Playback)   │
                                                      └───────────────────────┘
```

---

## 📂 Project Directory Structure

```
uajy-academic-rag-chatbot/
├── app/
│   ├── main_streamlit.py          # Main web application, voice recording, hands-free call mode
│   ├── prompt_builder.py          # Voice call prompts, language detection, number normalization
│   ├── query_rewriter.py          # Cross-lingual rewriter & multi-turn coreference resolver
│   ├── retrieval.py               # Hybrid retriever (FAISS + BM25 Okapi + RRF)
│   ├── reranker.py                # Relevance ranking component
│   ├── answer_guard.py            # Diagnostic circuit breaker with safe refusal recognizer
│   ├── citations.py               # Citation extractor and source line validator
│   ├── config.py                  # Thresholds, models, and retrieval configurations
│   └── llm_client.py              # Google GenAI SDK client with dynamic retry backoff
├── data/                          # 33 verified institutional source documents
│   ├── academics/                 # 2022-2024 & 2024-2028 curricula for CSE, DSAI, ECE
│   ├── admissions/                # Spot round, CSAB, JoSAA, Chhattisgarh 50% quota rules
│   ├── faculty/                   # Faculty profiles and departmental disciplines
│   └── general/                   # Academic ordinances, campus facilities, health center, CMIT
├── index/                         # Built index files (2,523 chunks)
│   ├── faiss.index                # FAISS vector index
│   ├── metadata.json              # Chunk text, page numbers, and heading paths
│   └── embedding_cache.pkl        # Progressive disk cache for embeddings
├── ingestion/
│   ├── build_index.py             # Incremental index builder
│   ├── chunking.py                # Header-aware Markdown chunker
│   └── extract_pdf.py             # PDF extraction utility
├── STT/stt-service/               # GPU Speech-to-Text Microservice (Port 8001)
│   ├── run.py                     # Uvicorn entry point
│   └── src/                       # Faster-Whisper Large-v3-Turbo & Meta MMS engines
├── TTS/                           # Local Hybrid Text-to-Speech Microservice (Port 8000)
│   ├── tts_server.py              # FastAPI microservice for GPU VITS & Windows SAPI
│   ├── streaming_tts.py           # Core streaming synthesizer
│   └── chattisgarhi-tts-models/   # Pre-trained VITS checkpoints (Female & Male voices)
├── tests/                         # Comprehensive unit & regression test suite (489 tests)
├── benchmark_200_test.py          # 200-question trilingual evaluation pipeline
├── conversational_benchmark_test.py # 27-question colloquial dialect evaluation suite
├── multiturn_context_test.py      # 18-turn multi-turn conversational context benchmark
├── BENCHMARK_200Q_TRILINGUAL_REPORT.md # Detailed 200-question evaluation report
├── CONVERSATIONAL_BENCHMARK_REPORT.md  # Detailed conversational benchmark report
├── MULTITURN_CONTEXT_BENCHMARK_REPORT.md # Detailed multi-turn context report
├── VOICE_AGENT_HANDOVER.md        # Technical runbook & developer handover
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Conda Environments

- **OS:** Windows 10/11 or Linux
- **GPU:** NVIDIA GPU with CUDA support (e.g. RTX 3050 / 4050 or higher recommended)
- **Environments & Requirements:**
  - **`uajy-rag`:** `C:\Users\ASUS\anaconda3\envs\uajy-rag\python.exe` (Python 3.11+, Streamlit, FAISS, Google GenAI)
    ```powershell
    pip install -r requirements.txt
    pip install -r requirements-dev.txt   # Optional: for unit testing
    ```
  - **`minor1`:** `C:\Users\ASUS\anaconda3\envs\minor1\python.exe` (Python 3.10 with CUDA 12.1+, PyTorch, Faster-Whisper, Coqui TTS)
    ```powershell
    # Server 2 (TTS Microservice):
    pip install -r TTS/requirements.txt

    # Server 3 (STT Microservice):
    pip install -r STT/stt-service/requirements.txt
    ```

### 2. Configuration (`.streamlit/secrets.toml`)

Create `.streamlit/secrets.toml` with your Google AI Studio API key:

```toml
GEMINI_API_KEY = "your_google_ai_studio_api_key"
LLM_MODEL = "gemini-3.5-flash-lite"
UTILITY_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-2"
APP_MODE = "public"
```

### 3. Launching the 3 Microservices

Open three separate PowerShell terminal windows:

#### Terminal 1: Start the Hybrid TTS Server (Port 8000)

```powershell
Set-Location "d:\iiit nr\sem 5\New folder (2)\uajy-academic-rag-chatbot\TTS"
& "C:\Users\ASUS\anaconda3\envs\minor1\python.exe" -m uvicorn tts_server:app --host 0.0.0.0 --port 8000
```

#### Terminal 2: Start the GPU STT Server (Port 8001)

```powershell
Set-Location "d:\iiit nr\sem 5\New folder (2)\uajy-academic-rag-chatbot\STT\stt-service"
& "C:\Users\ASUS\anaconda3\envs\minor1\python.exe" "run.py"
```

#### Terminal 3: Start the Streamlit Application (Port 8501)

```powershell
Set-Location "d:\iiit nr\sem 5\New folder (2)\uajy-academic-rag-chatbot"
& "C:\Users\ASUS\anaconda3\envs\uajy-rag\python.exe" -m streamlit run app/main_streamlit.py --server.port 8501
```

Open **`http://localhost:8501`** in your browser.

---

## 🧪 Testing & Verification

```powershell
# 1. Run all 489 unit tests (Retrieval, Answer Guard, Citations, Prompt Builder, Query Rewriter)
& "C:\Users\ASUS\anaconda3\envs\uajy-rag\python.exe" -m pytest tests/ -q

# 2. Verify TTS Server Health (Port 8000)
& "C:\Users\ASUS\anaconda3\envs\minor1\python.exe" -c "import requests; print('TTS Status:', requests.get('http://127.0.0.1:8000/docs').status_code)"

# 3. Verify STT Server Health (Port 8001)
& "C:\Users\ASUS\anaconda3\envs\minor1\python.exe" -c "import requests; print('STT Status:', requests.get('http://127.0.0.1:8001/health').json())"

# 4. Run the 18-turn Multi-Turn Context Benchmark
& "C:\Users\ASUS\anaconda3\envs\uajy-rag\python.exe" -u multiturn_context_test.py
```

---

## 📄 License

This project is developed for institutional academic assistance at **Dr. SPM IIIT Naya Raipur**. All institutional policies and documentation are copyright of IIIT Naya Raipur. Codebase is released under the MIT License.
