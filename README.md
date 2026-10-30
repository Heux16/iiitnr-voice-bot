# IIIT-NR Academic RAG Chatbot with Real-Time GPU-Accelerated TTS

**An intelligent, grounded Academic Assistant for IIIT Naya Raipur (Dr. SPM IIIT-NR) featuring Hybrid Retrieval (FAISS + BM25), Google Gemini 3.5, and Real-Time Sentence-by-Sentence Chhattisgarhi/Hindi Voice Narration on NVIDIA RTX 4050 GPU.**

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/LLM-Gemini%203.5%20Flash%20Lite-4285F4?style=flat-square&logo=google)
![Gemini Embedding 2](https://img.shields.io/badge/Embedding-Gemini%20Embedding%202%20(3072d)-blue?style=flat-square)
![Hybrid Retrieval](https://img.shields.io/badge/Retrieval-Hybrid%20BM25%20%2B%20Dense%20(RRF)-8E44AD?style=flat-square)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20IndexFlatIP-26A69A?style=flat-square)
![TTS Model](https://img.shields.io/badge/TTS-Coqui%20VITS%20(Chhattisgarhi)-E67E22?style=flat-square)
![CUDA Acceleration](https://img.shields.io/badge/GPU-NVIDIA%20RTX%204050%20(CUDA%2012.1)-76B900?style=flat-square&logo=nvidia)
![Tests Passing](https://img.shields.io/badge/tests-459%20passing-2ea44f?style=flat-square)

---

## 📌 Overview

The **IIIT-NR Academic Document RAG Chatbot** provides accurate, strictly verified answers to students, faculty, and campus visitors based on official IIIT Naya Raipur institutional documentation (academic handbooks, curriculum catalogs, fee structures, faculty profiles, and campus regulations).

To bridge accessibility and regional languages, the system includes an **end-to-end real-time speech narration pipeline**: answers generated in Hindi are synthesized sentence-by-sentence via a local **Coqui VITS deep learning acoustic model** accelerated by **CUDA on an NVIDIA GeForce RTX 4050 GPU**, streaming audio directly into the user's browser in real time.

---

## ✨ Key Capabilities

### 1. Hybrid Retrieval & Academic Grounding
* 🔀 **Hybrid Retrieval (RRF):** Merges semantic embeddings from **Google Gemini Embedding 2** (3,072 dimensions) with exact lexical token matching via **BM25 Okapi** using Reciprocal Rank Fusion.
* 🎚️ **Cross-Encoder LLM Reranking:** Filters and scores candidate chunks to eliminate false positives before prompt construction.
* 💬 **Multilingual Query Rewriter:** Translates incoming Hindi/Devanagari queries into English search queries for optimal retrieval against English source documents, while resolving conversational references (*"how much is it?"*, *"who is the HOD?"*).
* 📑 **Strict Citations & Guardrails:** Answers must cite exact page numbers (`📄 Source: Page N — Section Title`). Built-in circuit breakers block unverified or hallucinated responses.

### 2. Real-Time GPU Voice Narration (`TTS/`)
* ⚡ **7x – 27x Faster Than Real Time:** Powered by local NVIDIA GeForce RTX 4050 GPU (~0.15s – 0.3s per sentence).
* 🔊 **Sentence-by-Sentence In-Browser Audio Queue:** An embedded Web Audio API component in Streamlit connects via **WebSocket** (`ws://127.0.0.1:8000/ws_audio/{session_id}`). As Gemini streams tokens, completed sentences are dispatched and played back-to-back with **zero gap, zero click, and zero replay lag**.
* 🗣️ **Devanagari Speech Synthesis:** Uses the Chhattisgarhi VITS acoustic model trained on Devanagari script for both **Female** and **Male** voices.
* 🔢 **Automatic Number & Fee Normalizer:** Numbers, fee amounts, and currency are automatically converted into spoken Hindi words (e.g., `₹90,000` ➔ `नब्बे हज़ार रुपये`, `75%` ➔ `पचहत्तर प्रतिशत`), preventing digits from being skipped by the acoustic model.

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │     IIIT-NR Official Academic Documents      │
                    │   (Academics, Faculty, Admissions, Fees)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │       Ingestion Pipeline (build_index)       │
                    │   • Header-aware chunking (1400 chars)       │
                    │   • Gemini Embedding 2 (3072 dimensions)     │
                    │   • FAISS IndexFlatIP + In-Memory BM25       │
                    └──────────────────────┬───────────────────────┘
                                           │
 ──────────────────────────────────────────┼──────────────────────────────────────────
                                           ▼ (Query Flow)
                               ┌───────────────────────┐
                               │  User Query in Chat   │
                               │   (English or Hindi)  │
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │ Query Rewriter (LLM)  │  "अभिषेक शर्मा कौन हैं"
                               │  (Devanagari ➔ EN)   │   ➔ "Who is Abhishek Sharma?"
                               └───────────┬───────────┘
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
             ┌───────────────────────┐           ┌───────────────────────┐
             │ Dense Search (FAISS)  │           │ Lexical Search (BM25) │
             │ Gemini Embedding 2    │           │ Okapi Keyword Match   │
             └───────────┬───────────┘           └───────────┬───────────┘
                         └─────────────────┬─────────────────┘
                                           ▼
                               ┌───────────────────────┐
                               │ Reciprocal Rank Fusion│
                               │     + LLM Reranker    │
                               └───────────┬───────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │ Gemini 3.5 Flash Lite │  Answers directly in Hindi
                               │ (Token Streaming)     │  with numbers in words
                               └───────────┬───────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │ (Text Tokens)                               │ (Sentences on Boundary)
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │ Streamlit Chat UI     │                     │ FastAPI TTS Server    │
        │ Live Typing Markdown  │                     │ (RTX 4050 GPU)        │
        └───────────────────────┘                     └───────────┬───────────┘
                                                                  │ (WAV Audio Chunks)
                                                                  ▼
                                                      ┌───────────────────────┐
                                                      │ WebSocket Connection  │
                                                      │ Browser Web Audio API │
                                                      │ (Seamless Playback)   │
                                                      └───────────────────────┘
```

---

## 📂 Project Directory Structure

```text
uajy-academic-rag-chatbot/
├── app/
│   ├── main_streamlit.py          # Main Streamlit web application & Web Audio widget
│   ├── prompt_builder.py          # Grounded Hindi academic prompt templates & rules
│   ├── query_rewriter.py          # Standalone query rewrite & Devanagari normalizer
│   ├── retrieval.py               # HybridRetriever combining FAISS + BM25 Okapi
│   ├── reranker.py                # Cross-encoder LLM reranking gate
│   ├── answer_guard.py            # Circuit-breaker validating citations & safety
│   ├── citations.py               # Source citation parser and reference formatter
│   ├── config.py                  # Pipeline configuration, thresholds, and models
│   ├── llm_client.py              # Google GenAI SDK client for Gemini 3.5 & embeddings
│   ├── pdf_viewer.py              # In-app PDF document viewer component
│   └── theme.py                   # Academic portal UI theme and CSS styling
├── data/                          # 92 indexed Markdown documents across campus disciplines
│   ├── academics/                 # Regulations, ordinances, and grading systems
│   ├── admissions/                # Admission criteria, eligibility, and fee structure
│   ├── faculty/                   # Faculty profiles and research disciplines
│   └── general/                   # Campus administration, facilities, and hostels
├── index/                         # Built index files
│   ├── faiss.index                # FAISS vector index (2,484 vectors, dim 3072)
│   ├── metadata.json              # Document chunk metadata & page mappings
│   ├── embedding_cache.pkl        # Progressive disk cache for embeddings
│   └── index_info.json            # Ingestion build metadata & document registry
├── ingestion/
│   ├── build_index.py             # Index building script with batching & checkpointing
│   ├── chunking.py                # Document chunker with heading path retention
│   └── extract_pdf.py             # PDF text extraction utility
├── TTS/                           # Local GPU-Accelerated Text-to-Speech Engine
│   ├── tts_server.py              # FastAPI microservice with WebSocket streaming
│   ├── streaming_tts.py           # Core multi-threaded streaming synthesizer
│   ├── llm_streaming_tts_demo.py  # Standalone LLM-to-TTS streaming benchmark demo
│   └── chattisgarhi-tts-models/   # Pre-trained VITS neural TTS models
│       ├── Female/                # Female voice weights (best_model.pth) & config.json
│       └── Male/                  # Male voice weights (best_model.pth) & config.json
├── tests/                         # Comprehensive unit & regression test suite (459 tests)
├── test_e2e_rag_tts.py            # End-to-end integration verification script
├── requirements.txt               # Main application dependencies
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environments

* **Operating System:** Windows 10/11 or Linux
* **GPU:** NVIDIA GPU with CUDA support (e.g., RTX 3050/4050 or higher recommended)
* **Python Environments:**
  * **RAG Environment (`uajy-rag`):** Streamlit, Google GenAI SDK, FAISS, BM25
  * **TTS GPU Environment (`minor1`):** PyTorch with CUDA 12.1, Coqui TTS, FastAPI, Uvicorn

### 2. Configuration (`.streamlit/secrets.toml`)

Create `.streamlit/secrets.toml` with your Google AI Studio API key:

```toml
GEMINI_API_KEY = "your_google_ai_studio_key"
LLM_MODEL = "gemini-3.5-flash-lite"
UTILITY_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-2"
APP_MODE = "public"
```

### 3. Launching the Services

Open two terminals:

#### Terminal 1: Start the Local GPU TTS Server (Port 8000)
```powershell
# Using the GPU environment with PyTorch + CUDA
C:\Users\ASUS\anaconda3\envs\minor1\python.exe TTS/tts_server.py
```

#### Terminal 2: Start the Streamlit Web Application (Port 8501)
```powershell
# Using the RAG environment
C:\Users\ASUS\anaconda3\envs\uajy-rag\python.exe -m streamlit run app/main_streamlit.py
```

Open **`http://localhost:8501`** in your browser.

---

## 🧪 Testing & Verification

Run the full offline test suite (459 unit tests covering retrieval, citations, guards, and reranking):

```powershell
pytest -q
```

Run the end-to-end live integration test verifying query retrieval, Hindi answer synthesis, and GPU audio output:

```powershell
python test_e2e_rag_tts.py
```

---

## 📄 License

This project is licensed under the MIT License.
