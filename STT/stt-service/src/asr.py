"""
ASR engine abstraction with Multilingual Whisper and Chhattisgarhi MMS support.

- FasterWhisperEngine: multilingual Whisper (English, Hindi, Hinglish, auto-detection)
- ChhattisgarhiMMSEngine: Meta MMS (facebook/mms-1b-all) with 'hne' adapter
"""
from __future__ import annotations

import logging
import os
import sys
import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

LOGGER = logging.getLogger("stt-asr")

# Ensure CUDA runtime libraries from torch are discoverable by CTranslate2 on Windows
if sys.platform == "win32":
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except Exception:
            pass
        if torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel
from src.config import settings

# Phrases Whisper hallucinates when given silence or quiet background noise
SILENCE_HALLUCINATIONS = {
    "thank you", "thank you.", "thank you for watching", "thank you for watching.",
    "thanks for watching", "thanks for watching.", "subtitles by", "bye", "bye.", "you",
    "thank you very much.", "thank you very much", "watching", "watching."
}


@dataclass
class Transcript:
    text: str
    avg_logprob: float  # confidence proxy in [-1, 0]
    no_speech_prob: float  # probability segment was not speech (0-1)
    inference_ms: float
    detected_language: str = ""


class ASREngine(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, *, fast: bool, language: str | None = None) -> Transcript:
        """
        audio: float32 mono PCM at settings.SAMPLE_RATE, values in [-1, 1]
        fast:  True for partial transcripts (greedy), False for final transcripts (beam search)
        language: Optional language code ('auto', 'hne', 'hi', 'en')
        """
        raise NotImplementedError


def normalize_spoken_acronyms(text: str) -> str:
    """
    Normalise common phonetic misrecognitions of institute acronyms
    when spoken in Indian English, Hindi, or Chhattisgarhi.
    """
    if not text:
        return text

    # Phonetic variations of "Triple IT / Triple I T" in Devanagari or English
    text = re.sub(
        r"(?:प्रिबल|ट्रिपल|ट्रिपिल|त्रिपल|Triple)\s*(?:I\s*I\s*T|I\s*T|आई\s*आई\s*टी|आई\s*टी|eye\s*tea|I\.T\.)",
        "IIIT",
        text,
        flags=re.IGNORECASE,
    )
    # Devanagari expansion
    text = re.sub(r"आई\s*आई\s*आई\s*टी", "IIIT", text)
    # Common Devanagari phonetic misspelling for location query
    text = re.sub(r"\bइस्थित\b", "स्थित", text)
    return re.sub(r"\s+", " ", text).strip()


class FasterWhisperEngine(ASREngine):
    """
    Multilingual Whisper engine (CTranslate2).
    Capable of automatic language detection, fluent Hindi, English, and Hinglish.
    """
    def __init__(self):
        device = settings.ASR_DEVICE
        compute_type = settings.ASR_COMPUTE_TYPE
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            compute_type = "int8"

        self._model = WhisperModel(
            settings.ASR_MODEL_SIZE,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio: np.ndarray, *, fast: bool, language: str | None = None) -> Transcript:
        t0 = time.monotonic()
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        target_lang = None
        req_lang = (language or settings.ASR_LANGUAGE).lower()
        if req_lang not in ("auto", "none", "hne", ""):
            target_lang = req_lang

        # Domain guidance: language-specific prompts avoid cross-language confusion
        prompt = None
        hotwords = (
            "IIIT IIIT-NR Triple IT Triple I T ट्रिपल आईटी ट्रिपल आई टी आईआईआईटी "
            "Naya Raipur नया रायपुर B.Tech M.Tech CSE DSAI ECE admissions fees hostel "
            "cutoff स्थित सीजीपीए बीटेक एमटेक फीस"
        )
        if target_lang == "hi":
            prompt = "आईआईआईटी नया रायपुर, ट्रिपल आईटी, बीटेक, एमटेक, फीस, प्रवेश, हॉस्टल, छात्रवृत्ति, स्थित, सीजीपीए."
        elif target_lang == "en":
            prompt = "IIIT Naya Raipur, Dr. SPM IIIT-NR, Triple IT, B.Tech, M.Tech, Ph.D., admissions, fees, hostel, courses, CGPA."
        else:
            prompt = "IIIT Naya Raipur, ट्रिपल आई टी नया रायपुर, B.Tech, M.Tech, CSE, DSAI, ECE, फीस, स्थित, admissions, hostel."

        beam = settings.ASR_BEAM_SIZE if fast else min(settings.ASR_FINAL_BEAM_SIZE, 2)

        # 1. Run with lenient VAD filter (threshold=0.18, min_silence=1200ms) so natural conversational pauses don't cut words
        try:
            segments, info = self._model.transcribe(
                audio,
                language=target_lang,
                beam_size=beam,
                temperature=[0.0, 0.2],  # Cap fallback to 0.2: eliminates 25s multi-temp hangs & hallucinations
                vad_filter=True,
                vad_parameters=dict(threshold=0.18, min_silence_duration_ms=1200, speech_pad_ms=400),
                condition_on_previous_text=False,
                word_timestamps=False,
                initial_prompt=prompt,
                hotwords=hotwords,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            )
            segments = list(segments)
        except Exception as e:
            LOGGER.warning("Whisper transcription exception: %s", e)
            segments, info = [], None

        # 2. Fallback: if VAD returned nothing but the audio has energy, decode with beam 1 at temp 0.0
        if not segments and np.max(np.abs(audio)) > 0.02:
            LOGGER.info("VAD returned 0 segments on non-silent audio; retrying without VAD filter")
            try:
                segments, info = self._model.transcribe(
                    audio,
                    language=target_lang,
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    word_timestamps=False,
                    initial_prompt=prompt,
                    hotwords=hotwords,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                )
                segments = list(segments)
            except Exception as e:
                LOGGER.warning("Whisper fallback exception: %s", e)
                segments, info = [], None

        text = " ".join(s.text.strip() for s in segments).strip()
        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments) if segments else -1.0
        no_speech_prob = max((s.no_speech_prob for s in segments), default=1.0)
        detected_lang = getattr(info, "language", target_lang or "unknown") if info else (target_lang or "unknown")

        # Map related Indic language misdetections (common with Indian English/Hindi accent) to Hindi
        if detected_lang in ("mr", "ne", "ur", "sa", "pa", "gu", "bn", "ta", "te", "kn", "ml") or re.search(r"[\u0900-\u097F]", text):
            detected_lang = "hi"
        elif target_lang is None and detected_lang not in ("en", "hi", "hne"):
            detected_lang = "en"

        # Discard only known silence hallucination phrases (never discard real speech words)
        clean = text.strip().lower().rstrip(".!,?")
        if clean in SILENCE_HALLUCINATIONS:
            LOGGER.info("Discarded known silence hallucination phrase: %r", text)
            text = ""

        # Normalise spoken acronyms (e.g. "Triple IT" / "प्रिबल IIT" -> "IIIT")
        text = normalize_spoken_acronyms(text)

        return Transcript(
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            inference_ms=(time.monotonic() - t0) * 1000,
            detected_language=detected_lang,
        )


def clean_mms_devanagari(text: str) -> str:
    """
    MMS's CTC output can insert stray spaces before Devanagari matras/vowel signs on
    lower-resource adapters, which detaches them from their base consonant and renders as a
    dotted circle (◌). This rejoins them.
    """
    if not text:
        return ""
    text = re.sub(r'\s+([\u093e-\u094c\u0901-\u0903\u094d])', r'\1', text)
    text = normalize_spoken_acronyms(text)
    return re.sub(r'\s+', ' ', text).strip()


class ChhattisgarhiMMSEngine(ASREngine):
    """
    Meta MMS (facebook/mms-1b-all) for Chhattisgarhi ('hne').
    Uses Wav2Vec2 CTC with language adapter 'hne'.
    """
    def __init__(self):
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        model_id = "facebook/mms-1b-all"
        self._device = "cuda" if (settings.ASR_DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
        if settings.MMS_DTYPE not in {"float32", "float16", "bfloat16"}:
            raise ValueError("STT_MMS_DTYPE must be float32, float16, or bfloat16")
        self._dtype = getattr(torch, settings.MMS_DTYPE) if self._device == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(model_id, torch_dtype=self._dtype)
        self._processor.tokenizer.set_target_lang("hne")
        self._model.load_adapter("hne")
        self._model.to(self._device).eval()

    def transcribe(self, audio: np.ndarray, *, fast: bool, language: str | None = None) -> Transcript:
        t0 = time.monotonic()
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        # Trim leading/trailing silence with generous padding so soft consonants aren't lost
        abs_audio = np.abs(audio)
        speech_indices = np.where(abs_audio > 0.01)[0]
        if len(speech_indices) > 0:
            pad = int(settings.SAMPLE_RATE * 0.40)  # 400ms padding
            start = max(0, speech_indices[0] - pad)
            end = min(len(audio), speech_indices[-1] + pad)
            audio = audio[start:end]

        if len(audio) < int(settings.SAMPLE_RATE * 0.2):
            return Transcript(text="", avg_logprob=0.0, no_speech_prob=1.0, inference_ms=0.0, detected_language="hne")

        inputs = self._processor(audio, sampling_rate=settings.SAMPLE_RATE, return_tensors="pt")
        inputs = {
            k: v.to(device=self._device, dtype=self._dtype) if v.is_floating_point()
            else v.to(self._device) for k, v in inputs.items()
        }
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)[0]
        text = clean_mms_devanagari(self._processor.decode(ids).strip())
        return Transcript(
            text=text,
            avg_logprob=0.0,
            no_speech_prob=0.0 if text else 1.0,
            inference_ms=(time.monotonic() - t0) * 1000,
            detected_language="hne",
        )


_WHISPER_SINGLETON: FasterWhisperEngine | None = None
_MMS_SINGLETON: ChhattisgarhiMMSEngine | None = None


def get_engine(language: str = "auto") -> ASREngine:
    """
    Returns the appropriate engine singleton for the requested language:
    - 'hne': ChhattisgarhiMMSEngine
    - 'auto', 'hi', 'en', etc.: FasterWhisperEngine
    """
    global _WHISPER_SINGLETON, _MMS_SINGLETON
    lang = (language or settings.ASR_LANGUAGE).lower()

    if lang == "hne":
        if _MMS_SINGLETON is None:
            _MMS_SINGLETON = ChhattisgarhiMMSEngine()
        return _MMS_SINGLETON
    else:
        if _WHISPER_SINGLETON is None:
            _WHISPER_SINGLETON = FasterWhisperEngine()
        return _WHISPER_SINGLETON


def logprob_to_confidence(avg_logprob: float) -> float:
    """Rough mapping of avg_logprob -> a 0..1 'confidence'."""
    return max(0.0, min(1.0, 1.0 + avg_logprob))
