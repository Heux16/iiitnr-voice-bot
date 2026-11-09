"""
Streaming Text-to-Speech (TTS) Engine for Chhattisgarhi VITS Model.
Optimized for real-time streaming with LLMs on CUDA GPU.
"""

import os
import sys
import re
import queue
import threading
import time
from typing import Generator, Iterable, Optional, Tuple
import numpy as np

# Prevent current folder 'TTS' from conflicting with the installed 'TTS' package
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

_temp_path = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) not in (_current_dir, _parent_dir)]
import torch
from TTS.utils.synthesizer import Synthesizer
sys.path = _temp_path

# Ensure Windows stdout supports UTF-8 Devanagari characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


# ──────────────────────────────────────────────
# Hindi Number to Words Normalizer
# ──────────────────────────────────────────────

_HINDI_NUMS = {
    0: 'शून्य', 1: 'एक', 2: 'दो', 3: 'तीन', 4: 'चार', 5: 'पाँच', 6: 'छह', 7: 'सात', 8: 'आठ', 9: 'नौ',
    10: 'दस', 11: 'ग्यारह', 12: 'बारह', 13: 'तेरह', 14: 'चौदह', 15: 'पंद्रह', 16: 'सोलह', 17: 'सत्रह', 18: 'अठारह', 19: 'उन्नीस',
    20: 'बीस', 21: 'इक्कीस', 22: 'बाईस', 23: 'तेईस', 24: 'चौबीस', 25: 'पच्चीस', 26: 'छब्बीस', 27: 'सत्ताईस', 28: 'अट्ठाईस', 29: 'उनतीस',
    30: 'तीस', 31: 'इकतीस', 32: 'बत्तीस', 33: 'तैंतीस', 34: 'चौंतीस', 35: 'पैंतीस', 36: 'छत्तीस', 37: 'सैंतीस', 38: 'अड़तीस', 39: 'उनतालीस',
    40: 'चालीस', 41: 'इकतालीस', 42: 'बयालीस', 43: 'तैंतालीस', 44: 'चौवालीस', 45: 'पैंतालीस', 46: 'छियालीस', 47: 'सैंतालीस', 48: 'अड़तालीस', 49: 'उनचास',
    50: 'पचास', 51: 'इक्यावन', 52: 'बावन', 53: 'तिरपन', 54: 'चौवन', 55: 'पचपन', 56: 'छप्पन', 57: 'सत्तावन', 58: 'अट्ठावन', 59: 'उनसठ',
    60: 'साठ', 61: 'इकसठ', 62: 'बासठ', 63: 'तिरसठ', 64: 'चौंसठ', 65: 'पैंसठ', 66: 'छियासठ', 67: 'सरसठ', 68: 'अड़सठ', 69: 'उनहत्तर',
    70: 'सत्तर', 71: 'इकहत्तर', 72: 'बहत्तर', 73: 'तिहत्तर', 74: 'चौहत्तर', 75: 'पचहत्तर', 76: 'छियाहत्तर', 77: 'सतहत्तर', 78: 'अठहत्तर', 79: 'उन्नासी',
    80: 'अस्सी', 81: 'इक्यासी', 82: 'बयासी', 83: 'तिरासी', 84: 'चौरासी', 85: 'पचासी', 86: 'छियासी', 87: 'सत्तासी', 88: 'अट्ठासी', 89: 'नवासी',
    90: 'नब्बे', 91: 'इक्यानवे', 92: 'बानवे', 93: 'तिरानवे', 94: 'चौरानवे', 95: 'पचानवे', 96: 'छियानवे', 97: 'सत्तानवे', 98: 'अट्ठानवे', 99: 'निन्यानवे'
}

def num_to_hindi(n: int) -> str:
    if n < 0:
        return 'माइनस ' + num_to_hindi(-n)
    if n in _HINDI_NUMS:
        return _HINDI_NUMS[n]
    if n < 1000:
        hundreds = n // 100
        rem = n % 100
        res = _HINDI_NUMS.get(hundreds, '') + ' सौ'
        if rem > 0:
            res += ' ' + num_to_hindi(rem)
        return res
    if n < 100000:
        thousands = n // 1000
        rem = n % 1000
        res = num_to_hindi(thousands) + ' हज़ार'
        if rem > 0:
            res += ' ' + num_to_hindi(rem)
        return res
    if n < 10000000:
        lakhs = n // 100000
        rem = n % 100000
        res = num_to_hindi(lakhs) + ' लाख'
        if rem > 0:
            res += ' ' + num_to_hindi(rem)
        return res
    crores = n // 10000000
    rem = n % 10000000
    res = num_to_hindi(crores) + ' करोड़'
    if rem > 0:
        res += ' ' + num_to_hindi(rem)
    return res

def convert_numbers_in_text(text: str) -> str:
    def curr_repl(m):
        raw = m.group(1).replace(',', '')
        try:
            return num_to_hindi(int(raw)) + ' रुपये'
        except ValueError:
            return m.group(0)
    text = re.sub(r'(?:₹|Rs\.?|INR)\s*([0-9,]+)', curr_repl, text, flags=re.IGNORECASE)

    def pct_repl(m):
        raw = m.group(1).replace(',', '')
        try:
            return num_to_hindi(int(raw)) + ' प्रतिशत'
        except ValueError:
            return m.group(0)
    text = re.sub(r'([0-9,]+)\s*%', pct_repl, text)

    def num_repl(m):
        raw = m.group(0).replace(',', '')
        try:
            return num_to_hindi(int(raw))
        except ValueError:
            return m.group(0)
    text = re.sub(r'\b[0-9,]+\b', num_repl, text)
    return text


class StreamingChhattisgarhiTTS:
    """
    Concurrent, low-latency streaming TTS engine for Chhattisgarhi VITS model.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        voice: str = "Female",  # 'Female' or 'Male'
        use_cuda: bool = True,
        length_scale: float = 0.80,  # 0.80 = 1.25x faster speed (1.0 = normal, >1.0 slower)
    ):
        base_dir = os.path.join(_current_dir, "chattisgarhi-tts-models", voice)
        self.checkpoint_path = checkpoint_path or os.path.join(base_dir, "best_model.pth")
        self.config_path = config_path or os.path.join(base_dir, "config.json")
        
        self.use_cuda = use_cuda and torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if self.use_cuda else "CPU"
        print(f"[StreamingTTS] Initializing VITS ({voice}) on {device_name} (CUDA={self.use_cuda})...")
        
        t0 = time.time()
        self.synthesizer = Synthesizer(
            tts_checkpoint=self.checkpoint_path,
            tts_config_path=self.config_path,
            use_cuda=self.use_cuda
        )
        
        # Override duration scale for speech rate control
        self.synthesizer.tts_model.length_scale = length_scale
        self.sample_rate = 22050
        print(f"[StreamingTTS] Model loaded in {time.time() - t0:.2f}s | Sample Rate: {self.sample_rate}Hz")
        
        # Warmup GPU
        try:
            _ = self.synthesizer.tts("नमस्ते")
            print("[StreamingTTS] GPU warmed up and ready.")
        except Exception:
            pass

        # Regex for splitting on Hindi sentence endings (Danda ।, question mark ?, exclamation !, newline \n)
        # Excludes period (.) to prevent fragmenting abbreviations like डॉ., पी.एच.डी., एम.टेक.
        self.sentence_regex = re.compile(r'([।?!\n])')

    def synthesize_stream(
        self,
        token_stream: Iterable[str],
        min_words_for_comma: int = 4
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Consumes an incoming LLM token stream.
        As soon as complete sentences or clauses are received,
        synthesizes audio concurrently on the GPU and yields (sentence_text, audio_array).
        """
        sentence_queue = queue.Queue()
        sentinel = object()
        output_queue = queue.Queue()

        def gpu_worker():
            while True:
                item = sentence_queue.get()
                if item is sentinel:
                    output_queue.put(sentinel)
                    sentence_queue.task_done()
                    break
                
                text_to_speak = convert_numbers_in_text(item.strip())
                if text_to_speak:
                    t_start = time.time()
                    try:
                        wav = self.synthesizer.tts(text=text_to_speak)
                        wav_arr = np.array(wav, dtype=np.float32)
                        infer_time = time.time() - t_start
                        output_queue.put((text_to_speak, wav_arr, infer_time))
                    except Exception as e:
                        print(f"[StreamingTTS Error] Failed on '{text_to_speak}': {e}")
                
                sentence_queue.task_done()

        worker_thread = threading.Thread(target=gpu_worker, daemon=True)
        worker_thread.start()

        # Producer thread logic: parse tokens into sentences
        buffer = ""
        first_chunk = True

        for token in token_stream:
            buffer += token
            # Split by delimiters
            parts = self.sentence_regex.split(buffer)
            if len(parts) > 1:
                # We have at least one delimited section
                candidate = "".join(parts[:-1]).strip()
                word_count = len(candidate.split())
                
                # For ultra-low latency, the very first chunk can be dispatched on a comma or 3-4 words
                # Subsequent chunks are dispatched on full sentence boundaries or after enough words
                if first_chunk or word_count >= min_words_for_comma or any(p in candidate for p in ['।', '?', '!', '.']):
                    if candidate:
                        sentence_queue.put(candidate)
                        first_chunk = False
                        buffer = parts[-1]  # Keep remainder
                
                # Check for audio chunks ready to yield
                while not output_queue.empty():
                    item = output_queue.get()
                    if item is not sentinel:
                        text, audio, dur = item
                        yield (text, audio)

        # Flush whatever remains in buffer
        if buffer.strip():
            sentence_queue.put(buffer.strip())

        sentence_queue.put(sentinel)

        # Yield all remaining synthesized audio chunks
        while True:
            item = output_queue.get()
            if item is sentinel:
                break
            text, audio, dur = item
            yield (text, audio)

        worker_thread.join()

    def play_live_stream(self, token_stream: Iterable[str]):
        """
        Plays audio live to speakers using sounddevice as the LLM generates tokens.
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("Please install sounddevice: pip install sounddevice")

        all_audio = []
        for text, audio in self.synthesize_stream(token_stream):
            print(f"🔊 Speaking: {text}")
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()  # Wait for current sentence playback to finish before starting next
            all_audio.append(audio)

        return np.concatenate(all_audio) if all_audio else np.array([])
