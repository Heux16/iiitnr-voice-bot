import sys
import os
import time

# Ensure we don't import the local 'TTS' directory by accident
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path = [p for p in sys.path if os.path.abspath(p) not in (current_dir, parent_dir)]

import torch
from TTS.utils.synthesizer import Synthesizer

model_path = os.path.join(current_dir, "chattisgarhi-tts-models", "Female", "best_model.pth")
config_path = os.path.join(current_dir, "chattisgarhi-tts-models", "Female", "config.json")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

start_load = time.time()
tts = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    use_cuda=True
)
print(f"Model loaded in {time.time() - start_load:.2f}s")

# Warmup
_ = tts.tts(text="नमस्ते")

test_text = "जम्मू कश्मीर म पर्यटन उद्योग ला बढ़ावा देना उहाँ के अर्थबेवस्था ला सुचारू रूप ले चलाय बर जरुरी हे"
start_infer = time.time()
wav = tts.tts(text=test_text)
infer_time = time.time() - start_infer

audio_duration = len(wav) / 22050.0
rtf = infer_time / audio_duration
print(f"Inference time: {infer_time:.3f}s for {audio_duration:.2f}s of audio (RTF: {rtf:.4f})")
