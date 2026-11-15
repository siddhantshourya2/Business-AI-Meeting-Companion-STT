import time
import gc
import torch
from transformers import pipeline

MODEL = "openai/whisper-tiny.en"
AUDIO = "Jeena_Jeena.mp3"

def benchmark(device, device_name):
    print(f"\n{'='*45}")
    print(f"Testing: {device_name}")
    print("="*45)

    start_load = time.perf_counter()

    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL,
        device=device,
        chunk_length_s=30
    )

    load_time = time.perf_counter() - start_load

    # Warm-up: first run overhead ko reduce karne ke liye
    pipe(AUDIO, return_timestamps=True)

    if device == 0:
        torch.cuda.synchronize()

    start = time.perf_counter()

    result = pipe(AUDIO)

    if device == 0:
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - start

    print(f"Model loading time: {load_time:.2f} seconds")
    print(f"Transcription time: {inference_time:.2f} seconds")
    print(f"Transcript: {result['text']}")

    del pipe
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return inference_time


# CPU benchmark
cpu_time = benchmark(-1, "CPU - Ryzen 5 5500H")

# GPU benchmark
gpu_time = benchmark(0, "GPU - RTX 2050")

print("\n========== FINAL COMPARISON ==========")
print(f"CPU Time: {cpu_time:.2f} seconds")
print(f"GPU Time: {gpu_time:.2f} seconds")
print(f"GPU Speedup: {cpu_time / gpu_time:.2f}x faster")
