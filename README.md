# Business AI Meeting Companion (STT)

A local Python application that transcribes meeting audio and generates a concise summary.

## Features
- Upload an audio file or record audio using the microphone.
- English speech-to-text using `openai/whisper-tiny.en`.
- Long-form audio processing with Whisper chunking and timestamps.
- Meeting summarization using `sshleifer/distilbart-cnn-12-6`.
- NVIDIA CUDA acceleration for Whisper when available, with CPU fallback.
- Local Gradio interface; no OpenAI API key or IBM Watson credentials required.

## Technology
Python, PyTorch, Hugging Face Transformers, Whisper Tiny English, DistilBART, and Gradio.

## Development hardware
- CPU: AMD Ryzen 5 5500H
- GPU: NVIDIA GeForce RTX 2050 (4 GB VRAM)
- RAM: 16 GB

## Windows setup

```powershell
cd E:\Mini Projects\Business-AI-Meeting-Companion-STT
python -m venv my_env
.\my_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build compatible with your NVIDIA driver using the official selector:
https://pytorch.org/get-started/locally/

For CPU-only use, install the CPU build instead.

Verify GPU availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

Run:

```powershell
python app.py
```

Open the local Gradio URL printed in the terminal, usually `http://127.0.0.1:7860`.

## Model execution
- Whisper runs on CUDA GPU (`device=0`) when available; otherwise it runs on CPU.
- DistilBART runs on CPU to conserve VRAM on a 4 GB GPU.
- First launch downloads pretrained model files from Hugging Face and requires internet access.

## CPU vs GPU benchmark

One short-audio test on the development laptop produced:

| Device | Transcription time |
|---|---:|
| Ryzen 5 5500H CPU | 4.59 seconds |
| RTX 2050 GPU | 1.06 seconds |

Measured speedup: **4.31×** for that specific recording and run. Results vary by audio, model, drivers, software versions, and background load; this is not a universal benchmark.

## Limitations
- The Whisper model is English-focused.
- The generated summary may omit details or contain inaccuracies; review it against the transcript.
- Very long transcripts may be truncated to fit the summarizer's input limit.
- Speaker identification and action-item extraction are not included.
- First-time model downloads require internet access.

## Models
- `openai/whisper-tiny.en`
- `sshleifer/distilbart-cnn-12-6`

Check the model cards and licenses on Hugging Face before redistribution or commercial use.
