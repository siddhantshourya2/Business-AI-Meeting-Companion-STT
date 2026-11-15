import warnings
import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging

# Reduce unnecessary warning messages
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Initialize the Whisper speech recognition pipeline
pipe = pipeline(
  "automatic-speech-recognition",
  model = "openai/whisper-tiny.en",
  chunk_length_s=30
)

# Audio file to transcribe
audio_file = "download_audio.mp3"

# Conver speech into text
result = pipe(audio_file, batch_size=8)["text"]

# Display the transcription
print("\nTranscribed Text:")
print(result)