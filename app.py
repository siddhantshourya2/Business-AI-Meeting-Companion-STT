import gradio as gr
from transformers import pipeline

print("Loading Whisper model...")

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
  device=0
)

print("Whisper model laoded!")

def transcribe_audio(audio_file):
  if audio_file is None:
    return "Please upload an audio file."

  result = pipe(audio_file, batch_size=8)
  return result["text"]

demo = gr.Interface(
  fn=transcribe_audio,
  inputs=gr.Audio(
    sources=["upload", "microphone"],
    type="filepath",
    label="Upload or Record Meeting Audio"
  ),
  outputs=gr.Textbox(
    label="Transcribed Text",
    lines=10
  ),
  title="Business AI Meeting Companion",
  description="Convert meeting recordings into text using OpenAI Whisper."
)

demo.launch()