import torch
from transformers import pipeline
print("Loading summarization model...")

summarizer = pipeline(
  "summarization",
  model="sshleifer/distilbart-cnn-12-6",
  device=0 if torch.cuda.is_available() else -1
)

print("Summarization model loaded!")

meeting_text = """
Hello, I want to speak fast because I want to test four speech-to-text applications.
Today, the weather is sunny with a slight breeze, making a perfect day for outdoor
activity. Later I plan to visit a busy local park, maybe even a picnic.
The test is designed to assess the accuracy and responsiveness of the speech-to-text
feature. Thank you for participating in this test.
"""

result = summarizer(
  meeting_text,
  max_length=100,
  min_length=30,
  do_sample=False
)

summary = result[0]["summary_text"]

print("\n========== MEETING SUMMARY ==========")
print(summary.strip())
print("\n========== SUMMARY COMPLETE ==========")