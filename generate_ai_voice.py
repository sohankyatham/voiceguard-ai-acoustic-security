from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Initialize client 
client = OpenAI()

# Get Downloads folder path 
downloads_path = Path.home() / "Downloads" / "ai_raw.wav"

# Paste ~3-5 minutes of text here like a Wikipedia article or something
text_input = """
The history of Rome
"""

print("Generating synthetic audio via OpenAI TTS API...")

# OpenAI API call
response = client.audio.speech.create(
    model="tts-1-hd",
    voice="shimmer",    # Choices: alloy, nova, echo, fable, onyx, shimmer
    input=text_input,
    response_format="wav"
)

# Save the file 
with open(downloads_path, "wb") as f:
    for chunk in response.iter_bytes():
        f.write(chunk)

print(f"Success! Saved directly to your Downloads: {downloads_path}")