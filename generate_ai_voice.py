from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import re

load_dotenv()

# Initialize client 
client = OpenAI()

# Define directory paths relative to the script
script_dir = Path(__file__).parent
text_file_path = script_dir / "sample_text.txt"
# Get Downloads folder path 
downloads_path = Path.home() / "Downloads" / "ai_raw.wav"

# Read sample text from file
if not text_file_path.exists():
    raise FileNotFoundError(f"Could not find {text_file_path}. Make sure sample_text.txt is in the same folder")
with open(text_file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Strip Wikipedia bracket citations like [1] so TTS engine doesnt read it out loud
clean_text = re.sub(r'\[\d+\]', '', raw_text).strip()

print("Generating synthetic audio via OpenAI TTS API...")

# OpenAI API call
response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",    
    input=clean_text,
    response_format="wav"
)

# Save the file 
with open(downloads_path, "wb") as f:
    for chunk in response.iter_bytes():
        f.write(chunk)

print(f"Success! Saved directly to your Downloads: {downloads_path}")