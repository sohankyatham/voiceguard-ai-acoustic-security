from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import re

# Load environment variables 
project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")

# Initialize client 
client = OpenAI()

# Define directory paths relative to the script
text_file_path = project_root / "data" / "sample_text.txt"
raw_data_dir = project_root / "data" / "raw"
raw_data_dir.mkdir(parents=True, exist_ok=True)


# Read sample text from file
if not text_file_path.exists():
    raise FileNotFoundError(f"Could not find {text_file_path}. Make sure sample_text.txt is in data/")
with open(text_file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()


# Clean Wikipedia bracket citations like [1] so TTS engine doesnt read it out loud
clean_text = re.sub(r'\[\d+\]', '', raw_text).strip()


# Generate AI voice with TTS engine and save the file
def generate_voice(voice_name: str, output_filename: str):
    output_path = raw_data_dir / output_filename
    print(f"Generating synthetic audio via OpenAI TTS API using voice '{voice_name}'...")

    # OpenAI API call
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice=voice_name,    
        input=clean_text,
        response_format="wav"
    )

    # Save the file 
    with open(output_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)

    print(f"Success! Saved to {output_path} \n")


if __name__ == "__main__":
    generate_voice("sage", "ai_raw_sage.wav")