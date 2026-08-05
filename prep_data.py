from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence

downloads = Path.home() / "Downloads"

def process_and_chunk_audio(input_filename, output_subfolder, file_prefix):
    input_path = downloads / input_filename
    output_dir = downloads / "dataset" / output_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Could not find {input_filename} in Downloads!")
        return

    print(f"Processing {input_filename}...")
    sound = AudioSegment.from_file(input_path)
    
    # Split on silence: drops volume below -40dB lasting longer than 300ms
    chunks = split_on_silence(
        sound, 
        min_silence_len=300, 
        silence_thresh=-40, 
        keep_silence=150
    )
    
    valid_count = 0
    for chunk in chunks:
        # Keep audio chunks between 1.5 seconds and 5 seconds
        if 1500 <= len(chunk) <= 5000:
            output_file = output_dir / f"{file_prefix}_{valid_count}.wav"
            chunk.export(output_file, format="wav")
            valid_count += 1
            
    print(f"Successfully generated {valid_count} clean chunks in {output_dir}\n")

if __name__ == "__main__":
    process_and_chunk_audio("human_raw.wav", "human", "human")
    process_and_chunk_audio("ai_raw.wav", "ai", "ai")