from pathlib import Path

# Directory setup relative to script location
project_root = Path(__file__).resolve().parents[1]
processed_dir = project_root / "data" / "processed"
human_dir = processed_dir / "human"
ai_dir = processed_dir / "ai"

models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

def extract_mfcc_features():
    """Extracts MFCC features"""
    pass

def train():
    """Train the SVM model"""
    pass

if __name__ == "__main__":
  train()