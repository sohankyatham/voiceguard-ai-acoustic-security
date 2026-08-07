from pathlib import Path
import librosa
import numpy as np

# Directory setup relative to script location
project_root = Path(__file__).resolve().parents[1]
processed_dir = project_root / "data" / "processed"
human_dir = processed_dir / "human"
ai_dir = processed_dir / "ai"

models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

def extract_mfcc_features(file_path):
    """Extracts MFCC features and computes mean + std dev"""
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)
    return np.hstack([mfccs_mean, mfccs_std])

def train():
    """Train the SVM model"""
    pass

if __name__ == "__main__":
  train()