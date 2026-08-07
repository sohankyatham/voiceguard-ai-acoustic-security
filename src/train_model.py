from pathlib import Path
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

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

    print(f"Checking for human files in: {human_dir.resolve()}")
    print(f"Checking for AI files in: {ai_dir.resolve()}")

    print("Extracting MFCC features from processed audio chunks...")

    X = []
    y = []

    # Process the Human Audio Chunks (label: 0)
    human_files = list(human_dir.glob("*.wav"))
    for f in human_files:
       X.append(extract_mfcc_features(f))
       y.append(0)

    print(f"Loaded {len(human_files)} human audio samples.")

    # Process the AI Audio Chunks (label: 1)
    ai_files = list(ai_dir.glob("*.wav"))
    for f in ai_files:
       X.append(extract_mfcc_features(f))
       y.append(1)

    print(f"Loaded {len(ai_files)} AI audio samples.")

    X = np.array(X)
    y = np.array(y)

    # Stratified split maintains ratio across train and test sets
    X_train, X_test, y_train, y_test, = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n Training SVM Model on {len(X_train)} samples...")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM with Probability Calibration
    base_svm = SVC(kernel="rbf", C=1.0, random_state=42)
    model = CalibratedClassifierCV(estimator=base_svm, ensemble=False)
    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print(f"MODEL ACCURACY: {acc * 100:.2f}%")
    print("=" * 40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Human (0)", "AI (1)"]))




if __name__ == "__main__":
  train()