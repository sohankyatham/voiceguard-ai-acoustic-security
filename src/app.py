import io
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import joblib
import librosa
import numpy as np
import soundfile as sf

# Streamlit page configuration
st.set_page_config(
    page_title="VoiceGuard - AI Acoustic Security",
    page_icon="",
    layout="wide",
)

# Environment & path setup 
project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / ".env")

model_path = project_root / "models" / "voiceguard_svm.pkl"


# Load trained ML model (cached in memory)
@st.cache_resource
def load_acoustic_model():
    if not model_path.exists():
        st.error(
            f"Model file not found at {model_path}. Please run src/train_model.py first"
        )
        st.stop()
    artifact = joblib.load(model_path)
    return artifact["model"], artifact["scaler"]

svm_model, scaler = load_acoustic_model()

# Real ML inference functions
def run_real_acoustic_inference(y, sr):
    """Extracts 26 MFCC features from audio signal and predicts using trained SVM"""
    # Extract 13 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_std = np.std(mfccs.T, axis=0)

    # Combine into 26-dimensional vector
    feature_vector = np.hstack([mfccs_mean, mfccs_std])

    # Scale and then predict w/ SVM
    features_scaled = scaler.transform([feature_vector])
    prediction = svm_model.predict(features_scaled)[0]  # 0 = Human, 1 = AI
    probabilities = svm_model.predict_proba(features_scaled)[0]

    spoof_prob = (
        probabilities[1] if prediction == 1 else (1.0 - probabilities[0])
    )
    return int(prediction), float(spoof_prob)



# User Interface
st.title("VoiceGuard - AI Acoustic Security")
st.markdown("""**Acoustic Anomaly Detection**
This tool intercepts voice traffic and performs MFCC analysis to detect biometric spoofing, voice cloning, and synthetic audio generation.
""")

# Telemetry Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="System Status", value="ACTIVE", delta="Secure")
col2.metric(label="Latency", value="42ms", delta="-3ms", delta_color="inverse")
col3.metric(label="Nodes Protected", value="142", delta="1")
col4.metric(
    label="Threats Blocked (24h)", value="8,492", delta="12%", delta_color="off"
)

st.divider()

# Audio Input Selection Allow user to upload file or record live with microphone
st.subheader("Intercept & Analyze Audio")
col_upload, col_record = st.columns(2)

with col_upload:
    audio_file = st.file_uploader(
        "Upload intercepted payload (.wav, .mp3, .ogg)",
        type=["wav", "mp3", "ogg"],
    )

with col_record:
    recorded_audio = st.audio_input("Or record live with microphone")

# Determine active audio source
audio_to_process = None
source_name = ""

if audio_file is not None:
  audio_to_process = audio_file
  source_name = audio_file.name
elif recorded_audio is not None:
  audio_to_process = recorded_audio
  source_name = "Live Microphone Recording"

# Display intercepted audio player
if audio_to_process is not None:
  st.success(f"Intercepted Audio Payload Ready: **{source_name}**")
  st.audio(audio_to_process)

  with st.status("Analyzing acoustic payload...", expanded=True) as status:
    st.write(f"📥 Intercepted source: **{source_name}**")

    try:
      st.write("⚙️ Extracting MFCCs and spectral structures...")

    except Exception as e:
      st.error(f"Error processing audio payload: {e}")
      status.update(label="Analysis Failed", state="error")