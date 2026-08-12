import streamlit as st

# Streamlit page configuration
st.set_page_config(
    page_title="VoiceGuard - AI Acoustic Security",
    page_icon="",
    layout="wide",
)

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