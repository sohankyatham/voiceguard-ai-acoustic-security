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

