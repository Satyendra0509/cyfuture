import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import whisper
import torch

# Load Whisper model (cached to avoid reloading)
@st.cache_resource
def load_whisper_model():
    model = whisper.load_model("small")  
    return model

whisper_model = load_whisper_model()

# Load Emotion Detection Model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# Predict emotion from text
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Get emotion probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Transcribe audio using Whisper
def transcribe_audio(audio_file):
    st.info("Processing audio with Whisper...")
    try:
        result = whisper_model.transcribe(audio_file)
    except FileNotFoundError:
        st.error("FFmpeg not found. Please install it and ensure it's on your PATH.")
        return ""
    return result["text"]

# Streamlit UI
st.title("Speech & Text Emotion Detection")
st.subheader("Upload an Audio File or Enter Text to Detect Emotions")

# Tabs for Text and Audio Processing
tab1, tab2 = st.tabs(["📜 Text Input", "🎙️ Audio Input"])

# **Tab 1: Text Emotion Detection**
with tab1:
    with st.form(key='text_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Analyze Text Emotion')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]
            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

# **Tab 2: Audio Transcription and Emotion Detection**
with tab2:
    uploaded_file = st.file_uploader("Upload an Audio File", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")
        st.write("Transcribing...")

        # Save uploaded file temporarily
        with open("temp_audio.mp3", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Transcribe using Whisper
        transcription = transcribe_audio("temp_audio.mp3")

        st.success("Transcribed Text")
        st.write(transcription)

        # Emotion detection on transcribed text
        st.success("Emotion Analysis of Transcribed Text")
        transcribed_emotion = predict_emotions(transcription)
        emoji_icon = emotions_emoji_dict[transcribed_emotion]
        st.write(f"Predicted Emotion: {transcribed_emotion} {emoji_icon}")
