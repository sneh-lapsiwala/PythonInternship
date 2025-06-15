import streamlit as st
import os, time, joblib, numpy as np, pandas as pd
from datetime import datetime
from utils.feature_extraction import extract_features_from_audio

st.set_page_config(
    page_title="EmotionSense+",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎤 EmotionSense+: Advanced Voice Emotion Detector")

if os.path.exists("assets/logo.jpg"):
    st.image("assets/logo.jpg", width=150)
else:
    st.image("https://via.placeholder.com/150?text=EmotionSense+", width=150)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload", "Record", "History", "About"])

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_model():
    return joblib.load("model/emotion_model.pkl")

@st.cache_data
def extract_features_cached(path):
    return extract_features_from_audio(path)

def predict_emotion(audio_path, uploaded_file_name):

    model = load_model()
  
    features = extract_features_cached(audio_path)

    if features is None:
        st.error("❌ Feature extraction failed (e.g., too short/silent audio).")
        return

    if features.shape[0] != 53:
        st.error(f"❌ Extracted {features.shape[0]} features, but 53 were expected. Check the feature extraction code.")
        return

    features = features.reshape(1, -1)
    scaler = joblib.load("model/scaler.pkl")
    features_scaled = scaler.transform(features)

    # ✅ If model expects PCA input, apply PCA
    try:
        pca = joblib.load("model/pca.pkl")
        features_scaled = pca.transform(features_scaled)
    except FileNotFoundError:
        pass  # Do not apply PCA if not found

    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]


    st.audio(audio_path, format='audio/wav')
    st.success(f"🎯 Predicted Emotion: **{prediction.capitalize()}**")
    st.toast("Emotion prediction complete ✅", icon="🔍")
    st.bar_chart({label: [prob] for label, prob in zip(model.classes_, proba)})

    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "file": uploaded_file_name,
        "prediction": prediction
    })

if page == "Upload":
    st.header("🔊 Upload Audio File")
    uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])

    if uploaded_file:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.warning("File too large. Please upload a smaller audio file.")
            st.stop()

        os.makedirs("audio/uploaded_clips", exist_ok=True)
        audio_path = f"audio/uploaded_clips/{int(time.time())}.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())

        predict_emotion(audio_path, uploaded_file.name)

elif page == "Record":
    st.header("🎙️ Record Voice (3 Samples)")

    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    import queue
    import tempfile
    import soundfile as sf
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt

    # Helper: Show waveform
    def show_waveform(audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            fig, ax = plt.subplots()
            ax.plot(y)
            ax.set_title("🎵 Recorded Audio Waveform")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ Could not plot waveform: {e}")

    # Instruction
    st.markdown("🔁 Please record 3 short audio samples (~2–3 seconds) for better prediction accuracy.")

    predictions = []

    for i in range(3):
        st.subheader(f"🎤 Sample {i+1}")
        record_key = f"emotion-recorder-{i}"

        ctx = webrtc_streamer(
            key=record_key,
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=256,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        if ctx.audio_receiver:
            st.success("🎙️ Recording started... Speak now!")

            audio_frames = []
            try:
                while True:
                    audio_frame = ctx.audio_receiver.get_frames(timeout=2)
                    for frame in audio_frame:
                        audio_frames.append(frame.to_ndarray().flatten())
                    if len(audio_frames) > 100:
                        break
            except queue.Empty:
                st.warning("⚠️ No audio received.")

            if audio_frames:
                st.info("✅ Recording complete. Preparing predictions...")
                predictions = []
                for i in range(3):
                    st.markdown(f"🔊 Analyzing Sample {i+1}")
                    audio_data = np.concatenate(audio_frames)
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(temp_file.name, audio_data, samplerate=48000)

                    st.audio(temp_file.name)
                    show_waveform(temp_file.name)

                # Preprocess and validate
                y, sr = librosa.load(temp_file.name, sr=22050)
                y, _ = librosa.effects.trim(y)
                y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

                if np.max(np.abs(y)) < 0.01 or len(y) < sr * 1.5:
                    st.warning(f"⚠️ Sample {i+1} was too short or too quiet. Skipped.")
                else:
                    # Predict
                    pred = predict_emotion(temp_file.name, f"recorded_{i+1}.wav")
                    if pred:
                        predictions.append(pred)

        st.markdown("---")

    # Majority vote if all samples are collected
    if len(predictions) == 3:
        final_pred = max(set(predictions), key=predictions.count)
        st.success(f"🧠 Final Emotion Prediction (Voting Result): **{final_pred.upper()}**")

elif page == "History":
    st.header("📜 Session History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        st.download_button("📥 Download History as CSV", df.to_csv(index=False), "history.csv", "text/csv")
    else:
        st.info("No history yet. Upload or record a file first.")

elif page == "About":
    st.header("ℹ️ About EmotionSense+")
    st.markdown("""
        EmotionSense+ is a voice emotion detection tool that uses MFCC features and an SVM model 
        to detect emotions like Angry, Happy, Sad, and more from `.wav` audio files.
    """)
