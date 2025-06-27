# 📁 streamlit_app/app.py

import streamlit as st
import requests
import speech_recognition as sr
import os
import pandas as pd
from PIL import Image
import sys
import platform

# 🌐 Setup absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# Read CSV with full path
career_map_path = os.path.join(DATA_DIR, 'career_map.csv')
career_map_df = pd.read_csv(career_map_path)


is_local = platform.system() in ["Windows", "Linux", "Darwin"]  # local OSes


# 🛠️ Add root project path to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Correct imports based on your existing utils/ folder
from utils.resume_parser import extract_text_from_pdf, extract_tables_from_pdf
from utils.predictor import CareerPredictor
  # ✅ Correct spelling
  # renamed correctly
#from utils.visualization import draw_graph
from utils.personality_mapper import map_personality
# If course recommender exists
from utils.course_recommender import recommend_courses

# ✅ Load your ML-based career predictor
model_path = os.path.join(DATA_DIR, "intent_classifier.pkl")
scaler_path = os.path.join(DATA_DIR, "scaler.pkl")
label_encoder_path = os.path.join(DATA_DIR, "label_encoder.pkl")
predictor = CareerPredictor(model_path, scaler_path, label_encoder_path)
# Set Streamlit page config
st.set_page_config(page_title="AI Career Counsellor", layout="centered")
st.title("🎓 AI Career Counsellor")

# --- Rasa config
RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"

def send_message_to_rasa(message, custom_data=None):
    payload = {
        "sender": "streamlit_user",
        "message": message
    }
    if custom_data:
        payload["customData"] = custom_data

    try:
        response = requests.post(RASA_SERVER_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Rasa connection error: {e}")
        return None

# --- Session & Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- UI Elements ---
st.markdown("### 📚 Your Current Education/Job Status")
status_options = ["Select...", "12th Pass", "Diploma", "BTech Student", "Graduate", "Postgraduate", "Working Professional"]
selected_status = st.selectbox("Select your current status", status_options)

st.markdown("Type or speak your interests/career goals:")
col1, col2 = st.columns(2)

user_input_text = ""
with col1:
    user_input_text = st.text_input("💬 Enter message", key="text_input_box")

with col2:
    if st.button("🎙 Voice Input"):
       if is_local:
          try:
              r = sr.Recognizer()
              with sr.Microphone() as source:
                  st.info("Listening...")
                  audio = r.listen(source)
              user_input_text = r.recognize_google(audio)
              st.success(f"You said: {user_input_text}")
              st.session_state.text_input_box = user_input_text
          except Exception as e:
              st.warning(f"Voice error: {e}")
       else: 
         st.warning("Voice input is only supported in local environment.")

# --- Upload PDFs ---
st.markdown("### 📄 Upload Resume and Subjects PDF")
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_file")
subject_files = st.file_uploader("Upload Subject PDFs", type=["pdf"], key="subject_files", accept_multiple_files=True)

extracted_resume_text = ""
extracted_subject_text = ""
extracted_subject_tables = []

if resume_file:
    try:
        extracted_resume_text = extract_text_from_pdf(resume_file)
        st.success("Resume text extracted!")
    except Exception as e:
        st.error(f"Resume extract error: {e}")

if subject_files:
    for file in subject_files:
        try:
            text = extract_text_from_pdf(file)
            tables = extract_tables_from_pdf(file)
            extracted_subject_text += text + "\n"
            extracted_subject_tables.extend(tables or [])
            st.success(f"Extracted: {file.name}")
        except Exception as e:
            st.error(f"{file.name} error: {e}")

# --- Predict using local ML model ---
if user_input_text:
    pred_label, confidence = predictor.predict(user_input_text)
    st.info(f"🔍 Our Model Prediction: **{pred_label}** with confidence **{round(confidence * 100, 2)}%**")

    # 📌 Show career path and suggested courses if field is found
    row = career_map_df[career_map_df["field"].str.lower() ==        pred_label.lower()]
    if not row.empty:
       st.markdown(f"**🛣 Career Path for {pred_label}:** {row.iloc[0]['career_path']}")
    
       st.markdown("### 🎓 Suggested Courses")
       courses = recommend_courses(pred_label)
       for category, course_list in courses.items():
           st.markdown(f"**{category}:**")
           for course in course_list:
               st.markdown(f"- {course}")

    else:
       st.warning("No career path found for this prediction.")




# --- Send message to Rasa ---
if user_input_text:
    st.session_state.messages.append({"role": "user", "content": user_input_text})
    with st.chat_message("user"):
        st.markdown(user_input_text)

    custom_data = {
        "status": selected_status if selected_status != "Select..." else None,
        "resume_text": extracted_resume_text or None,
        "subject_text": extracted_subject_text or None,
        "subject_tables": [t for t in extracted_subject_tables] if extracted_subject_tables else None
    }

    rasa_responses = send_message_to_rasa(user_input_text, custom_data)

    if rasa_responses:
        for response in rasa_responses:
            if "text" in response:
                st.session_state.messages.append({"role": "assistant", "content": response["text"]})
                with st.chat_message("assistant"):
                    st.markdown(response["text"])
            if "custom" in response:
                st.markdown("🎯 Custom Response from Rasa:")
                st.json(response["custom"])

# --- Default Suggestions ---
if not user_input_text and len(st.session_state.messages) == 0:
    st.markdown("#### 💡 Try these:")
    suggestions = ["I like coding", "Interested in business", "Want to be a designer", "I'm creative", "Suggest a career"]
    selected = st.selectbox("Suggestions", ["Select..."] + suggestions)
    if selected != "Select...":
        st.session_state.messages.append({"role": "user", "content": selected})
        rasa_responses = send_message_to_rasa(selected)
        if rasa_responses:
            for response in rasa_responses:
                if "text" in response:
                    st.session_state.messages.append({"role": "assistant", "content": response["text"]})
                    with st.chat_message("assistant"):
                        st.markdown(response["text"])


st.markdown("---\n📬 Made with ❤️ by your AI Career Guide")
