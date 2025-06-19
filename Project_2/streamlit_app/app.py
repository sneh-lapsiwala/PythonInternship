import streamlit as st
import requests # Make sure 'requests' is installed: pip install requests
import speech_recognition as sr
import os # For checking file existence if needed
import pandas as pd # Assuming you use pandas for table data
from PIL import Image # If you use PIL for image processing (e.g., from pdf2image)

# --- Import your utility functions ---
# Make sure your 'utils' folder is correctly placed and contains these functions
# For example, if you have utils/pdf_processor.py with these functions:
# from utils.pdf_processor import extract_text_from_pdf, extract_tables_from_pdf, extract_keywords_from_text
# OR if they are directly in utils/__init__.py or a single utils.py:
import sys
# Adjust this path if your 'utils' folder is not directly under 'AI_Career_Counsellor_Project'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
import utils # This assumes utils.py or utils/__init__.py directly contains the functions


# Set Streamlit page config
st.set_page_config(page_title="AI Career Counsellor", layout="centered")

# --- Function to communicate with Rasa ---
RASA_SERVER_URL = "http://localhost:5005/webhooks/rest/webhook"

def send_message_to_rasa(message, custom_data=None):
    payload = {
        "sender": "streamlit_user", # A unique sender ID for Streamlit
        "message": message
    }
    if custom_data:
        # You can send custom data as part of the payload, e.g., extracted text
        payload["customData"] = custom_data

    try:
        response = requests.post(RASA_SERVER_URL, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to Rasa server at {RASA_SERVER_URL}. Please ensure 'rasa run' is active.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Rasa server returned an error: {e}. Response: {response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- UI Starts ---
st.title("🎓 AI Career Counsellor")
st.write("Welcome! I’ll help guide your career path based on your interests, personality, and skills.")

# Initialize chat history (important for continuous conversation)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


st.markdown("### 📚 Current Education or Job Status")
status_options = ["Select...", "12th Pass", "Diploma", "BTech Student", "Graduate", "Postgraduate", "Working Professional"]
selected_status = st.selectbox("Your current status", status_options)

st.markdown("Type or speak your interests/career goals below:")

col1, col2 = st.columns(2)
user_input_text = "" # Renamed to avoid conflict with `user_input` in voice
with col1:
    user_input_text = st.text_input("💬 Enter your message", key="text_input_box")

with col2:
    if st.button("🎙 Voice Input", key="voice_button"):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Speak now...")
                audio = r.listen(source)
            user_input_text = r.recognize_google(audio)
            st.success(f"You said: {user_input_text}")
            # Ensure text_input_box updates with voice input
            st.session_state.text_input_box = user_input_text
        except sr.UnknownValueError:
            st.warning("Could not understand audio")
            user_input_text = ""
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            user_input_text = ""

# --- PDF Uploaders ---
st.markdown("### 📄 Upload Documents for Analysis")
resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"], key="resume_uploader")
subject_file = st.file_uploader("Upload Course/Subject PDFs", type=["pdf"], key="subject_uploader", accept_multiple_files=True)

extracted_resume_text = ""
extracted_subject_text = ""
extracted_subject_tables = []

if resume_file:
    with st.spinner("Extracting resume data..."):
        try:
            extracted_resume_text = utils.extract_text_from_pdf(resume_file)
            st.success("Resume text extracted!")
            # You might want to extract keywords here too
            # resume_keywords = utils.extract_keywords_from_text(extracted_resume_text)
            # st.json({"Resume Text Sample": extracted_resume_text[:500] + "..."}) # Display a sample
            # st.json({"Resume Keywords (sample)": resume_keywords[:5]}) # Display a sample
        except Exception as e:
            st.error(f"Error extracting resume: {e}")

if subject_file:
    for i, file in enumerate(subject_file):
        with st.spinner(f"Extracting data from {file.name}..."):
            try:
                # Assuming utils.extract_text_from_pdf and utils.extract_tables_from_pdf exist
                text_from_subject_pdf = utils.extract_text_from_pdf(file)
                tables_from_subject_pdf = utils.extract_tables_from_pdf(file)

                extracted_subject_text += text_from_subject_pdf + "\n"
                if tables_from_subject_pdf:
                    extracted_subject_tables.extend(tables_from_subject_pdf)
                st.success(f"Data extracted from {file.name}!")
                # st.json({f"Text from {file.name} (sample)": text_from_subject_pdf[:500] + "..."})
                # if tables_from_subject_pdf:
                #     st.json({f"Tables from {file.name} (sample)": tables_from_subject_pdf[0].df.head().to_dict('records')}) # Display first table head
            except Exception as e:
                st.error(f"Error extracting from {file.name}: {e}")

# --- Send Extracted Data to Rasa ---
# This is a crucial part. How do you want Rasa to receive this?
# Option 1: Append to user message (simple, but can make message very long)
# Option 2: Send as a custom event/payload (more structured, requires Rasa custom actions)

# Let's use a combination for demonstration.
# When a new text or voice input is given, send it.
# If PDF data is available, you could send it as a follow-up message/event.

# User text input or voice input is primary.
if user_input_text:
    st.session_state.messages.append({"role": "user", "content": user_input_text})
    with st.chat_message("user"):
        st.markdown(user_input_text)

    # Prepare custom data to send with the message
    custom_data_to_rasa = {
        "status": selected_status if selected_status != "Select..." else None,
        "resume_text": extracted_resume_text if extracted_resume_text else None,
        # Convert tables to a serializable format (e.g., list of lists or dicts)
        "subject_tables": [table.df.to_dict('records') for table in extracted_subject_tables] if extracted_subject_tables else None,
        "subject_text": extracted_subject_text if extracted_subject_text else None
    }

    # Send the main message + custom data to Rasa
    rasa_responses = send_message_to_rasa(user_input_text, custom_data=custom_data_to_rasa)

    if rasa_responses:
        for response in rasa_responses:
            if "text" in response:
                st.session_state.messages.append({"role": "assistant", "content": response["text"]})
                with st.chat_message("assistant"):
                    st.markdown(response["text"])
            # Handle custom payloads from Rasa, e.g., to trigger specific Streamlit visuals
            if "custom" in response:
                if response["custom"].get("type") == "display_graph":
                    st.write("--- Displaying a graph based on Rasa's suggestion ---")
                    # Here you'd call your actual graph display function, e.g.,
                    # display_graphs(response["custom"].get("graph_id"))
                    st.image("https://via.placeholder.com/300x200?text=Graph+Placeholder") # Example placeholder
                if response["custom"].get("type") == "skill_match_score":
                    score = response["custom"].get("score", "N/A")
                    st.info(f"Your skill match score: **{score}%**")
                    # And maybe display more details from response["custom"]

# Handle initial interaction or suggestions if no direct input yet
if not user_input_text and len(st.session_state.messages) == 0:
    st.markdown("#### 💡 Try these suggestions:")
    suggestions = ["I like coding", "I'm interested in business", "I want to be a doctor", "I enjoy painting", "What is my career path?"] # Added a general one
    selected_suggestion = st.selectbox("Quick options", ["Select..."] + suggestions, key="suggestion_box")
    if selected_suggestion != "Select...":
        # Simulate user input and send to Rasa
        st.session_state.messages.append({"role": "user", "content": selected_suggestion})
        with st.chat_message("user"):
            st.markdown(selected_suggestion)
        rasa_responses = send_message_to_rasa(selected_suggestion)
        if rasa_responses:
            for response in rasa_responses:
                if "text" in response:
                    st.session_state.messages.append({"role": "assistant", "content": response["text"]})
                    with st.chat_message("assistant"):
                        st.markdown(response["text"])

# Display selected status (can be sent to Rasa as custom data with the message)
if selected_status != "Select...":
    st.info(f"📜 You selected: *{selected_status}* (This info is now sent to Rasa as customData with your message)")


st.markdown("---")
# Hardcoded suggestions for courses/colleges can be replaced by Rasa's responses if needed.
st.subheader("🎓 Recommended Courses & Colleges (Example - will be driven by Rasa actions eventually)")
st.markdown("- B.Tech CSE @ IITs/NITs\n- Online: Coursera - Google IT Automation")


st.markdown("""
---
📬 Made with ❤ for students. | [GitHub](#) | [Contact](#)
""")