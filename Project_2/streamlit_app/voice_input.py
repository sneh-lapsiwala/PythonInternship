# Voice input functionality
import speech_recognition as sr

def get_voice_input():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎤 Listening for input...")
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        return f"API error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
