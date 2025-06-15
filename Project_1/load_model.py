import joblib
import os

model_path = os.path.join("model", "emotion_model.pkl")

try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully. Type: {type(model)}")
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_path}'.")
    print("Please ensure 'emotion_model.pkl' is in a 'model' subfolder relative to your script,")
    print("or provide the correct absolute path.")
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")