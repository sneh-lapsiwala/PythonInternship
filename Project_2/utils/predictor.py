# predictor.py

import joblib

class CareerPredictor:
    def __init__(self, model_path, vectorizer_path, label_encoder_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)  # ✅ Add this

    def predict(self, user_input):
        vec = self.vectorizer.transform([user_input])
        pred_index = self.model.predict(vec)[0]
        prediction = self.label_encoder.inverse_transform([pred_index])[0]  # ✅ Decode label

        if hasattr(self.model, "predict_proba"):
            confidence = max(self.model.predict_proba(vec)[0])
        else:
            confidence = 1.0

        return prediction, confidence
