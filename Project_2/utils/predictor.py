# Intent predictor
import joblib

class CareerPredictor:
    def __init__(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, user_input):
        vec = self.vectorizer.transform([user_input])
        prediction = self.model.predict(vec)[0]
        confidence = max(self.model.predict_proba(vec)[0])
        return prediction, confidence
