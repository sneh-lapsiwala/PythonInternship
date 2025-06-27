# train_intent_classifier.py

import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import joblib
import os

# Load intents
with open("data/intents.json", "r") as f:
    intents = json.load(f)

X = []
y = []

for intent in intents["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(tag)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Build a pipeline (Vectorizer + Classifier)
vectorizer = TfidfVectorizer()
model = SVC(kernel="linear", probability=True)

# Fit
X_vec = vectorizer.fit_transform(X)
model.fit(X_vec, y_encoded)

# Save
os.makedirs("data", exist_ok=True)
joblib.dump(model, "data/intent_classifier.pkl")
joblib.dump(vectorizer, "data/scaler.pkl")  # scaler = vectorizer here
  # ✔ Corrected line
joblib.dump(label_encoder, "data/label_encoder.pkl")

print("✅ Model training complete. Saved to data/")
