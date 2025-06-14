# Training script placeholder
import os
import numpy as np
import librosa
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm


# Emotion labels from RAVDESS dataset
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

X, y = [], []
dataset_path = "ravdess"  # Folder where you put .wav files

count = 0

for root, _, files in os.walk(dataset_path):
    for file in tqdm(files, desc="Extracting features"):
        if file.endswith(".wav"):
            try:
                parts = file.split("-")
                if len(parts) < 3:
                    raise ValueError("Invalid filename format")
                emotion = emotion_map[parts[2]]
                features = extract_features(os.path.join(root, file))
                X.append(features)
                y.append(emotion)
                count += 1
          
                if count % 500 == 0:
                    print(f"✅ Processed {count} audio files...")
            except Exception as e:
                print(f"⚠️ Skipped file {file} due to error: {e}")

print(f"✅ Total files processed: {count}")

print("🔄 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size=0.2, random_state=42)
print("✅ Split complete.")

import os
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

# ✅ Ensure model directory exists
os.makedirs("model", exist_ok=True)

# ✅ Train and save SVM
print("🧠 Training SVM model...")
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
joblib.dump(svm, "model/svm.pkl")
print("✅ SVM model saved to model/svm.pkl")

# ✅ Train and save MLP
print("🧠 Training MLP model...")
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
joblib.dump(mlp, "model/mlp.pkl")
print("✅ MLP model saved to model/mlp.pkl")

# ✅ Optional: Evaluate both models
print("📊 Evaluating SVM...")
svm_pred = svm.predict(X_test)
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))

print("📊 Evaluating MLP...")
mlp_pred = mlp.predict(X_test)
print("MLP Classification Report:\n", classification_report(y_test, mlp_pred))


print("🧠 Training SGDClassifier (SVM approximation)...")

from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
base_model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, verbose=1)
model = CalibratedClassifierCV(base_model, cv=5)  # ✅ Enables predict_proba
model.fit(X_train, y_train)

print("✅ Model training complete.")

joblib.dump(model, "model/sgd.pkl")
print("✅ SGDClassifier saved to model/sgd.pkl")

print("📊 Evaluating SGDClassifier...")
print("SGD Classification Report:\n",classification_report(y_test, model.predict(X_test)))

print("💾 Saving model to disk...")
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/emotion_model.pkl")
print("✅ Model saved successfully at model/emotion_model.pkl")
