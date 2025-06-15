# Training script placeholder
import os
import numpy as np
import librosa
import joblib
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# Emotion labels from RAVDESS dataset
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    features = np.hstack([
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(zcr.T, axis=0)
    ])
    return features

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ✅ Ensure model directory exists
os.makedirs("model", exist_ok=True)

print("🧼 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Save scaler BEFORE PCA
joblib.dump(scaler, "model/scaler.pkl")

print("🧬 Applying PCA...")
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# ✅ Save PCA after scaler
joblib.dump(pca, "model/pca.pkl")

# Use the reduced data for model training
X_train = X_train_reduced
X_test = X_test_reduced

# ✅ Train and save MLP
print("🧠 Training MLP model...")
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42, early_stopping=True)
mlp.fit(X_train, y_train)
joblib.dump(mlp, "model/mlp.pkl")
print("✅ MLP model saved to model/mlp.pkl")

print("🧠 Training SGDClassifier (SVM approximation)...")

base_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
base_model.fit(X_train, y_train)  # ✅ Pre-train before calibration

calibrated_sgd = CalibratedClassifierCV(base_model, cv="prefit")  # Use prefit to avoid CV error
calibrated_sgd.fit(X_train, y_train)

joblib.dump(calibrated_sgd, "model/sgd.pkl")
print("✅ Calibrated SGDClassifier saved to model/sgd.pkl")

# ✅ Train and save SVM
print("🧠 Training SVM model...")
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(probability=True), param_grid, cv=3)
grid.fit(X_train, y_train)
svm = grid.best_estimator_

joblib.dump(svm, "model/svm.pkl")
print("✅ SVM model saved to model/svm.pkl")


models = {"SVM": svm, "MLP": mlp, "SGD": calibrated_sgd}

best_model = max(models.items(), key=lambda m: m[1].score(X_test, y_test))[1]
joblib.dump(best_model, "model/emotion_model.pkl")
print(f"✅ Best model saved as emotion_model.pkl: {type(best_model)}")


# Classification reports
print("\n📊 Classification Reports") 

for name, model in models.items():
    print(f"\n🧠 {name} Report:")
    print(classification_report(y_test, model.predict(X_test)))
    print(f"✅ Train Accuracy: {model.score(X_train, y_train):.2f}")
    print(f"✅ Test Accuracy:  {model.score(X_test, y_test):.2f}")

# ✅ Voting Ensemble
print("🗳️ Creating VotingClassifier...")

from sklearn.ensemble import VotingClassifier

# Fit ensemble manually after all components are trained
ensemble = VotingClassifier(
    estimators=[
        ('svm', svm),
        ('mlp', mlp),
        ('sgd', calibrated_sgd)
    ],
    voting='soft'
)
# Do not call ensemble.fit() again — just use it for prediction

# Print individual model evaluations
for name, model in models.items():
    print(f"\n🧠 {name} Report:")
    print(classification_report(y_test, model.predict(X_test)))
    print(f"✅ Train Accuracy: {model.score(X_train, y_train):.2f}")
    print(f"✅ Test Accuracy:  {model.score(X_test, y_test):.2f}")

# Manual Soft Voting Ensemble
print("\n🧠 Manual Soft Voting Ensemble Report:")

# Ensure consistent class labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train + y_test)

# Get model probabilities
svm_proba = svm.predict_proba(X_test)
mlp_proba = mlp.predict_proba(X_test)
sgd_proba = calibrated_sgd.predict_proba(X_test)

# Average probabilities
avg_proba = (svm_proba + mlp_proba + sgd_proba) / 3
final_preds = np.argmax(avg_proba, axis=1)
final_labels = [svm.classes_[i] for i in final_preds]

# Report
print(classification_report(y_test, final_labels))

# Optional: Save individual models and label encoder for Streamlit use
print("💾 Saving final components...")
joblib.dump(svm, "model/svm.pkl")
joblib.dump(mlp, "model/mlp.pkl")
joblib.dump(calibrated_sgd, "model/sgd.pkl")
joblib.dump(le, "model/label_encoder.pkl")
print("✅ All models and encoder saved successfully.")
