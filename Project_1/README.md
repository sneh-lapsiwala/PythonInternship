---

## 🧠 EmotionSense+ – Voice Emotion Recognition System

EmotionSense+ is an advanced **real-time voice emotion detection web app** built with **Streamlit**. It predicts the emotional tone of a speaker using `.wav` audio files, supporting both file uploads and microphone recordings.

---

### 🚀 Features

* 🎤 **Real-time Voice Recording**
* 🔊 **Upload `.wav` Files** for prediction
* 📈 **Waveform Visualization** of recorded audio
* 🧠 **Multi-Model Prediction** using:

  * `SVC` (Support Vector Machine)
  * `MLPClassifier` (Neural Network)
  * `SGDClassifier` (Linear SVM Approximation)
* 🗳️ **Voting-Based Prediction** from 3 recordings
* 📜 **Session History** & CSV Export
* 🧪 **Preprocessing**: Silence trimming, volume normalization
* ✅ Trained on **RAVDESS** dataset

---

### 📁 Project Structure

```bash
emotion-sense/
├── streamlit_app.py         # Main Streamlit app
├── train_model.py           # Script to train SVM, MLP, SGD models
├── requirements.txt         # Python dependencies
├── README.md                # You're here!
├── model/
│   ├── svm.pkl              # Trained SVM model
│   ├── mlp.pkl              # Trained MLP model
│   └── sgd.pkl              # Calibrated SGD model
├── audio/
│   └── uploaded_clips/      # Temporary recordings and uploads
├── utils/
│   └── feature_extraction.py # Preprocessing + MFCC extraction
└── assets/
    └── logo.jpg             # Optional app branding
```

---

### 🧪 How It Works

1. 🎙️ Record voice (3 samples) or upload a `.wav` file
2. 🧹 The audio is trimmed, normalized, and converted to MFCCs
3. 🧠 Three models predict emotions
4. 🗳️ Voting determines the final emotion
5. 📊 Results and audio history are displayed

---

### 🛠️ Setup Instructions

#### ✅ 1. Clone the repository

```bash
git clone https://github.com/yourname/emotion-sense.git
cd emotion-sense
```

#### ✅ 2. Install requirements

```bash
pip install -r requirements.txt
```

> If using microphone recording:

```bash
pip install streamlit-webrtc
```

#### ✅ 3. Train the models

```bash
python train_model.py
```

#### ✅ 4. Run the app

```bash
streamlit run streamlit_app.py
```

---

### 🎯 Sample Emotions Detected

* Angry
* Happy
* Neutral
* Sad
* Fear
* Disgust
* Calm
* Surprise

---

### 🧠 Future Improvements (Suggested)

* 🎬 Add emotion visualization (emoji, avatar, etc.)
* 🧵 Support longer audio or emotion timeline
* 📦 Deploy on HuggingFace Spaces or Streamlit Cloud

---

### 🙌 Credits

* Dataset: [RAVDESS](https://zenodo.org/record/1188976)
* Built with: `Streamlit`, `scikit-learn`, `librosa`, `joblib`

---