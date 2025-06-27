@echo off
title AI Career Chatbot Launcher
echo =====================================================
echo 🚀 Launching AI Career Guidance Assistant Setup...
echo =====================================================

REM === SETUP ===
set "BASE_PATH=E:\backup_c\PythonInternship\Project_2"
set "VENV_PATH=%BASE_PATH%\venv"
set "PYTHON=%VENV_PATH%\Scripts\python.exe"

REM === STEP 0: Create virtual environment if not exists ===
if not exist "%VENV_PATH%" (
    echo 📦 Creating virtual environment...
    call "C:\Program Files\Python39\python.exe" -m venv "%VENV_PATH%"
)

REM === STEP 1: Activate venv and upgrade pip ===
call "%VENV_PATH%\Scripts\activate"
echo 🔁 Upgrading pip...
call "%PYTHON%" -m pip install --upgrade pip

REM === STEP 2: Install dependencies ===
if not exist "%BASE_PATH%\cleaned_final_requirements.txt" (
    echo ❌ ERROR: cleaned_final_requirements.txt not found.
    pause
    exit /b
)
echo 📦 Installing requirements...
call "%PYTHON%" -m pip install -r "%BASE_PATH%\cleaned_final_requirements.txt"

REM ✅ Install key modules if not in requirements
echo 🔍 Ensuring all required modules are available...
call "%PYTHON%" -m pip install streamlit SpeechRecognition PyPDF2 pdfplumber PyMuPDF docx2txt


REM === STEP 2.1: Fix packaging bug for Rasa ===
call "%PYTHON%" -m pip install packaging==21.3 --force-reinstall

REM === STEP 3: Train intent classifier if needed ===
if exist "%BASE_PATH%\data\intent_classifier.pkl" if exist "%BASE_PATH%\data\scaler.pkl" (
    echo ✅ intent_classifier.pkl and scaler.pkl already exist.
) else (
    echo ⚠️ Training intent classifier...
    call "%PYTHON%" "%BASE_PATH%\train_intent_classifier.py"
)


REM === STEP 4: Check Rasa files exist ===
if not exist "%BASE_PATH%\rasa_project\config.yml" (
    echo ❌ config.yml missing
    pause
    exit /b
)
if not exist "%BASE_PATH%\rasa_project\domain.yml" (
    echo ❌ domain.yml missing
    pause
    exit /b
)
if not exist "%BASE_PATH%\rasa_project\data\nlu.yml" (
    echo ❌ nlu.yml missing
    pause
    exit /b
)
if not exist "%BASE_PATH%\rasa_project\data\stories.yml" (
    echo ❌ stories.yml missing
    pause
    exit /b
)

REM === STEP 5: Train Rasa model ===
cd /d "%BASE_PATH%\rasa_project"
echo 🧠 Training Rasa model...
call "%PYTHON%" -m rasa train

REM === STEP 6: Start Rasa Action Server ===
echo 🛠️ Starting Rasa Action Server...
start "Rasa Actions" cmd /k "cd /d E:\backup_c\PythonInternship\Project_2\rasa_project && E:\backup_c\PythonInternship\Project_2\venv\Scripts\python.exe -m rasa run actions"

REM === STEP 7: Wait to avoid connection errors ===
timeout /t 5 >nul

REM === STEP 8: Start Rasa HTTP API Server ===
echo 🌐 Starting Rasa API Server...
start "Rasa Server" cmd /k "cd /d E:\backup_c\PythonInternship\Project_2\rasa_project && E:\backup_c\PythonInternship\Project_2\venv\Scripts\python.exe -m rasa run --enable-api --cors \"*\""

call "%BASE_PATH%\venv\Scripts\python.exe" -m pip install SpeechRecognition
echo 🔍 Verifying Python location:
call "%PYTHON%" -c "import sys; print('Using:', sys.executable)"
call "%PYTHON%" -m pip list


REM STEP 9: Start Streamlit App UI
echo 🖥️ Launching Streamlit App UI...
cd /d E:\backup_c\PythonInternship\Project_2\streamlit_app
start "Streamlit UI" cmd /k %PYTHON% -m streamlit run "%BASE_PATH%\streamlit_app\app.py"


echo =====================================================
echo ✅ All systems started! Visit http://localhost:8501
echo =====================================================
pause
