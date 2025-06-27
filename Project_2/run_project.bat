@echo off
title AI Career Chatbot Launcher
echo =====================================================
echo 🚀 Launching AI Career Guidance Assistant Setup...
echo =====================================================

REM STEP 0: Set base path
set "BASE_PATH=E:\backup c\PythonInternship\Project_2"

REM STEP 1: Confirm the cleaned requirements file exists
if not exist "%BASE_PATH%\cleaned_final_requirements.txt" (
    echo ❌ ERROR: cleaned_final_requirements.txt not found in %BASE_PATH%
    pause
    exit /b
)

REM STEP 2: Upgrade pip
echo 📦 Upgrading pip...
call "C:\Program Files\Python39\python.exe" -m pip install --upgrade pip

REM STEP 3: Install dependencies from cleaned_final_requirements.txt
echo 🔄 Installing all dependencies from cleaned_final_requirements.txt...
cd /d "%BASE_PATH%"
call "C:\Program Files\Python39\python.exe" -m pip install -r cleaned_final_requirements.txt

REM ✅ STEP 3.1: Force install packaging==21.3 (fix for Rasa LegacyVersion bug)
call "C:\Program Files\Python39\python.exe" -m pip install packaging==21.3 --force-reinstall

REM ✅ STEP 3.5: Check if model files exist, else train them
if exist "%BASE_PATH%\data\intent_classifier.pkl" if exist "%BASE_PATH%\data\scaler.pkl" (
    echo ✅ intent_classifier.pkl and scaler.pkl already exist.
) else (
    echo ⚠️ Model files missing. Training intent classifier...
    call "C:\Program Files\Python39\python.exe" "%BASE_PATH%\train_intent_classifier.py"
)

REM STEP 4: Confirm Rasa project files exist
if not exist "%BASE_PATH%\rasa_project\config.yml" (
    echo ❌ ERROR: config.yml missing in rasa_project.
    pause
    exit /b
)
if not exist "%BASE_PATH%\rasa_project\domain.yml" (
    echo ❌ ERROR: domain.yml missing in rasa_project.
    pause
    exit /b
)
if not exist "%BASE_PATH%\rasa_project\data\nlu.yml" (
    echo ❌ ERROR: data/nlu.yml missing in rasa_project.
    pause
    exit /b
)
if not exist "%BASE_PATH%\rasa_project\data\stories.yml" (
    echo ❌ ERROR: data/stories.yml missing in rasa_project.
    pause
    exit /b
)


REM STEP 5: Train the Rasa model
cd /d "%BASE_PATH%\rasa_project"
echo 🧠 Training Rasa model...
call "C:\Program Files\Python39\python.exe" -m rasa train

REM STEP 6: Start Rasa Action Server in separate window and wait
echo 🛠️ Launching Rasa Action Server...
start "Rasa Actions" cmd /k "cd /d \"%BASE_PATH%\rasa_project\" && \"C:\Program Files\Python39\python.exe\" -m rasa run actions"

REM STEP 7: Delay to let actions start (optional: wait 5 seconds)
timeout /t 5 >nul

REM STEP 8: Start Rasa HTTP API Server
echo 🌐 Launching Rasa HTTP API Server...
start "Rasa Server" cmd /k "cd /d \"%BASE_PATH%\rasa_project\" && \"C:\Program Files\Python39\python.exe\" -m rasa run --enable-api --cors \"*\""


REM STEP 9: Start Streamlit App UI
echo 🖥️ Launching Streamlit App UI...
cd /d "%BASE_PATH%\streamlit_app"
start "" "C:\Program Files\Python39\python.exe" -m streamlit run app.py

echo =====================================================
echo ✅ All systems started! Visit http://localhost:8501
echo =====================================================
pause
