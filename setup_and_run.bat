@echo off
title Smart Hostel System - Setup and First Run
echo ======================================================================
echo   SMART HOSTEL DEMAND PREDICTION & ROOM ALLOCATION SYSTEM
echo   Automated Setup Script for Windows
echo ======================================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your Windows PATH.
    echo Please install Python 3.10+ from python.org and check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo [1/4] Found Python:
python --version
echo.

:: 2. Create virtual environment if not already present
if not exist "venv\Scripts\activate.bat" (
    echo [2/4] Creating virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
) else (
    echo [2/4] Virtual environment (venv) already exists.
)
echo.

:: 3. Activate and install requirements
echo [3/4] Activating venv and installing requirements...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] pip install encountered an error.
    pause
    exit /b 1
)
echo.

:: 4. Run pipeline if models or outputs are missing
if not exist "outputs\evaluation_metrics.csv" (
    echo [4/4] Generating data, training ML models and running allocation pipeline...
    python run_pipeline.py
) else (
    echo [4/4] Pre-trained models and outputs found!
)
echo.

echo ======================================================================
echo   Setup Complete! Starting Streamlit Dashboard...
echo   The web app will open automatically in your default browser.
echo   Press Ctrl+C in this terminal window when you want to stop the app.
echo ======================================================================
echo.

streamlit run app.py
pause

