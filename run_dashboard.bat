@echo off
title Smart Hostel System - Dashboard
echo Starting Smart Hostel System Dashboard...
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Running initial setup first...
    call setup_and_run.bat
    exit /b
)

call venv\Scripts\activate.bat
streamlit run app.py
pause

