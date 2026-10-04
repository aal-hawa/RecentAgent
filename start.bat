@echo off
title RecentAgent
cd /d "%~dp0"

rem Tip: to force a dependency reinstall, delete the .venv folder and run this again.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    echo         then re-run start.bat
    pause
    exit /b 1
)

if not exist .venv (
    echo [1/2] Creating virtual environment + installing dependencies ^(first run only, takes a few minutes^)...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -q --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
)

".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Dependencies missing - reinstalling...
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
    ".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Could not install dependencies. Check your internet connection and re-run start.bat
        pause
        exit /b 1
    )
)

echo [2/2] Starting RecentAgent on http://localhost:8501 ...
echo       To stop: double-click stop.bat - the server AND this window will close.
if not exist .nobrowser start "RA-browser" /min cmd /c "ping -n 6 127.0.0.1 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py
