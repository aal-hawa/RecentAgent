#!/usr/bin/env bash
# RecentAgent launcher for macOS / Linux
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "[2/3] Installing dependencies..."
pip install -q -r requirements.txt
echo "[3/3] Starting RecentAgent..."
streamlit run app.py
