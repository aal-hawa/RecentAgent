#!/usr/bin/env bash
# RecentAgent stopper for macOS / Linux
PIDS=$(lsof -ti :8501 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs kill -9
    echo "RecentAgent stopped."
else
    echo "RecentAgent is not running."
fi
