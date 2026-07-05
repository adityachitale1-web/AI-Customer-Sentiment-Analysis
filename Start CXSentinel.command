#!/bin/bash
# Double-click this file in Finder to start CXSentinel (API + dashboard).
# Both servers keep running after you close the terminal window.

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

echo "Starting CXSentinel…"

# API (port 8000)
if lsof -ti :8000 >/dev/null 2>&1; then
  echo "• API already running on port 8000"
else
  (cd "$DIR/3_API" && nohup ../.venv/bin/python -m uvicorn main:app --port 8000 \
     >> "$DIR/api.log" 2>&1 &)
  echo "• API starting on http://127.0.0.1:8000 (log: api.log)"
fi

# Dashboard (port 8501)
if lsof -ti :8501 >/dev/null 2>&1; then
  echo "• Dashboard already running on port 8501"
else
  (cd "$DIR/5_Dashboard" && nohup ../.venv/bin/python -m streamlit run app.py \
     --server.port 8501 --server.headless true >> "$DIR/dashboard.log" 2>&1 &)
  echo "• Dashboard starting on http://localhost:8501 (log: dashboard.log)"
fi

# Wait for the API to come up (model loading takes ~10s), then open the site
echo "• Waiting for the model to load…"
for i in $(seq 1 60); do
  curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  sleep 1
done
open "http://localhost:8501"
echo "Done — CXSentinel is up. You can close this window."
