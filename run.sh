#!/usr/bin/env bash
# Set up the venv + model on first run, then start the app.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if [ ! -d .venv ]; then
  echo "creating .venv ..."
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
  ./.venv/bin/pip install --quiet -r requirements.txt
fi

if [ ! -f models/hand_landmarker.task ]; then
  echo "downloading hand landmark model ..."
  mkdir -p models
  curl -sSL -o models/hand_landmarker.task "$MODEL_URL"
fi

exec ./.venv/bin/python main.py "$@"
