#!/bin/bash
set -e

cd "$(dirname "$0")"

# Find Python 3.13 (pyobjc compat) or fall back
PYTHON=""
for p in python3.13 python3.12 python3.11 python3; do
    if command -v "$p" &>/dev/null; then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python 3 not found"
    exit 1
fi

echo "🐍 Using: $PYTHON ($($PYTHON --version))"

# Create venv if needed
if [ ! -d .venv ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON -m venv .venv
fi

source .venv/bin/activate

# Install deps
pip install -q -r requirements.txt

# Download model if needed
MODEL="models/hand_landmarker.task"
if [ ! -f "$MODEL" ]; then
    echo "📥 Downloading hand landmarker model..."
    mkdir -p models
    curl -sL "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -o "$MODEL"
fi

echo ""
echo "🎨 Starting Air Canvas!"
echo "   👆 Point = Draw"
echo "   ✊ Fist = Stop"
echo "   🤏 Pinch = Change color"
echo "   🖐️ Palm (hold) = Clear"
echo "   s = Save | c = Clear | q = Quit"
echo ""

python air_canvas.py
