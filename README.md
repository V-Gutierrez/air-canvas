# Air Canvas 🎨

**Low-stimulus digital toy for kids — paint in the air with your hands.**

No scores. No feeds. No rewards. No dopamine loops. Just pure creativity, drawn in the air.

<p align="center">
  <img src="assets/demo.jpg" alt="Father and son painting in the air together" width="600">
</p>

Air Canvas turns your webcam into a finger-painting canvas. Point your finger to draw, pinch to change colors, flash a V-sign to stamp playful shapes, and toggle rainbow mode for extra color. Two hands, eight colors, infinite imagination.

Built for kids who deserve to create without consuming.

## Why Air Canvas?

### 🧠 Cognitive Development
- **Hand-eye coordination** — kids learn to control precise finger movements while watching the screen respond in real time
- **Spatial awareness** — understanding how their hand position maps to the canvas builds spatial reasoning
- **Cause and effect** — every gesture has an immediate, visible result

### 🎨 Creative Expression
- **No rules, no templates** — just a blank canvas and colors
- **Two-hand painting** — left hand and right hand have different color palettes, encouraging bilateral coordination
- **Pressure simulation** — move slowly for thick lines, fast for thin — teaches control and intentionality

### 🧘 Low-Stimulus Design
- **No scores or achievements** — removes performance anxiety
- **No social features** — no sharing, no likes, no comparison
- **No ads, no in-app purchases** — runs 100% locally, no internet needed
- **Dark canvas** — easy on the eyes, calming environment
- **No time pressure** — kids paint at their own pace

### 👨‍👩‍👧‍👦 Parent-Child Bonding
- **Two-hand support** — parent and child can paint together on the same canvas
- **Save & print** — press `s` to save their masterpiece as PNG
- **Screen time with purpose** — active creation instead of passive consumption

## Controls

| Gesture | Action |
|---------|--------|
| 👆 Point | Draw |
| ✌️ V-sign | Place sticker |
| ✊ Fist | Stop drawing |
| 🤏 Pinch | Change color |
| 🖐️ Palm (hold) | Clear canvas |
| **r** | Toggle rainbow mode |
| **s** | Save drawing |
| **c** | Clear canvas |
| **q** | Quit |

## Features

- **Two-hand drawing** — left hand has 4 colors, right hand has 4 colors (8 total)
- **Gesture stickers** — flash a V-sign to stamp stars, hearts, circles, and smileys
- **Rainbow mode** — press `r` to shift brush colors through the full spectrum while drawing
- **Sparkle particles** — a light particle trail follows the fingertip without overwhelming the screen
- **Fun sounds** — playful synthesized tones react to drawing, color changes, clears, and stamps
- **Pressure simulation** — brush gets thicker when moving slowly, thinner when fast
- **Neon glow effect** — drawings look magical on a dark background
- **Fullscreen mode** — immersive experience for kids
- **Save to PNG** — keep their masterpieces
- **Zero accounts, zero internet** — everything runs locally

## Requirements

- macOS (tested on Apple Silicon)
- Python 3.11+ (3.13 recommended)
- Webcam

## Quick Start

```bash
git clone https://github.com/V-Gutierrez/air-canvas.git
cd air-canvas
./run.sh
```

The script automatically:
1. Creates a Python virtual environment
2. Installs dependencies
3. Downloads the hand tracking model (~10MB)
4. Launches Air Canvas in fullscreen

## Camera Setup

By default, Air Canvas uses camera index `1` to skip iPhone Continuity Camera on MacBooks. If your webcam isn't detected, edit `config.py`:

```python
CAMERA_INDEX = 0  # Try 0 if camera doesn't open
```

## Configuration

All settings live in `config.py` — colors, brush thickness, gesture thresholds, and more. Kid-friendly defaults out of the box.

## How It Works

- **MediaPipe Hand Landmarker** — tracks 21 hand landmarks per hand at 30fps
- **OpenCV** — renders the canvas and camera feed
- **Gesture recognition** — simple heuristics (finger extension, pinch distance, palm detection)
- **Threaded camera** — camera reads run on a separate thread for smooth performance

## Built With

- [MediaPipe](https://developers.google.com/mediapipe) — hand tracking
- [OpenCV](https://opencv.org/) — rendering
- [NumPy](https://numpy.org/) — math

## License

MIT — do whatever you want with it. Make your kids happy. ✌️
