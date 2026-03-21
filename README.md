# Air Canvas 🎨

**Low-stimulus digital toy for kids — paint in the air with your hands.**

No scores. No feeds. No rewards. No dopamine loops. Just pure creativity, drawn in the air.

Air Canvas turns your webcam into a finger-painting canvas. Point your finger to draw, pinch to change colors, close your fist to pause. Two hands, eight colors, infinite imagination.

Built for kids who deserve to create without consuming.

## Demo

> 👆 Point = Draw  
> ✊ Fist = Stop  
> 🤏 Pinch = Change color  
> 🖐️ Palm (hold) = Clear canvas  
> **s** = Save drawing | **c** = Clear | **q** = Quit

## Features

- **Two-hand drawing** — left hand has 4 colors, right hand has 4 colors (8 total)
- **Pressure simulation** — brush gets thicker when moving slowly, thinner when fast
- **Neon glow effect** — drawings look magical on a dark background
- **Fullscreen mode** — immersive experience for kids
- **Save to PNG** — press `s` to keep their masterpiece
- **Zero accounts, zero internet** — everything runs locally on your Mac

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
