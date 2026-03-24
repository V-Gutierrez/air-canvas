# Air Canvas — Config
# Drawing app for kids: two-hand finger painting with MediaPipe

import os

CAMERA_INDEX = 1  # Skip iPhone Continuity Camera

DEBUG = False

DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5
MAX_HANDS = 2

# Canvas
CANVAS_BG_COLOR = (20, 20, 20)  # Near-black background
WINDOW_NAME = "Air Canvas 🎨"
FULLSCREEN = True

# Brush
BRUSH_THICKNESS = 12  # Base thickness
BRUSH_MIN_THICKNESS = 6  # Min when moving fast
BRUSH_MAX_THICKNESS = 24  # Max when moving slow
BRUSH_GLOW = True  # Neon glow effect
BRUSH_GLOW_RADIUS = 20  # Glow blur radius

# Speed → thickness mapping
SPEED_SLOW_THRESHOLD = 0.005  # Below this = max thickness
SPEED_FAST_THRESHOLD = 0.05  # Above this = min thickness

# Colors — bright, kid-friendly
LEFT_HAND_COLORS = [
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 0),  # Green
    (255, 255, 0),  # Yellow
]

RIGHT_HAND_COLORS = [
    (255, 100, 0),  # Orange
    (255, 0, 100),  # Pink
    (100, 100, 255),  # Light blue
    (255, 255, 255),  # White
]

# Gestures
PINCH_THRESHOLD = 0.06  # Thumb+index distance to cycle color
PINCH_COOLDOWN = 0.8  # Seconds between color cycles
CLEAR_HOLD_TIME = 1.5  # Open palm held = clear canvas
CLEAR_STILLNESS = 0.03  # Hand must be still to clear
THUMB_EXTENSION_THRESHOLD = 0.04

# Smoothing
DRAW_SMOOTHING = 0.5  # EMA alpha for brush position

RAINBOW_ENABLED = True
RAINBOW_KEY = ord("r")
RAINBOW_HUE_STEP = 3

DRAW_ALIVE_ENABLED = True
DRAW_ALIVE_KEY = ord("a")
DRAW_ALIVE_DELAY = 3.0
DRAW_ALIVE_FREQUENCY = 0.5
DRAW_ALIVE_SHIFT_PX = 3
DRAW_ALIVE_BREATHE_SCALE = 0.01

SHAPE_HUNT_ENABLED = False
SHAPE_HUNT_KEY = ord("h")
SHAPE_HUNT_SUCCESS_COVERAGE = 0.6
SHAPE_HUNT_TARGET_THICKNESS = 18
SHAPE_HUNT_START_SIZE = 140
SHAPE_HUNT_MIN_SIZE = 56
SHAPE_HUNT_SHRINK_STEP = 8

# Background themes — 'b' key cycles through; default is dark
BACKGROUND_THEME = "camera"
THEME_ENABLED = True
THEME_KEY = ord("b")
THEMES = ["camera", "dark", "space", "forest", "ocean"]
CAMERA_BG_DARKEN_ALPHA = 0.18

# Per-theme dot density for subtle static-dot backgrounds
THEME_DOT_COUNT = {
    "space": 220,  # Stars — bright tiny dots on deep purple
    "forest": 160,  # Fireflies/leaves — dim green dots on deep green
    "ocean": 180,  # Bubbles — soft blue dots on deep blue
}

# Avatars
AVATARS_ENABLED = True
AVATAR_SIZE = 80
AVATAR_BOB_AMPLITUDE = 6
AVATAR_BOB_SPEED = 1.6
AVATAR_LEFT = "penguin"  # Left hand gets penguin
AVATAR_RIGHT = "cat"  # Right hand gets cat

PALETTE_DWELL_TIME = 0.3
PALETTE_CIRCLE_RADIUS = 25
PALETTE_EDGE_MARGIN = 18
PALETTE_VERTICAL_GAP = 18
HAND_COLOR_BLOB_RADIUS = 60
CURSOR_FILL_ALPHA = 0.35
RAINBOW_ARC_THICKNESS = 18

PARTICLES_ENABLED = True
PARTICLE_MAX_COUNT = 100
PARTICLE_EMIT_COUNT = 3
PARTICLE_MIN_LIFE = 10
PARTICLE_MAX_LIFE = 24
PARTICLE_SPEED = 2.5
PARTICLE_DECAY = 0.82

SAVE_KEY = ord("s")  # Press 's' to save PNG
EXPORT_KEY = ord("p")
EXPORT_DIR = os.path.expanduser("~/Desktop/air-canvas-art")
EXPORT_UPSCALE = 2
SAVE_OVERLAY_DURATION = 3.0
SAVE_FLASH_DURATION = 0.1
SAVE_THUMBNAIL_WIDTH = 160
SAVE_THUMBNAIL_HEIGHT = 110
CLEAR_KEY = ord("c")  # Press 'c' to clear
QUIT_KEY = ord("q")  # Press 'q' to quit
