# Air Canvas — Config
# Drawing app for kids: two-hand finger painting with MediaPipe

CAMERA_INDEX = 1  # Skip iPhone Continuity Camera

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

# Smoothing
DRAW_SMOOTHING = 0.5  # EMA alpha for brush position

# Fun
DRAW_SOUND = True
SOUND_VOLUME = 0.25
SOUND_DURATION = 0.12

STICKERS_ENABLED = True
STAMP_SIZE = 24
STAMP_COOLDOWN = 0.8

RAINBOW_ENABLED = True
RAINBOW_KEY = ord("r")
RAINBOW_HUE_STEP = 3

PARTICLES_ENABLED = True
PARTICLE_MAX_COUNT = 100
PARTICLE_EMIT_COUNT = 3
PARTICLE_MIN_LIFE = 10
PARTICLE_MAX_LIFE = 24
PARTICLE_SPEED = 2.5
PARTICLE_DECAY = 0.82

SAVE_KEY = ord("s")  # Press 's' to save PNG
CLEAR_KEY = ord("c")  # Press 'c' to clear
QUIT_KEY = ord("q")  # Press 'q' to quit
