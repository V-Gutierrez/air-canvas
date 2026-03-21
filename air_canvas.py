#!/usr/bin/env python3
"""Air Canvas — Two-hand finger painting with MediaPipe for kids."""

import cv2
import numpy as np
import math
import time
import os
import io
import wave
import shutil
import tempfile
import subprocess
import random
import threading
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

from config import *

# ---------------------------------------------------------------------------
# MediaPipe Task API setup (0.10.33+)
# ---------------------------------------------------------------------------
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Landmark indices
INDEX_TIP = 8
INDEX_PIP = 6
THUMB_TIP = 4
THUMB_IP = 3
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

SOUND_SAMPLE_RATE = 22050
STAMP_SEQUENCE = ("star", "heart", "circle", "smiley")
SOUND_EVENT_FREQUENCIES = {
    "draw_start": 392,
    "color_change": 523,
    "clear": 262,
    "stamp": 659,
}


def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading hand_landmarker model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


class CameraThread:
    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._frame: Optional[np.ndarray] = None
        self._new_frame = False
        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ret, frame = self._cap.read()
            if not self._running:
                break
            if not ret:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._new_frame = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            is_new_frame = self._new_frame
            self._new_frame = False
            return is_new_frame, self._frame.copy()

    def stop(self):
        self._running = False
        self._cap.release()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Gesture helpers
# ---------------------------------------------------------------------------


def is_finger_extended(landmarks, tip_idx, pip_idx) -> bool:
    """A finger is extended if TIP is above PIP (lower y = higher on screen)."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def is_fist(landmarks) -> bool:
    """All 4 fingers curled (not thumb — thumb is unreliable for kids)."""
    return (
        not is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
        and not is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not is_finger_extended(landmarks, RING_TIP, RING_PIP)
        and not is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP)
    )


def is_open_palm(landmarks) -> bool:
    """All 5 fingers extended."""
    return (
        is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
        and is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and is_finger_extended(landmarks, RING_TIP, RING_PIP)
        and is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP)
    )


def is_pointing(landmarks) -> bool:
    """Only index finger extended — drawing mode."""
    return (
        is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
        and not is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not is_finger_extended(landmarks, RING_TIP, RING_PIP)
        and not is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP)
    )


def pinch_distance(landmarks) -> float:
    dx = landmarks[THUMB_TIP].x - landmarks[INDEX_TIP].x
    dy = landmarks[THUMB_TIP].y - landmarks[INDEX_TIP].y
    return math.hypot(dx, dy)


def fingertip_pos(landmarks, frame_w, frame_h) -> Tuple[int, int]:
    x = int(landmarks[INDEX_TIP].x * frame_w)
    y = int(landmarks[INDEX_TIP].y * frame_h)
    return x, y


def v_sign_center(landmarks, frame_w, frame_h) -> Tuple[int, int]:
    x = int(((landmarks[INDEX_TIP].x + landmarks[MIDDLE_TIP].x) / 2) * frame_w)
    y = int(((landmarks[INDEX_TIP].y + landmarks[MIDDLE_TIP].y) / 2) * frame_h)
    return x, y


def is_v_sign(landmarks) -> bool:
    return (
        is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
        and is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not is_finger_extended(landmarks, RING_TIP, RING_PIP)
        and not is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP)
    )


def hue_to_bgr(hue: int) -> Tuple[int, int, int]:
    hsv = np.array([[[int(hue) % 180, 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def generate_tone(frequency: int, duration: float, volume: float) -> np.ndarray:
    sample_count = max(1, int(SOUND_SAMPLE_RATE * duration))
    t = np.arange(sample_count, dtype=np.float32) / SOUND_SAMPLE_RATE
    tone = np.sin(2 * np.pi * frequency * t)
    fade_count = min(sample_count // 2, max(1, int(SOUND_SAMPLE_RATE * 0.01)))
    fade_in = np.linspace(0.0, 1.0, fade_count, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_count, dtype=np.float32)
    tone[:fade_count] *= fade_in
    tone[-fade_count:] *= fade_out
    return (tone * float(volume)).astype(np.float32)


def tone_to_wav_bytes(samples: np.ndarray) -> bytes:
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SOUND_SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def resolve_sound_player() -> Optional[str]:
    if shutil.which("afplay"):
        return "afplay"
    for command in ("aplay", "paplay"):
        if shutil.which(command):
            return command
    return None


class SoundEngine:
    def __init__(self, enabled: bool, max_concurrent: int = MUSIC_MAX_CONCURRENT):
        self.enabled = enabled
        self.player = resolve_sound_player() if enabled else None
        self.tones = {
            name: tone_to_wav_bytes(generate_tone(freq, SOUND_DURATION, SOUND_VOLUME))
            for name, freq in SOUND_EVENT_FREQUENCIES.items()
        }
        self._active_count = 0
        self._count_lock = threading.Lock()
        self._max_concurrent = max_concurrent

    def play(self, event_name: str):
        if not self.enabled or self.player is None:
            return
        tone = self.tones.get(event_name)
        if tone is None:
            return
        self._launch(tone)

    def play_tone(self, frequency: int):
        if not self.enabled or self.player is None:
            return
        with self._count_lock:
            if self._active_count >= self._max_concurrent:
                return
        wav = tone_to_wav_bytes(generate_tone(frequency, SOUND_DURATION, SOUND_VOLUME))
        self._launch(wav)

    def _launch(self, wav_bytes: bytes):
        with self._count_lock:
            if self._active_count >= self._max_concurrent:
                return
            self._active_count += 1
        threading.Thread(
            target=self._play_bytes, args=(wav_bytes,), daemon=True
        ).start()

    def _play_bytes(self, wav_bytes: bytes):
        temp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(wav_bytes)
                temp_path = temp_file.name
            if self.player is None:
                return
            command: List[str] = [self.player]
            if self.player in {"aplay", "paplay"}:
                command.append("-q")
            command.append(temp_path)
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            return
        finally:
            with self._count_lock:
                self._active_count -= 1
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


_sounddevice = None
try:
    import sounddevice as _sounddevice

    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False


def draw_star(canvas, center: Tuple[int, int], size: int, color: Tuple[int, int, int]):
    outer = max(4, size)
    inner = max(2, int(outer * 0.45))
    points = []
    for index in range(10):
        angle = (math.pi / 5 * index) - (math.pi / 2)
        radius = outer if index % 2 == 0 else inner
        points.append(
            [
                int(center[0] + math.cos(angle) * radius),
                int(center[1] + math.sin(angle) * radius),
            ]
        )
    polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [polygon], color)


def draw_heart(canvas, center: Tuple[int, int], size: int, color: Tuple[int, int, int]):
    radius = max(4, size // 2)
    cx, cy = center
    cv2.circle(canvas, (cx - radius // 2, cy - radius // 3), radius // 2, color, -1)
    cv2.circle(canvas, (cx + radius // 2, cy - radius // 3), radius // 2, color, -1)
    triangle = np.array(
        [
            [cx - radius, cy - radius // 4],
            [cx + radius, cy - radius // 4],
            [cx, cy + radius],
        ],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [triangle], color)


def draw_smiley(
    canvas, center: Tuple[int, int], size: int, color: Tuple[int, int, int]
):
    radius = max(6, size)
    cx, cy = center
    face_color = tuple(min(255, component + 40) for component in color)
    cv2.circle(canvas, (cx, cy), radius, face_color, -1)
    cv2.circle(canvas, (cx, cy), radius, color, 2)
    eye_radius = max(1, radius // 6)
    cv2.circle(canvas, (cx - radius // 3, cy - radius // 4), eye_radius, color, -1)
    cv2.circle(canvas, (cx + radius // 3, cy - radius // 4), eye_radius, color, -1)
    cv2.ellipse(
        canvas,
        (cx, cy + radius // 6),
        (max(2, radius // 2), max(2, radius // 3)),
        0,
        0,
        180,
        color,
        2,
    )


def draw_stamp(
    canvas,
    center: Tuple[int, int],
    stamp_type: str,
    size: int,
    color: Tuple[int, int, int],
):
    if stamp_type == "star":
        draw_star(canvas, center, size, color)
        return
    if stamp_type == "heart":
        draw_heart(canvas, center, size, color)
        return
    if stamp_type == "circle":
        cv2.circle(canvas, center, max(4, size), color, -1)
        return
    draw_smiley(canvas, center, size, color)


def generate_theme_background(theme: str, width: int, height: int) -> np.ndarray:
    if theme == "dark":
        return np.full((height, width, 3), CANVAS_BG_COLOR, dtype=np.uint8)
    if theme == "space":
        bg = np.full((height, width, 3), (30, 10, 20), dtype=np.uint8)
        rng = random.Random(42)
        for _ in range(THEME_DOT_COUNT.get("space", 220)):
            x = rng.randint(0, width - 1)
            y = rng.randint(0, height - 1)
            brightness = rng.randint(140, 230)
            bg[y, x] = (brightness, brightness, brightness)
        return bg
    if theme == "forest":
        bg = np.full((height, width, 3), (10, 28, 10), dtype=np.uint8)
        rng = random.Random(7)
        for _ in range(THEME_DOT_COUNT.get("forest", 160)):
            x = rng.randint(0, width - 1)
            y = rng.randint(0, height - 1)
            g = rng.randint(45, 90)
            bg[y, x] = (5, g, 5)
        return bg
    if theme == "ocean":
        bg = np.full((height, width, 3), (60, 30, 10), dtype=np.uint8)
        rng = random.Random(13)
        for _ in range(THEME_DOT_COUNT.get("ocean", 180)):
            x = rng.randint(0, width - 1)
            y = rng.randint(0, height - 1)
            b = rng.randint(100, 180)
            g = rng.randint(50, 100)
            bg[y, x] = (b, g, 15)
        return bg
    return np.full((height, width, 3), CANVAS_BG_COLOR, dtype=np.uint8)


def draw_penguin(canvas, center: Tuple[int, int], size: int):
    cx, cy = center
    # Body
    cv2.ellipse(
        canvas, (cx, cy + 5), (size, int(size * 1.2)), 0, 0, 360, (30, 30, 30), -1
    )
    # Belly
    cv2.ellipse(
        canvas,
        (cx, cy + 10),
        (int(size * 0.6), int(size * 0.9)),
        0,
        0,
        360,
        (240, 240, 240),
        -1,
    )
    # Eyes
    off_x = size // 3
    off_y = size // 2
    cv2.circle(canvas, (cx - off_x, cy - off_y), 4, (255, 255, 255), -1)
    cv2.circle(canvas, (cx + off_x, cy - off_y), 4, (255, 255, 255), -1)
    cv2.circle(canvas, (cx - off_x, cy - off_y), 2, (0, 0, 0), -1)
    cv2.circle(canvas, (cx + off_x, cy - off_y), 2, (0, 0, 0), -1)
    # Beak
    pts = np.array([[cx - 4, cy - 5], [cx + 4, cy - 5], [cx, cy + 2]], np.int32)
    cv2.fillPoly(canvas, [pts], (0, 165, 255))


def draw_cat(canvas, center: Tuple[int, int], size: int):
    cx, cy = center
    color = (200, 200, 200)
    # Ears
    pts_l = np.array(
        [[cx - size + 5, cy - size], [cx - 5, cy - 5], [cx - size, cy]], np.int32
    )
    pts_r = np.array(
        [[cx + size - 5, cy - size], [cx + 5, cy - 5], [cx + size, cy]], np.int32
    )
    cv2.fillPoly(canvas, [pts_l], color)
    cv2.fillPoly(canvas, [pts_r], color)
    # Head
    cv2.circle(canvas, (cx, cy), size - 2, color, -1)
    # Eyes
    off_x = size // 3
    off_y = size // 4
    cv2.circle(canvas, (cx - off_x, cy - off_y), 4, (0, 255, 0), -1)
    cv2.circle(canvas, (cx + off_x, cy - off_y), 4, (0, 255, 0), -1)
    cv2.circle(canvas, (cx - off_x, cy - off_y), 1, (0, 0, 0), -1)
    cv2.circle(canvas, (cx + off_x, cy - off_y), 1, (0, 0, 0), -1)
    # Nose
    cv2.circle(canvas, (cx, cy + 5), 3, (150, 100, 255), -1)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: int
    color: Tuple[int, int, int]
    radius: int


class ParticleSystem:
    def __init__(self, max_particles: int = PARTICLE_MAX_COUNT):
        self.max_particles = max_particles
        self.particles: List[Particle] = []

    def emit(self, position: Tuple[int, int], color: Tuple[int, int, int], count: int):
        if len(self.particles) >= self.max_particles:
            return
        available_slots = self.max_particles - len(self.particles)
        emit_count = min(count, available_slots)
        for _ in range(emit_count):
            life = random.randint(PARTICLE_MIN_LIFE, PARTICLE_MAX_LIFE)
            angle = random.uniform(0, math.tau)
            speed = random.uniform(0.4, PARTICLE_SPEED)
            self.particles.append(
                Particle(
                    x=float(position[0]),
                    y=float(position[1]),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    life=life,
                    color=color,
                    radius=random.randint(1, 3),
                )
            )

    def update(self):
        next_particles: List[Particle] = []
        for particle in self.particles:
            particle.x += particle.vx
            particle.y += particle.vy
            particle.vx *= PARTICLE_DECAY
            particle.vy = particle.vy * PARTICLE_DECAY + 0.08
            particle.life -= 1
            if particle.life > 0:
                next_particles.append(particle)
        self.particles = next_particles

    def draw(self, overlay: np.ndarray):
        for particle in self.particles:
            fade = max(0.0, min(1.0, particle.life / float(PARTICLE_MAX_LIFE)))
            color = tuple(int(component * fade) for component in particle.color)
            cv2.circle(
                overlay,
                (int(particle.x), int(particle.y)),
                max(1, int(particle.radius * fade) or 1),
                color,
                -1,
                cv2.LINE_AA,
            )


# ---------------------------------------------------------------------------
# Hand state tracker (one per hand)
# ---------------------------------------------------------------------------


class HandState:
    def __init__(self, colors: List[Tuple[int, int, int]]):
        self.colors = colors
        self.color_idx = 0
        self.prev_pos: Optional[Tuple[int, int]] = None
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.drawing = False
        self.last_pinch_time = 0.0
        self.open_palm_start: Optional[float] = None
        self.prev_wrist_pos: Optional[Tuple[float, float]] = None
        self.speed = 0.0
        self.cursor_pos: Optional[Tuple[int, int]] = None
        self.cursor_thickness = BRUSH_MIN_THICKNESS
        self.last_stamp_time = 0.0
        self.stamp_idx = 0

    @property
    def color(self) -> Tuple[int, int, int]:
        return self.colors[self.color_idx]

    def cycle_color(self):
        self.color_idx = (self.color_idx + 1) % len(self.colors)

    def smooth(self, x: int, y: int) -> Tuple[int, int]:
        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x, self.smooth_y = float(x), float(y)
        else:
            self.smooth_x += DRAW_SMOOTHING * (x - self.smooth_x)
            self.smooth_y += DRAW_SMOOTHING * (y - self.smooth_y)
        return int(self.smooth_x), int(self.smooth_y)

    def calc_speed(self, norm_x: float, norm_y: float):
        if self.prev_wrist_pos is not None:
            dx = norm_x - self.prev_wrist_pos[0]
            dy = norm_y - self.prev_wrist_pos[1]
            self.speed = math.hypot(dx, dy)
        self.prev_wrist_pos = (norm_x, norm_y)

    def get_thickness(self) -> int:
        if self.speed < SPEED_SLOW_THRESHOLD:
            return BRUSH_MAX_THICKNESS
        elif self.speed > SPEED_FAST_THRESHOLD:
            return BRUSH_MIN_THICKNESS
        else:
            t = (self.speed - SPEED_SLOW_THRESHOLD) / (
                SPEED_FAST_THRESHOLD - SPEED_SLOW_THRESHOLD
            )
            return int(
                BRUSH_MAX_THICKNESS - t * (BRUSH_MAX_THICKNESS - BRUSH_MIN_THICKNESS)
            )

    def reset_draw(self):
        self.prev_pos = None
        self.smooth_x = None
        self.smooth_y = None
        self.prev_wrist_pos = None
        self.speed = 0.0
        self.open_palm_start = None
        self.drawing = False
        self.cursor_pos = None


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


class AirCanvas:
    def __init__(self):
        download_model()

        self.cap = self._open_camera()
        if self.cap is None:
            raise RuntimeError("Cannot open camera")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera")

        self.camera_thread = CameraThread(self.cap)
        self.camera_thread.start()

        self.frame_h, self.frame_w = frame.shape[:2]
        self.canvas = np.full(
            (self.frame_h, self.frame_w, 3), CANVAS_BG_COLOR, dtype=np.uint8
        )

        self.left_hand = HandState(LEFT_HAND_COLORS)
        self.right_hand = HandState(RIGHT_HAND_COLORS)
        self.sound_engine = SoundEngine(DRAW_SOUND)
        self.rainbow_mode = False
        self.rainbow_hue = 0
        self.particle_system = ParticleSystem(PARTICLE_MAX_COUNT)
        self.particle_overlay = np.zeros_like(self.canvas)

        self._music_last_note_left = 0.0
        self._music_last_note_right = 0.0

        default_idx = (
            THEMES.index(BACKGROUND_THEME) if BACKGROUND_THEME in THEMES else 0
        )
        self.theme_idx = default_idx
        self.themes = THEMES if THEME_ENABLED else ["dark"]
        self.theme_bg_cache: dict = {}

        self.snap_clear_event = threading.Event()
        self._snap_clear_last = 0.0
        self._snap_clear_active = SNAP_CLEAR_ENABLED and _SOUNDDEVICE_AVAILABLE
        if self._snap_clear_active:
            threading.Thread(target=self._mic_listener, daemon=True).start()
        elif SNAP_CLEAR_ENABLED and not _SOUNDDEVICE_AVAILABLE:
            print("⚠️  snap-to-clear disabled: sounddevice not installed")

        # MediaPipe HandLandmarker
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_hand_presence_confidence=TRACKING_CONFIDENCE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.fps = 0.0
        self.fps_timer = time.time()
        self.fps_count = 0

        print("🎨 Air Canvas ready! Show your hands!")
        print("  👆 Point (index finger) = Draw")
        print("  ✊ Fist = Stop drawing (lift brush)")
        print("  🤏 Pinch = Change color")
        print("  ✌️ V-sign = Place sticker")
        print("  🖐️ Open palm (hold 1.5s) = Clear canvas")
        print("  Press 'r' for rainbow, 's' to save, 'c' to clear, 'q' to quit")

    def _current_draw_color(self, state: HandState) -> Tuple[int, int, int]:
        if not (RAINBOW_ENABLED and self.rainbow_mode):
            return state.color
        color = hue_to_bgr(self.rainbow_hue)
        self.rainbow_hue = (self.rainbow_hue + RAINBOW_HUE_STEP) % 180
        return color

    def _clear_canvas(self):
        self.canvas[:] = CANVAS_BG_COLOR
        self.particle_system.particles.clear()
        self.particle_overlay[:] = 0
        self.sound_engine.play("clear")
        print("🧹 Canvas cleared!")

    def _mic_listener(self):
        import numpy as _np

        rec = getattr(_sounddevice, "rec", None)
        if rec is None:
            return

        while True:
            try:
                chunk = rec(
                    SNAP_CLEAR_CHUNK,
                    samplerate=SNAP_CLEAR_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocking=True,
                )
                amplitude = float(_np.abs(chunk).max())
                now = time.time()
                if (
                    amplitude >= SNAP_CLEAR_THRESHOLD
                    and now - self._snap_clear_last >= SNAP_CLEAR_COOLDOWN
                ):
                    self._snap_clear_last = now
                    self.snap_clear_event.set()
            except Exception:
                time.sleep(0.1)

    def _place_stamp(self, state: HandState, center: Tuple[int, int], now: float):
        if not STICKERS_ENABLED or now - state.last_stamp_time < STAMP_COOLDOWN:
            state.cursor_pos = center
            state.cursor_thickness = STAMP_SIZE
            return
        stamp_type = STAMP_SEQUENCE[state.stamp_idx % len(STAMP_SEQUENCE)]
        draw_stamp(self.canvas, center, stamp_type, STAMP_SIZE, state.color)
        state.stamp_idx = (state.stamp_idx + 1) % len(STAMP_SEQUENCE)
        state.last_stamp_time = now
        state.cursor_pos = center
        state.cursor_thickness = STAMP_SIZE
        self.sound_engine.play("stamp")

    def _fade_particle_overlay(self):
        self.particle_overlay = cv2.addWeighted(
            self.particle_overlay,
            PARTICLE_DECAY,
            np.zeros_like(self.particle_overlay),
            1.0 - PARTICLE_DECAY,
            0,
        )

    def _draw_particles(self, display):
        if not PARTICLES_ENABLED:
            return
        self._fade_particle_overlay()
        self.particle_system.update()
        self.particle_system.draw(self.particle_overlay)
        cv2.add(display, self.particle_overlay, dst=display)

    def _open_camera(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FPS, 30)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera FPS requested: 30, actual: {actual_fps:.1f}")
            return cap
        # Fallback: try other indices
        for idx in range(5):
            if idx == CAMERA_INDEX:
                continue
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FPS, 30)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"Camera opened at index {idx}")
                print(f"Camera FPS requested: 30, actual: {actual_fps:.1f}")
                return cap
            cap.release()
        return None

    def _process_hand(self, landmarks, handedness: str, frame):
        is_left = handedness == "Left"
        state = self.right_hand if is_left else self.left_hand
        is_state_left = state is self.left_hand

        now = time.time()
        pos = fingertip_pos(landmarks, self.frame_w, self.frame_h)
        state.calc_speed(landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y)

        if pinch_distance(landmarks) < PINCH_THRESHOLD:
            if now - state.last_pinch_time > PINCH_COOLDOWN:
                state.cycle_color()
                state.last_pinch_time = now
                self.sound_engine.play("color_change")
                state.reset_draw()
            return

        if is_open_palm(landmarks):
            if state.open_palm_start is None:
                state.open_palm_start = now
            elif now - state.open_palm_start >= CLEAR_HOLD_TIME:
                self._clear_canvas()
                state.open_palm_start = None
            state.reset_draw()
            return
        else:
            state.open_palm_start = None

        if is_fist(landmarks):
            state.reset_draw()
            return

        if is_v_sign(landmarks):
            stamp_center = v_sign_center(landmarks, self.frame_w, self.frame_h)
            self._place_stamp(state, stamp_center, now)
            state.prev_pos = None
            state.drawing = False
            return

        if is_pointing(landmarks):
            smooth_pos = state.smooth(pos[0], pos[1])
            thickness = state.get_thickness()
            draw_color = self._current_draw_color(state)

            if not state.drawing:
                self.sound_engine.play("draw_start")

            if state.prev_pos is not None:
                cv2.line(
                    self.canvas,
                    state.prev_pos,
                    smooth_pos,
                    draw_color,
                    thickness,
                    cv2.LINE_AA,
                )

                if BRUSH_GLOW:
                    glow_layer = np.zeros_like(self.canvas)
                    cv2.line(
                        glow_layer,
                        state.prev_pos,
                        smooth_pos,
                        draw_color,
                        thickness + BRUSH_GLOW_RADIUS,
                        cv2.LINE_AA,
                    )
                    glow_layer = cv2.GaussianBlur(
                        glow_layer, (0, 0), BRUSH_GLOW_RADIUS // 2
                    )
                    self.canvas = cv2.add(self.canvas, glow_layer // 3)

                if PARTICLES_ENABLED:
                    self.particle_system.emit(
                        smooth_pos,
                        draw_color,
                        PARTICLE_EMIT_COUNT,
                    )

                if MUSIC_MODE:
                    notes = (
                        PENTATONIC_NOTES_LOW if is_state_left else PENTATONIC_NOTES_HIGH
                    )
                    freq = notes[state.color_idx % len(notes)]
                    last_attr = (
                        "_music_last_note_left"
                        if is_state_left
                        else "_music_last_note_right"
                    )
                    last_t = getattr(self, last_attr)
                    if now - last_t >= MUSIC_DRAW_COOLDOWN:
                        setattr(self, last_attr, now)
                        self.sound_engine.play_tone(freq)

            state.prev_pos = smooth_pos
            state.drawing = True
            state.cursor_pos = smooth_pos
            state.cursor_thickness = thickness
        else:
            state.reset_draw()

    def _draw_cursors(self, display):
        for state in (self.left_hand, self.right_hand):
            if state.cursor_pos is None:
                continue
            cv2.circle(
                display,
                state.cursor_pos,
                state.cursor_thickness // 2 + 4,
                state.color,
                2,
            )
            if AVATARS_ENABLED:
                is_left = state is self.left_hand
                avatar = AVATAR_LEFT if is_left else AVATAR_RIGHT
                pos = (state.cursor_pos[0], state.cursor_pos[1] - 40)
                if avatar == "penguin":
                    draw_penguin(display, pos, AVATAR_SIZE // 2)
                elif avatar == "cat":
                    draw_cat(display, pos, AVATAR_SIZE // 2)

    def _draw_ui(self, display):
        h, w = display.shape[:2]

        # Left hand color indicator
        cv2.circle(display, (40, h - 40), 20, self.left_hand.color, -1)
        cv2.putText(
            display,
            "L",
            (30, h - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Right hand color indicator
        cv2.circle(display, (w - 40, h - 40), 20, self.right_hand.color, -1)
        cv2.putText(
            display,
            "R",
            (w - 50, h - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # FPS
        cv2.putText(
            display,
            f"FPS: {self.fps:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if THEME_ENABLED:
            theme_name = self.themes[self.theme_idx].upper()
            cv2.putText(
                display,
                f"THEME: {theme_name}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        if RAINBOW_ENABLED and self.rainbow_mode:
            cv2.putText(
                display,
                "RAINBOW",
                (w // 2 - 60, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                hue_to_bgr(self.rainbow_hue),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            display,
            "Point=Draw | V=Stamp | Fist=Stop | Pinch=Color | Palm=Clear | r=Rain | b=Theme",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    def run(self):
        if FULLSCREEN:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

        try:
            while True:
                is_new_frame, frame = self.camera_thread.read()
                if frame is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key == QUIT_KEY:
                        break
                    continue

                frame = cv2.flip(frame, 1)  # Mirror

                if is_new_frame:
                    # FPS calculation
                    self.fps_count += 1
                    elapsed = time.time() - self.fps_timer
                    if elapsed >= 1.0:
                        self.fps = self.fps_count / elapsed
                        self.fps_count = 0
                        self.fps_timer = time.time()

                    small_frame = cv2.resize(frame, (640, 480))
                    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts_ms = int(time.monotonic() * 1000)
                    result = self.detector.detect_for_video(mp_image, ts_ms)

                    seen_states = set()
                    if result.hand_landmarks and result.handedness:
                        for landmarks, handedness in zip(
                            result.hand_landmarks, result.handedness
                        ):
                            hand_label = handedness[0].category_name
                            state = (
                                self.right_hand
                                if hand_label == "Left"
                                else self.left_hand
                            )
                            seen_states.add(state)
                            self._process_hand(landmarks, hand_label, frame)

                    for state in (self.left_hand, self.right_hand):
                        if state not in seen_states:
                            state.reset_draw()

                mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
                mask_3ch = cv2.merge([mask, mask, mask])

                current_theme = self.themes[self.theme_idx]
                if current_theme not in self.theme_bg_cache:
                    self.theme_bg_cache[current_theme] = generate_theme_background(
                        current_theme, self.frame_w, self.frame_h
                    )
                display = self.theme_bg_cache[current_theme].copy()

                display = np.where(mask_3ch > 0, self.canvas, display)

                self._draw_particles(display)
                self._draw_cursors(display)
                self._draw_ui(display)
                cv2.imshow(WINDOW_NAME, display)

                if self.snap_clear_event.is_set():
                    self.snap_clear_event.clear()
                    self._clear_canvas()

                key = cv2.waitKey(1) & 0xFF
                if key == QUIT_KEY:
                    break
                elif key == CLEAR_KEY:
                    self._clear_canvas()
                elif key == SAVE_KEY:
                    filename = f"drawing_{int(time.time())}.png"
                    cv2.imwrite(filename, self.canvas)
                    print(f"💾 Saved: {filename}")
                elif key == RAINBOW_KEY and RAINBOW_ENABLED:
                    self.rainbow_mode = not self.rainbow_mode
                elif key == THEME_KEY and THEME_ENABLED:
                    self.theme_idx = (self.theme_idx + 1) % len(self.themes)
                    self.particle_system.particles.clear()
                    self.particle_overlay[:] = 0
        finally:
            self.camera_thread.stop()
            self.detector.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    canvas = AirCanvas()
    canvas.run()
