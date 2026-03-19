#!/usr/bin/env python3
"""Air Canvas — Two-hand finger painting with MediaPipe for kids."""

import cv2
import numpy as np
import math
import time
import os
import threading
import urllib.request
from collections import deque
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
    """Get index finger tip position in pixel coordinates."""
    x = int(landmarks[INDEX_TIP].x * frame_w)
    y = int(landmarks[INDEX_TIP].y * frame_h)
    return x, y


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
        print("  🖐️ Open palm (hold 1.5s) = Clear canvas")
        print("  Press 's' to save, 'c' to clear, 'q' to quit")

    def _open_camera(self):
        """Open built-in camera, skip iPhone Continuity Camera."""
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
        """Process one detected hand."""
        is_left = handedness == "Left"
        # MediaPipe reports handedness from camera POV (mirrored)
        # So "Left" in MediaPipe = Right hand in mirror = user's right hand
        state = self.right_hand if is_left else self.left_hand

        now = time.time()
        pos = fingertip_pos(landmarks, self.frame_w, self.frame_h)
        state.calc_speed(landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y)

        # --- Pinch: cycle color ---
        if pinch_distance(landmarks) < PINCH_THRESHOLD:
            if now - state.last_pinch_time > PINCH_COOLDOWN:
                state.cycle_color()
                state.last_pinch_time = now
                state.reset_draw()
            return

        # --- Open palm: clear canvas ---
        if is_open_palm(landmarks):
            wrist = landmarks[0]
            if state.open_palm_start is None:
                state.open_palm_start = now
            elif now - state.open_palm_start >= CLEAR_HOLD_TIME:
                self.canvas[:] = CANVAS_BG_COLOR
                state.open_palm_start = None
                print("🧹 Canvas cleared!")
            state.reset_draw()
            return
        else:
            state.open_palm_start = None

        # --- Fist: stop drawing ---
        if is_fist(landmarks):
            state.reset_draw()
            return

        # --- Pointing: draw! ---
        if is_pointing(landmarks):
            smooth_pos = state.smooth(pos[0], pos[1])
            thickness = state.get_thickness()

            if state.prev_pos is not None:
                # Draw on canvas
                cv2.line(
                    self.canvas,
                    state.prev_pos,
                    smooth_pos,
                    state.color,
                    thickness,
                    cv2.LINE_AA,
                )

                # Glow effect
                if BRUSH_GLOW:
                    glow_layer = np.zeros_like(self.canvas)
                    cv2.line(
                        glow_layer,
                        state.prev_pos,
                        smooth_pos,
                        state.color,
                        thickness + BRUSH_GLOW_RADIUS,
                        cv2.LINE_AA,
                    )
                    glow_layer = cv2.GaussianBlur(
                        glow_layer, (0, 0), BRUSH_GLOW_RADIUS // 2
                    )
                    self.canvas = cv2.add(self.canvas, glow_layer // 3)

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

    def _draw_ui(self, display):
        """Draw UI overlay."""
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

        # Instructions
        cv2.putText(
            display,
            "Point=Draw | Fist=Stop | Pinch=Color | Palm=Clear",
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

                # Composite: camera feed + canvas overlay
                # Canvas strokes on top of camera (semi-transparent)
                mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
                mask_3ch = cv2.merge([mask, mask, mask])

                display = frame.copy()
                display = np.where(mask_3ch > 0, self.canvas, display)

                self._draw_cursors(display)
                self._draw_ui(display)
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key == QUIT_KEY:
                    break
                elif key == CLEAR_KEY:
                    self.canvas[:] = CANVAS_BG_COLOR
                    print("🧹 Canvas cleared!")
                elif key == SAVE_KEY:
                    filename = f"drawing_{int(time.time())}.png"
                    cv2.imwrite(filename, self.canvas)
                    print(f"💾 Saved: {filename}")
        finally:
            self.camera_thread.stop()
            self.detector.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    canvas = AirCanvas()
    canvas.run()
