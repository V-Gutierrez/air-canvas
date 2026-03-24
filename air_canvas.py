#!/usr/bin/env python3
"""Air Canvas — Two-hand finger painting with MediaPipe for kids."""

import numpy as np
import math
import time
import os
import random
import threading
import urllib.request
import importlib
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List, Protocol

_cv2_spec = importlib.util.find_spec("cv2")
if _cv2_spec is not None:
    cv2 = importlib.import_module("cv2")
else:
    import colorsys

    class _FallbackVideoCapture:
        def __init__(self, *_args, **_kwargs):
            self._opened = False

        def isOpened(self):
            return self._opened

        def read(self):
            return False, None

        def release(self):
            return None

        def set(self, *_args, **_kwargs):
            return False

        def get(self, *_args, **_kwargs):
            return 0.0

    class _CV2Fallback:
        COLOR_HSV2BGR = 1
        COLOR_BGR2GRAY = 2
        COLOR_BGR2RGB = 3
        THRESH_BINARY = 0
        LINE_AA = 16
        BORDER_CONSTANT = 0
        FONT_HERSHEY_SIMPLEX = 0
        WINDOW_NORMAL = 0
        WND_PROP_FULLSCREEN = 0
        WINDOW_FULLSCREEN = 0
        CAP_PROP_FPS = 5
        INTER_LANCZOS4 = 4
        VideoCapture = _FallbackVideoCapture

        @staticmethod
        def cvtColor(image, code):
            if code == _CV2Fallback.COLOR_HSV2BGR:
                hsv = np.asarray(image, dtype=np.uint8)
                bgr = np.zeros_like(hsv)
                for y in range(hsv.shape[0]):
                    for x in range(hsv.shape[1]):
                        h, s, v = hsv[y, x]
                        r, g, b = colorsys.hsv_to_rgb(
                            float(h) / 179.0,
                            float(s) / 255.0,
                            float(v) / 255.0,
                        )
                        bgr[y, x] = (
                            int(round(b * 255)),
                            int(round(g * 255)),
                            int(round(r * 255)),
                        )
                return bgr
            if code == _CV2Fallback.COLOR_BGR2GRAY:
                if image.ndim == 2:
                    return image.copy()
                blue = image[:, :, 0].astype(np.float32)
                green = image[:, :, 1].astype(np.float32)
                red = image[:, :, 2].astype(np.float32)
                return np.clip(
                    0.114 * blue + 0.587 * green + 0.299 * red, 0, 255
                ).astype(np.uint8)
            if code == _CV2Fallback.COLOR_BGR2RGB:
                return image[:, :, ::-1].copy()
            raise ValueError(f"Unsupported color conversion: {code}")

        @staticmethod
        def threshold(image, thresh, max_value, _threshold_type):
            binary = np.where(image > thresh, max_value, 0).astype(np.uint8)
            return thresh, binary

        @staticmethod
        def merge(channels):
            return np.stack(channels, axis=-1)

        @staticmethod
        def add(src1, src2, dst=None):
            result = np.clip(
                src1.astype(np.int16) + src2.astype(np.int16), 0, 255
            ).astype(np.uint8)
            if dst is not None:
                dst[:] = result
                return dst
            return result

        @staticmethod
        def addWeighted(src1, alpha, src2, beta, gamma, dst=None):
            result = (
                src1.astype(np.float32) * alpha + src2.astype(np.float32) * beta + gamma
            )
            clipped = np.clip(result, 0, 255).astype(np.uint8)
            if dst is not None:
                dst[:] = clipped
                return dst
            return clipped

        @staticmethod
        def resize(image, size, interpolation=None):
            width, height = size
            src_h, src_w = image.shape[:2]
            y_idx = np.linspace(0, src_h - 1, height).astype(int)
            x_idx = np.linspace(0, src_w - 1, width).astype(int)
            return image[np.ix_(y_idx, x_idx)]

        @staticmethod
        def copyMakeBorder(
            image, top, bottom, left, right, _border_type, value=(0, 0, 0)
        ):
            output = np.empty(
                (
                    image.shape[0] + top + bottom,
                    image.shape[1] + left + right,
                    image.shape[2],
                ),
                dtype=image.dtype,
            )
            output[:] = value
            output[top : top + image.shape[0], left : left + image.shape[1]] = image
            return output

        @staticmethod
        def putText(
            image, text, origin, _font, scale, color, thickness, _line_type=None
        ):
            x, y = origin
            height = max(6, int(12 * scale))
            width = max(1, int(len(text) * 7 * scale))
            return _CV2Fallback.rectangle(
                image, (x, y - height), (x + width, y), color, thickness
            )

        @staticmethod
        def getTextSize(text, _font, scale, _thickness):
            height = max(6, int(12 * scale))
            width = max(1, int(len(text) * 7 * scale))
            return (width, height), height

        @staticmethod
        def rectangle(image, pt1, pt2, color, thickness):
            x1, y1 = pt1
            x2, y2 = pt2
            x1, x2 = sorted((int(x1), int(x2)))
            y1, y2 = sorted((int(y1), int(y2)))
            if thickness < 0:
                image[y1 : y2 + 1, x1 : x2 + 1] = color
                return image
            image[y1 : y1 + thickness, x1 : x2 + 1] = color
            image[y2 - thickness + 1 : y2 + 1, x1 : x2 + 1] = color
            image[y1 : y2 + 1, x1 : x1 + thickness] = color
            image[y1 : y2 + 1, x2 - thickness + 1 : x2 + 1] = color
            return image

        @staticmethod
        def circle(image, center, radius, color, thickness, _line_type=None):
            cx, cy = center
            yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
            dist_sq = (xx - int(cx)) ** 2 + (yy - int(cy)) ** 2
            radius_sq = int(radius) ** 2
            if thickness < 0:
                mask = dist_sq <= radius_sq
            else:
                inner = max(0, int(radius) - max(1, int(thickness)))
                mask = (dist_sq <= radius_sq) & (dist_sq >= inner * inner)
            image[mask] = color
            return image

        @staticmethod
        def ellipse(image, center, axes, _angle, _start, _end, color, thickness):
            cx, cy = center
            ax, ay = max(1, int(axes[0])), max(1, int(axes[1]))
            yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
            norm = ((xx - int(cx)) / ax) ** 2 + ((yy - int(cy)) / ay) ** 2
            if thickness < 0:
                mask = norm <= 1.0
            else:
                inner_ax = max(1, ax - max(1, int(thickness)))
                inner_ay = max(1, ay - max(1, int(thickness)))
                inner = ((xx - int(cx)) / inner_ax) ** 2 + (
                    (yy - int(cy)) / inner_ay
                ) ** 2
                mask = (norm <= 1.0) & (inner >= 1.0)
            image[mask] = color
            return image

        @staticmethod
        def _draw_line(image, start, end, color, thickness):
            x1, y1 = start
            x2, y2 = end
            steps = max(abs(int(x2) - int(x1)), abs(int(y2) - int(y1)), 1)
            xs = np.linspace(x1, x2, steps + 1)
            ys = np.linspace(y1, y2, steps + 1)
            radius = max(1, int(math.ceil(thickness / 2)))
            for x, y in zip(xs, ys):
                _CV2Fallback.circle(
                    image, (int(round(x)), int(round(y))), radius, color, -1
                )
            return image

        @staticmethod
        def line(image, pt1, pt2, color, thickness, _line_type=None):
            return _CV2Fallback._draw_line(image, pt1, pt2, color, thickness)

        @staticmethod
        def polylines(image, polygons, is_closed, color, thickness):
            for polygon in polygons:
                points = np.asarray(polygon).reshape(-1, 2)
                for index in range(len(points) - 1):
                    _CV2Fallback._draw_line(
                        image,
                        tuple(points[index]),
                        tuple(points[index + 1]),
                        color,
                        thickness,
                    )
                if is_closed and len(points) > 1:
                    _CV2Fallback._draw_line(
                        image,
                        tuple(points[-1]),
                        tuple(points[0]),
                        color,
                        thickness,
                    )
            return image

        @staticmethod
        def fillPoly(image, polygons, color):
            for polygon in polygons:
                points = np.asarray(polygon).reshape(-1, 2)
                min_x = max(0, int(np.floor(points[:, 0].min())))
                max_x = min(image.shape[1] - 1, int(np.ceil(points[:, 0].max())))
                min_y = max(0, int(np.floor(points[:, 1].min())))
                max_y = min(image.shape[0] - 1, int(np.ceil(points[:, 1].max())))
                for y in range(min_y, max_y + 1):
                    intersections = []
                    for index in range(len(points)):
                        x1, y1 = points[index]
                        x2, y2 = points[(index + 1) % len(points)]
                        if y1 == y2:
                            continue
                        if y < min(y1, y2) or y >= max(y1, y2):
                            continue
                        ratio = (y - y1) / (y2 - y1)
                        intersections.append(x1 + ratio * (x2 - x1))
                    intersections.sort()
                    for start, end in zip(intersections[0::2], intersections[1::2]):
                        image[
                            y,
                            max(min_x, int(math.ceil(start))) : min(
                                max_x, int(math.floor(end))
                            )
                            + 1,
                        ] = color
            return image

        @staticmethod
        def GaussianBlur(image, _ksize, _sigma):
            return image.copy()

        @staticmethod
        def warpAffine(image, matrix, size):
            width, height = size
            output = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
            linear = np.vstack([matrix, [0.0, 0.0, 1.0]])
            inv = np.linalg.inv(linear)
            for y in range(height):
                for x in range(width):
                    src_x, src_y, _ = inv @ np.array([x, y, 1.0], dtype=np.float32)
                    src_x = int(round(src_x))
                    src_y = int(round(src_y))
                    if 0 <= src_x < image.shape[1] and 0 <= src_y < image.shape[0]:
                        output[y, x] = image[src_y, src_x]
            return output

        @staticmethod
        def imwrite(path, image):
            with open(path, "wb") as handle:
                handle.write(b"P6\n")
                handle.write(
                    f"{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
                )
                handle.write(np.asarray(image, dtype=np.uint8)[:, :, ::-1].tobytes())
            return True

        @staticmethod
        def flip(image, mode):
            if mode == 1:
                return image[:, ::-1].copy()
            return image[::-1].copy()

        @staticmethod
        def namedWindow(*_args, **_kwargs):
            return None

        @staticmethod
        def setWindowProperty(*_args, **_kwargs):
            return None

        @staticmethod
        def waitKey(*_args, **_kwargs):
            return -1

        @staticmethod
        def imshow(*_args, **_kwargs):
            return None

        @staticmethod
        def destroyAllWindows():
            return None

    cv2 = _CV2Fallback()

from config import *

# ---------------------------------------------------------------------------
# MediaPipe Task API setup (0.10.33+)
# ---------------------------------------------------------------------------
_mediapipe_spec = importlib.util.find_spec("mediapipe")
if _mediapipe_spec is not None:
    mp = importlib.import_module("mediapipe")
    mp_tasks = importlib.import_module("mediapipe.tasks.python")
    vision = importlib.import_module("mediapipe.tasks.python.vision")
    _MEDIAPIPE_AVAILABLE = True
else:
    mp = None
    mp_tasks = None
    vision = None
    _MEDIAPIPE_AVAILABLE = False


class CaptureLike(Protocol):
    def read(self) -> Tuple[bool, Optional[np.ndarray]]: ...

    def release(self) -> None: ...

    def isOpened(self) -> bool: ...

    def set(self, *_args: object, **_kwargs: object) -> bool: ...

    def get(self, *_args: object, **_kwargs: object) -> float: ...


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Landmark indices
WRIST = 0
INDEX_TIP = 8
INDEX_PIP = 6
THUMB_TIP = 4
THUMB_IP = 3
MIDDLE_TIP = 12
MIDDLE_MCP = 9
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

SHAPE_HUNT_SEQUENCE = ("circle", "triangle", "square", "star")


def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading hand_landmarker model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


class CameraThread:
    def __init__(self, cap: "CaptureLike"):
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


def is_thumb_extended(landmarks, handedness: str) -> bool:
    thumb_delta = landmarks[THUMB_TIP].x - landmarks[THUMB_IP].x
    if handedness == "Left":
        return thumb_delta < -THUMB_EXTENSION_THRESHOLD
    return thumb_delta > THUMB_EXTENSION_THRESHOLD


def is_palm_facing_camera(landmarks) -> bool:
    """Palm faces camera when middle MCP is closer (smaller z) than wrist."""
    return landmarks[MIDDLE_MCP].z < landmarks[WRIST].z


def is_open_palm(landmarks, handedness: str) -> bool:
    """All 5 fingers extended and palm facing the camera."""
    return (
        is_thumb_extended(landmarks, handedness)
        and is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
        and is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and is_finger_extended(landmarks, RING_TIP, RING_PIP)
        and is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP)
        and is_palm_facing_camera(landmarks)
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


def is_intentional_pinch(landmarks) -> bool:
    return (
        pinch_distance(landmarks) < PINCH_THRESHOLD
        and not is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not is_finger_extended(landmarks, RING_TIP, RING_PIP)
        and not is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP)
    )


def fingertip_pos(landmarks, frame_w, frame_h) -> Tuple[int, int]:
    x = int(landmarks[INDEX_TIP].x * frame_w)
    y = int(landmarks[INDEX_TIP].y * frame_h)
    return x, y


def hue_to_bgr(hue: int) -> Tuple[int, int, int]:
    hsv = np.array([[[int(hue) % 180, 255, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


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


def compute_draw_alive_transform(
    now: float, last_draw_time: float, delay: float
) -> Tuple[int, int, float]:
    idle_time = now - last_draw_time
    if idle_time < delay:
        return 0, 0, 1.0
    phase_time = idle_time - delay
    phase = 2 * math.pi * DRAW_ALIVE_FREQUENCY * phase_time
    shift_x = int(round(DRAW_ALIVE_SHIFT_PX * math.sin(phase)))
    shift_y = int(round(DRAW_ALIVE_SHIFT_PX * math.cos(phase * 0.7)))
    scale = 1.0 + DRAW_ALIVE_BREATHE_SCALE * math.sin(phase * 0.5)
    return shift_x, shift_y, scale


def compute_mask_coverage(
    target_mask: np.ndarray, user_delta_mask: np.ndarray
) -> float:
    target_pixels = int(np.count_nonzero(target_mask))
    if target_pixels == 0:
        return 0.0
    overlap_pixels = int(np.count_nonzero((target_mask > 0) & (user_delta_mask > 0)))
    return overlap_pixels / float(target_pixels)


def next_shape_hunt_size(current_size: int, min_size: int = SHAPE_HUNT_MIN_SIZE) -> int:
    return max(min_size, current_size - SHAPE_HUNT_SHRINK_STEP)


def compose_art_layers(
    background: np.ndarray,
    stroke_layer: np.ndarray,
    stroke_mask: np.ndarray,
) -> np.ndarray:
    composite = background.copy()
    stroke_mask_3ch = cv2.merge([stroke_mask, stroke_mask, stroke_mask])
    composite = np.where(stroke_mask_3ch > 0, stroke_layer, composite)
    return composite


def darken_frame(frame: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return frame.copy()
    return cv2.addWeighted(
        frame,
        max(0.0, 1.0 - alpha),
        np.zeros_like(frame),
        min(1.0, alpha),
        0,
    )


def build_export_filepath(export_dir: str, dt: datetime, prefix: str = "art") -> str:
    return os.path.join(export_dir, dt.strftime(f"{prefix}-%Y-%m-%d-%H%M%S.png"))


def display_path(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        return path.replace(home, "~", 1)
    return path


def build_save_thumbnail(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (SAVE_THUMBNAIL_WIDTH, SAVE_THUMBNAIL_HEIGHT))


def create_print_ready_image(art: np.ndarray, date_label: str) -> np.ndarray:
    height, width = art.shape[:2]
    upscaled = cv2.resize(
        art,
        (width * EXPORT_UPSCALE, height * EXPORT_UPSCALE),
        interpolation=cv2.INTER_LANCZOS4,
    )
    border_size = 20
    framed = cv2.copyMakeBorder(
        upscaled,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=(0, 200, 255),
    )
    cv2.putText(
        framed,
        date_label,
        (border_size, framed.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return framed


def draw_ghost_shape(
    canvas: np.ndarray,
    shape_name: str,
    center: Tuple[int, int],
    size: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    cx, cy = center
    half = max(4, size // 2)
    if shape_name == "circle":
        cv2.circle(canvas, center, half, color, thickness, cv2.LINE_AA)
        return
    if shape_name == "triangle":
        points = np.array(
            [[cx, cy - half], [cx - half, cy + half], [cx + half, cy + half]],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [points.reshape((-1, 1, 2))], True, color, thickness)
        return
    if shape_name == "square":
        cv2.rectangle(
            canvas, (cx - half, cy - half), (cx + half, cy + half), color, thickness
        )
        return
    outer = half
    inner = max(2, int(outer * 0.45))
    points = []
    for index in range(10):
        angle = (math.pi / 5 * index) - (math.pi / 2)
        radius = outer if index % 2 == 0 else inner
        points.append(
            [
                int(cx + math.cos(angle) * radius),
                int(cy + math.sin(angle) * radius),
            ]
        )
    cv2.polylines(
        canvas,
        [np.array(points, dtype=np.int32).reshape((-1, 1, 2))],
        True,
        color,
        thickness,
    )


def generate_target_mask(
    shape_name: str,
    center: Tuple[int, int],
    size: int,
    canvas_shape: Tuple[int, ...],
) -> np.ndarray:
    mask = np.zeros(canvas_shape[:2], dtype=np.uint8)
    cx, cy = center
    half = max(4, size // 2)
    if shape_name == "circle":
        cv2.circle(mask, center, half, 255, -1)
        return mask
    if shape_name == "triangle":
        points = np.array(
            [[cx, cy - half], [cx - half, cy + half], [cx + half, cy + half]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], 255)
        return mask
    if shape_name == "square":
        cv2.rectangle(mask, (cx - half, cy - half), (cx + half, cy + half), 255, -1)
        return mask
    outer = half
    inner = max(2, int(outer * 0.45))
    points = []
    for index in range(10):
        angle = (math.pi / 5 * index) - (math.pi / 2)
        radius = outer if index % 2 == 0 else inner
        points.append(
            [
                int(cx + math.cos(angle) * radius),
                int(cy + math.sin(angle) * radius),
            ]
        )
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32).reshape((-1, 1, 2))], 255)
    return mask


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
        self.pinch_active = False
        self.open_palm_start: Optional[float] = None
        self.prev_wrist_pos: Optional[Tuple[float, float]] = None
        self.speed = 0.0
        self.cursor_pos: Optional[Tuple[int, int]] = None
        self.cursor_thickness = BRUSH_MIN_THICKNESS
        self.palette_hover_idx: Optional[int] = None
        self.palette_hover_start: Optional[float] = None
        self.palette_pop_until = 0.0
        self.open_palm_progress = 0.0
        self.clear_indicator_pos: Optional[Tuple[int, int]] = None

    @property
    def color(self) -> Tuple[int, int, int]:
        return self.colors[self.color_idx]

    def cycle_color(self):
        self.color_idx = (self.color_idx + 1) % len(self.colors)

    def set_color_idx(self, idx: int, now: float):
        self.color_idx = idx % len(self.colors)
        self.palette_pop_until = now + 0.18

    def reset_open_palm(self):
        self.open_palm_start = None
        self.open_palm_progress = 0.0
        self.clear_indicator_pos = None

    def reset_palette_hover(self):
        self.palette_hover_idx = None
        self.palette_hover_start = None

    def reset_stroke(self):
        self.prev_pos = None
        self.smooth_x = None
        self.smooth_y = None
        self.prev_wrist_pos = None
        self.speed = 0.0
        self.drawing = False
        self.cursor_pos = None

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
        self.reset_stroke()
        self.reset_open_palm()
        self.reset_palette_hover()


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


class AirCanvas:
    def __init__(self):
        if not _MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe is required to run Air Canvas")

        download_model()

        self.cap = self._open_camera()
        if self.cap is None:
            raise RuntimeError("Cannot open camera")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera")
        if frame is None:
            raise RuntimeError("Camera returned an empty frame")

        self.camera_thread = CameraThread(self.cap)
        self.camera_thread.start()

        self.frame_h, self.frame_w = frame.shape[:2]
        self.stroke_layer = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        self.left_hand = HandState(LEFT_HAND_COLORS)
        self.right_hand = HandState(RIGHT_HAND_COLORS)
        self.rainbow_mode = False
        self.rainbow_hue = 0
        self.particle_system = ParticleSystem(PARTICLE_MAX_COUNT)
        self.particle_overlay = np.zeros_like(self.stroke_layer)
        self.last_draw_time = time.time()
        self.draw_alive_active = DRAW_ALIVE_ENABLED
        self.shape_hunt_active = False
        self.shape_hunt_shape_idx = 0
        self.shape_hunt_shape_name = SHAPE_HUNT_SEQUENCE[0]
        self.shape_hunt_size = SHAPE_HUNT_START_SIZE
        self.shape_hunt_center = (self.frame_w // 2, self.frame_h // 2)
        self.shape_hunt_snapshot: Optional[np.ndarray] = None
        self.shape_hunt_target_mask = np.zeros(
            (self.frame_h, self.frame_w), dtype=np.uint8
        )
        self.save_overlay_until = 0.0
        self.save_flash_until = 0.0
        self.save_overlay_path = ""
        self.save_overlay_thumbnail: Optional[np.ndarray] = None

        default_idx = (
            THEMES.index(BACKGROUND_THEME) if BACKGROUND_THEME in THEMES else 0
        )
        self.theme_idx = default_idx
        self.themes = THEMES if THEME_ENABLED else ["dark"]
        self.theme_bg_cache: dict = {}

        # MediaPipe HandLandmarker
        if vision is None or mp_tasks is None:
            raise RuntimeError("mediapipe task runtime is unavailable")
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
        print(
            "  Press 'r' rainbow | 'a' draw alive | 'h' shape hunt | 's' save | 'p' print export | 'c' clear | 'q' quit"
        )

    def _palette_positions(self, state: HandState) -> List[Tuple[int, int]]:
        x = (
            PALETTE_EDGE_MARGIN + PALETTE_CIRCLE_RADIUS
            if state is self.left_hand
            else self.frame_w - PALETTE_EDGE_MARGIN - PALETTE_CIRCLE_RADIUS
        )
        total_height = (
            len(state.colors) * (PALETTE_CIRCLE_RADIUS * 2)
            + (len(state.colors) - 1) * PALETTE_VERTICAL_GAP
        )
        start_y = max(50, (self.frame_h - total_height) // 2)
        spacing = PALETTE_CIRCLE_RADIUS * 2 + PALETTE_VERTICAL_GAP
        return [(x, start_y + index * spacing) for index in range(len(state.colors))]

    def _detect_palette_hover(
        self, state: HandState, pos: Tuple[int, int]
    ) -> Optional[int]:
        for idx, center in enumerate(self._palette_positions(state)):
            if (
                math.hypot(pos[0] - center[0], pos[1] - center[1])
                <= PALETTE_CIRCLE_RADIUS * 1.45
            ):
                return idx
        return None

    def _update_palette_hover(
        self, state: HandState, pos: Tuple[int, int], now: float
    ) -> bool:
        hover_idx = self._detect_palette_hover(state, pos)
        if hover_idx is None:
            state.reset_palette_hover()
            return False
        if state.palette_hover_idx != hover_idx:
            state.palette_hover_idx = hover_idx
            state.palette_hover_start = now
            return True
        if (
            state.palette_hover_start is not None
            and now - state.palette_hover_start >= PALETTE_DWELL_TIME
        ):
            state.palette_hover_start = now
        return True

    def _save_art(self, include_frame: bool = False):
        os.makedirs(EXPORT_DIR, exist_ok=True)
        now_dt = datetime.now()
        filepath = build_export_filepath(EXPORT_DIR, now_dt)
        art = self._compose_current_art()
        image = (
            create_print_ready_image(art, now_dt.strftime("%Y-%m-%d"))
            if include_frame
            else art
        )
        cv2.imwrite(filepath, image)
        self.save_overlay_until = time.time() + SAVE_OVERLAY_DURATION
        self.save_flash_until = time.time() + SAVE_FLASH_DURATION
        self.save_overlay_path = display_path(filepath)
        self.save_overlay_thumbnail = build_save_thumbnail(art)
        print(f"💾 Saved art: {filepath}")
        return filepath

    @property
    def canvas(self) -> np.ndarray:
        return self._compose_current_art()

    @canvas.setter
    def canvas(self, value: np.ndarray):
        self.stroke_layer = value.copy()

    def _compose_current_art(self) -> np.ndarray:
        background = np.full(
            (self.frame_h, self.frame_w, 3), CANVAS_BG_COLOR, dtype=np.uint8
        )
        stroke_mask = self._layer_mask(self.stroke_layer)
        return compose_art_layers(
            background,
            self.stroke_layer,
            stroke_mask,
        )

    def _layer_mask(self, layer: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        return mask

    def _art_since_snapshot(self) -> np.ndarray:
        current_art = self._compose_current_art()
        if self.shape_hunt_snapshot is None:
            return current_art
        current_gray = cv2.cvtColor(current_art, cv2.COLOR_BGR2GRAY)
        snapshot_gray = cv2.cvtColor(self.shape_hunt_snapshot, cv2.COLOR_BGR2GRAY)
        delta = np.abs(
            current_gray.astype(np.int16) - snapshot_gray.astype(np.int16)
        ).astype(np.uint8)
        _, delta_mask = cv2.threshold(delta, 5, 255, cv2.THRESH_BINARY)
        return delta_mask

    def _start_shape_hunt(self):
        self.shape_hunt_active = True
        margin = max(self.shape_hunt_size // 2 + 20, 30)
        max_x = max(margin, self.frame_w - margin)
        max_y = max(margin, self.frame_h - margin)
        center_x = random.randint(margin, max_x)
        center_y = random.randint(margin, max_y)
        self.shape_hunt_shape_name = SHAPE_HUNT_SEQUENCE[
            self.shape_hunt_shape_idx % len(SHAPE_HUNT_SEQUENCE)
        ]
        self.shape_hunt_center = (center_x, center_y)
        self.shape_hunt_snapshot = self._compose_current_art().copy()
        self.shape_hunt_target_mask = generate_target_mask(
            self.shape_hunt_shape_name,
            self.shape_hunt_center,
            self.shape_hunt_size,
            self.shape_hunt_snapshot.shape,
        )

    def _evaluate_shape_hunt_progress(self) -> float:
        if not self.shape_hunt_active:
            return 0.0
        delta_mask = self._art_since_snapshot()
        coverage = compute_mask_coverage(self.shape_hunt_target_mask, delta_mask)
        if coverage >= SHAPE_HUNT_SUCCESS_COVERAGE:
            self.particle_system.emit(
                self.shape_hunt_center,
                (0, 220, 255),
                PARTICLE_EMIT_COUNT * 4,
            )
            self.shape_hunt_shape_idx = (self.shape_hunt_shape_idx + 1) % len(
                SHAPE_HUNT_SEQUENCE
            )
            self.shape_hunt_size = next_shape_hunt_size(self.shape_hunt_size)
            self._start_shape_hunt()
        return coverage

    def _draw_shape_hunt_overlay(self, display: np.ndarray):
        if not self.shape_hunt_active:
            return
        ghost_layer = np.zeros_like(display)
        draw_ghost_shape(
            ghost_layer,
            self.shape_hunt_shape_name,
            self.shape_hunt_center,
            self.shape_hunt_size,
            (120, 120, 120),
            SHAPE_HUNT_TARGET_THICKNESS,
        )
        cv2.addWeighted(display, 1.0, ghost_layer, 0.35, 0, display)
        coverage = self._evaluate_shape_hunt_progress()
        cv2.putText(
            display,
            f"HUNT {int(coverage * 100)}%",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def _visible_stroke_layer(self, now: float) -> np.ndarray:
        if not self.draw_alive_active:
            return self.stroke_layer
        shift_x, shift_y, scale = compute_draw_alive_transform(
            now, self.last_draw_time, DRAW_ALIVE_DELAY
        )
        if shift_x == 0 and shift_y == 0 and abs(scale - 1.0) < 1e-6:
            return self.stroke_layer
        tx = shift_x + (1.0 - scale) * (self.frame_w / 2.0)
        ty = shift_y + (1.0 - scale) * (self.frame_h / 2.0)
        transform = np.array(
            [[scale, 0.0, tx], [0.0, scale, ty]],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            self.stroke_layer, transform, (self.frame_w, self.frame_h)
        )

    def _export_print(self):
        return self._save_art(include_frame=True)

    def _current_draw_color(self, state: HandState) -> Tuple[int, int, int]:
        if not (RAINBOW_ENABLED and self.rainbow_mode):
            return state.color
        color = hue_to_bgr(self.rainbow_hue)
        self.rainbow_hue = (self.rainbow_hue + RAINBOW_HUE_STEP) % 180
        return color

    def _clear_canvas(self):
        self.stroke_layer[:] = 0
        self.shape_hunt_snapshot = None
        self.shape_hunt_target_mask[:] = 0
        self.particle_system.particles.clear()
        self.particle_overlay[:] = 0
        print("🧹 Canvas cleared!")

    def _draw_clear_progress(self, display: np.ndarray, state: HandState):
        if state.clear_indicator_pos is None or state.open_palm_progress <= 0:
            return
        radius = 42
        cv2.circle(display, state.clear_indicator_pos, radius, (255, 255, 255), 2)
        cv2.ellipse(
            display,
            state.clear_indicator_pos,
            (radius, radius),
            -90,
            0,
            int(360 * min(1.0, state.open_palm_progress)),
            (120, 240, 255),
            6,
        )

    def _draw_color_blob(
        self,
        display: np.ndarray,
        center: Tuple[int, int],
        color: Tuple[int, int, int],
        pulse: float,
    ):
        outer = max(10, int(HAND_COLOR_BLOB_RADIUS + pulse * 5))
        cv2.circle(display, center, outer, tuple(min(255, c + 35) for c in color), -1)
        cv2.circle(display, center, max(8, outer - 16), color, -1)
        cv2.circle(display, center, outer, (255, 255, 255), 3)

    def _draw_palette(self, display: np.ndarray, state: HandState, now: float):
        for idx, center in enumerate(self._palette_positions(state)):
            color = state.colors[idx]
            radius = PALETTE_CIRCLE_RADIUS
            if state.palette_hover_idx == idx:
                radius = int(radius * 1.18)
            if state.color_idx == idx and state.palette_pop_until > now:
                radius = int(radius * 1.35)
            cv2.circle(display, center, radius + 8, (255, 255, 255), 2)
            cv2.circle(display, center, radius, color, -1)
            if state.color_idx == idx:
                cv2.circle(display, center, radius + 13, (120, 240, 255), 4)
            if state.palette_hover_idx == idx and state.palette_hover_start is not None:
                dwell = min(1.0, (now - state.palette_hover_start) / PALETTE_DWELL_TIME)
                cv2.ellipse(
                    display,
                    center,
                    (radius + 18, radius + 18),
                    -90,
                    0,
                    int(360 * dwell),
                    (255, 255, 255),
                    4,
                )

    def _draw_rainbow_arc(self, display: np.ndarray):
        if not self.rainbow_mode:
            return
        center = (self.frame_w // 2, 0)
        radius = max(120, self.frame_w // 5)
        for step in range(12):
            start = 180 + step * 15
            end = start + 16
            cv2.ellipse(
                display,
                center,
                (radius, radius),
                0,
                start,
                end,
                hue_to_bgr((self.rainbow_hue + step * 15) % 180),
                RAINBOW_ARC_THICKNESS,
            )

    def _draw_save_overlay(self, display: np.ndarray, now: float):
        if now >= self.save_overlay_until:
            return
        overlay = display.copy()
        panel_width = 460
        panel_height = 170
        x1 = max(20, self.frame_w - panel_width - 30)
        y1 = max(20, self.frame_h - panel_height - 30)
        x2 = x1 + panel_width
        y2 = y1 + panel_height
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (35, 35, 45), -1)
        cv2.addWeighted(display, 1.0, overlay, 0.72, 0, display)
        if self.save_overlay_thumbnail is not None:
            thumb_h, thumb_w = self.save_overlay_thumbnail.shape[:2]
            display[y1 + 20 : y1 + 20 + thumb_h, x1 + 20 : x1 + 20 + thumb_w] = (
                self.save_overlay_thumbnail
            )
            cv2.rectangle(
                display,
                (x1 + 20, y1 + 20),
                (x1 + 20 + thumb_w, y1 + 20 + thumb_h),
                (255, 255, 255),
                2,
            )
        cv2.putText(
            display,
            "Saved! ✨",
            (x1 + 205, y1 + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (120, 255, 180),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            self.save_overlay_path,
            (x1 + 205, y1 + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    def _apply_save_flash(self, display: np.ndarray, now: float):
        if now >= self.save_flash_until:
            return
        remaining = max(0.0, self.save_flash_until - now)
        alpha = min(0.9, remaining / SAVE_FLASH_DURATION)
        flash = np.full_like(display, 255)
        cv2.addWeighted(display, 1.0 - alpha, flash, alpha, 0, display)

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

        now = time.time()
        pos = fingertip_pos(landmarks, self.frame_w, self.frame_h)
        palm_pos = (
            int(landmarks[WRIST].x * self.frame_w),
            int(landmarks[WRIST].y * self.frame_h),
        )
        state.calc_speed(landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y)
        state.cursor_pos = pos
        state.clear_indicator_pos = palm_pos

        hovering_palette = self._update_palette_hover(state, pos, now)

        if is_open_palm(landmarks, handedness):
            state.pinch_active = False
            if state.speed < CLEAR_STILLNESS:
                if state.open_palm_start is None:
                    state.open_palm_start = now
                state.open_palm_progress = min(
                    1.0, (now - state.open_palm_start) / CLEAR_HOLD_TIME
                )
                if state.open_palm_progress >= 1.0:
                    self._clear_canvas()
                    state.reset_draw()
                    return
            else:
                state.reset_open_palm()
            state.reset_stroke()
            return
        state.reset_open_palm()

        if is_fist(landmarks):
            state.pinch_active = False
            state.reset_stroke()
            return

        if hovering_palette:
            state.pinch_active = False
            state.reset_stroke()
            return

        if is_intentional_pinch(landmarks):
            if not state.pinch_active:
                state.pinch_active = True
                state.cycle_color()
                state.palette_pop_until = now + 0.18
            state.reset_stroke()
            return
        state.pinch_active = False

        if is_pointing(landmarks):
            smooth_pos = state.smooth(pos[0], pos[1])
            thickness = state.get_thickness()
            draw_color = self._current_draw_color(state)

            if state.prev_pos is not None:
                cv2.line(
                    self.stroke_layer,
                    state.prev_pos,
                    smooth_pos,
                    draw_color,
                    thickness,
                    cv2.LINE_AA,
                )
                self.last_draw_time = now

                if BRUSH_GLOW:
                    glow_layer = np.zeros_like(self.stroke_layer)
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
                    self.stroke_layer = cv2.add(self.stroke_layer, glow_layer // 3)

                if PARTICLES_ENABLED:
                    self.particle_system.emit(
                        smooth_pos,
                        draw_color,
                        PARTICLE_EMIT_COUNT,
                    )

            state.prev_pos = smooth_pos
            state.drawing = True
            state.cursor_pos = smooth_pos
            state.cursor_thickness = thickness
        else:
            state.reset_stroke()

    def _draw_cursors(self, display):
        now = time.time()
        for state in (self.left_hand, self.right_hand):
            if state.cursor_pos is None:
                continue
            cursor_overlay = display.copy()
            cv2.circle(
                cursor_overlay,
                state.cursor_pos,
                state.cursor_thickness // 2 + 4,
                state.color,
                -1,
            )
            cv2.addWeighted(
                display,
                1.0 - CURSOR_FILL_ALPHA,
                cursor_overlay,
                CURSOR_FILL_ALPHA,
                0,
                display,
            )
            cv2.circle(
                display,
                state.cursor_pos,
                state.cursor_thickness // 2 + 4,
                (255, 255, 255),
                2,
            )
            self._draw_clear_progress(display, state)
            if AVATARS_ENABLED:
                is_left = state is self.left_hand
                avatar = AVATAR_LEFT if is_left else AVATAR_RIGHT
                bob = int(
                    math.sin(now * AVATAR_BOB_SPEED + (0 if is_left else math.pi))
                    * AVATAR_BOB_AMPLITUDE
                )
                pos = (state.cursor_pos[0], state.cursor_pos[1] - 70 + bob)
                if avatar == "penguin":
                    draw_penguin(display, pos, AVATAR_SIZE // 2)
                elif avatar == "cat":
                    draw_cat(display, pos, AVATAR_SIZE // 2)

    def _draw_gesture_overlay(self, display: np.ndarray):
        h, w = display.shape[:2]
        lines = [
            "POINT Draw | PINCH Color | PALM Clear 1.5s",
            "FIST Pause | S Save",
        ]
        font_scale = 0.42
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        if hasattr(cv2, "getTextSize"):
            text_metrics = [
                cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines
            ]
            text_w = max(metric[0] for metric in text_metrics)
            text_h = max(metric[1] for metric in text_metrics)
        else:
            text_w = max(int(len(line) * 11 * font_scale) for line in lines)
            text_h = int(22 * font_scale)

        margin_x = 14
        margin_y = 10
        line_gap = 10
        box_w = text_w + margin_x * 2
        box_h = len(lines) * text_h + (len(lines) - 1) * line_gap + margin_y * 2

        x = (w - box_w) // 2
        y_box_bottom = h - 14
        y_box_top = y_box_bottom - box_h

        overlay = display.copy()
        cv2.rectangle(
            overlay,
            (x, y_box_top),
            (x + box_w, y_box_bottom),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(display, 1.0, overlay, 0.6, 0, display)

        for index, line in enumerate(lines):
            y_text = y_box_top + margin_y + text_h + index * (text_h + line_gap)
            cv2.putText(
                display,
                line,
                (x + margin_x, y_text),
                font,
                font_scale,
                (200, 200, 200),
                thickness,
                cv2.LINE_AA,
            )

    def _draw_ui(self, display):
        h, w = display.shape[:2]
        now = time.time()
        self._draw_palette(display, self.left_hand, now)
        self._draw_palette(display, self.right_hand, now)
        pulse = math.sin(now * 2.4)
        self._draw_color_blob(display, (80, h - 85), self.left_hand.color, pulse)
        self._draw_color_blob(display, (w - 80, h - 85), self.right_hand.color, -pulse)
        self._draw_rainbow_arc(display)
        self._draw_save_overlay(display, now)
        self._draw_gesture_overlay(display)
        if DEBUG:
            cv2.putText(
                display,
                f"FPS: {self.fps:.0f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display,
                f"THEME: {self.themes[self.theme_idx].upper()}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
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
                    mp_module = mp
                    if mp_module is None:
                        raise RuntimeError(
                            "mediapipe became unavailable during runtime"
                        )
                    mp_image = mp_module.Image(
                        image_format=mp_module.ImageFormat.SRGB,
                        data=rgb,
                    )
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

                current_theme = self.themes[self.theme_idx]
                if (
                    current_theme != "camera"
                    and current_theme not in self.theme_bg_cache
                ):
                    self.theme_bg_cache[current_theme] = generate_theme_background(
                        current_theme, self.frame_w, self.frame_h
                    )
                visible_strokes = self._visible_stroke_layer(time.time())
                stroke_mask = self._layer_mask(visible_strokes)
                background = (
                    darken_frame(frame, CAMERA_BG_DARKEN_ALPHA)
                    if current_theme == "camera"
                    else self.theme_bg_cache[current_theme]
                )
                display = compose_art_layers(
                    background,
                    visible_strokes,
                    stroke_mask,
                )

                self._draw_particles(display)
                self._draw_shape_hunt_overlay(display)
                self._draw_cursors(display)
                self._draw_ui(display)
                self._apply_save_flash(display, time.time())
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key == QUIT_KEY:
                    break
                elif key == CLEAR_KEY:
                    self._clear_canvas()
                elif key == SAVE_KEY:
                    self._save_art(include_frame=False)
                elif key == EXPORT_KEY:
                    self._export_print()
                elif key == DRAW_ALIVE_KEY and DRAW_ALIVE_ENABLED:
                    self.draw_alive_active = not self.draw_alive_active
                elif key == SHAPE_HUNT_KEY and SHAPE_HUNT_ENABLED:
                    self.shape_hunt_active = not self.shape_hunt_active
                    if self.shape_hunt_active:
                        self._start_shape_hunt()
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
