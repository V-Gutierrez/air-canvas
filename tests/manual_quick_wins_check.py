import os
import sys
import tempfile
from unittest import mock

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import air_canvas


def main():
    canvas = np.zeros((180, 180, 3), dtype=np.uint8)
    overlay = np.zeros_like(canvas)

    star_color = air_canvas.hue_to_bgr(0)
    rainbow_color = air_canvas.hue_to_bgr(90)
    air_canvas.draw_stamp(canvas, (45, 45), "star", 18, star_color)
    air_canvas.draw_stamp(canvas, (90, 45), "heart", 18, rainbow_color)
    air_canvas.draw_stamp(canvas, (135, 45), "circle", 18, (255, 255, 255))
    air_canvas.draw_stamp(canvas, (90, 100), "smiley", 18, (0, 255, 255))

    particles = air_canvas.ParticleSystem(max_particles=100)
    particles.emit((90, 130), (255, 255, 255), count=10)
    for _ in range(8):
        particles.update()
        particles.draw(overlay)

    combined = cv2.add(canvas, overlay)
    output_path = os.path.join(
        tempfile.gettempdir(), "air_canvas_quick_wins_preview.png"
    )
    cv2.imwrite(output_path, combined)

    with mock.patch.object(air_canvas, "resolve_sound_player", return_value=None):
        engine = air_canvas.SoundEngine(enabled=True)

    tone_bytes = engine.tones["draw_start"]

    print(f"preview={output_path}")
    print(f"canvas_sum={int(canvas.sum())}")
    print(f"overlay_sum={int(overlay.sum())}")
    print(f"tone_bytes={len(tone_bytes)}")
    print(f"tone_header={tone_bytes[:4].decode('ascii')}")
    print(f"rainbow_color={rainbow_color}")


if __name__ == "__main__":
    main()
