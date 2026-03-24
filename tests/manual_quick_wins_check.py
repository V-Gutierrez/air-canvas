import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import air_canvas

cv2 = air_canvas.cv2


def main():
    canvas = np.zeros((180, 180, 3), dtype=np.uint8)
    overlay = np.zeros_like(canvas)

    rainbow_color = air_canvas.hue_to_bgr(90)
    cv2.line(canvas, (35, 45), (145, 45), rainbow_color, 10, cv2.LINE_AA)
    cv2.circle(canvas, (90, 100), 24, (0, 255, 255), -1)

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

    print(f"preview={output_path}")
    print(f"canvas_sum={int(canvas.sum())}")
    print(f"overlay_sum={int(overlay.sum())}")
    print(f"rainbow_color={rainbow_color}")


if __name__ == "__main__":
    main()
