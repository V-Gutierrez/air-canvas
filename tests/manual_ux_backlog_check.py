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
import config


def main():
    canvas = air_canvas.AirCanvas.__new__(air_canvas.AirCanvas)
    canvas.frame_w = 640
    canvas.frame_h = 480
    canvas.stroke_layer = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas.stamp_layer = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas.left_hand = air_canvas.HandState(config.LEFT_HAND_COLORS)
    canvas.right_hand = air_canvas.HandState(config.RIGHT_HAND_COLORS)
    canvas.sound_engine = mock.Mock()
    canvas.rainbow_mode = True
    canvas.rainbow_hue = 30
    canvas.particle_system = mock.Mock()
    canvas.particle_overlay = np.zeros_like(canvas.stroke_layer)
    canvas.last_draw_time = 0.0
    canvas.draw_alive_active = False
    canvas.shape_hunt_active = False
    canvas.save_overlay_until = 9999999999.0
    canvas.save_flash_until = 9999999999.0
    canvas.save_overlay_path = "~/Desktop/air-canvas-art/art-demo.png"
    canvas.save_overlay_thumbnail = np.full(
        (config.SAVE_THUMBNAIL_HEIGHT, config.SAVE_THUMBNAIL_WIDTH, 3),
        180,
        dtype=np.uint8,
    )
    canvas.theme_idx = 0
    canvas.themes = config.THEMES
    canvas.theme_bg_cache = {
        "dark": np.full((480, 640, 3), config.CANVAS_BG_COLOR, dtype=np.uint8),
        "space": air_canvas.generate_theme_background("space", 640, 480),
        "forest": air_canvas.generate_theme_background("forest", 640, 480),
        "ocean": air_canvas.generate_theme_background("ocean", 640, 480),
    }

    canvas.left_hand.cursor_pos = (120, 240)
    canvas.left_hand.cursor_thickness = 24
    canvas.left_hand.palette_hover_idx = 1
    canvas.left_hand.palette_hover_start = 0.0
    canvas.left_hand.open_palm_progress = 0.65
    canvas.left_hand.clear_indicator_pos = (120, 150)

    canvas.right_hand.cursor_pos = (520, 260)
    canvas.right_hand.cursor_thickness = 30
    canvas.right_hand.selected_stamp = "heart"
    canvas.right_hand.stamp_hover_idx = 1

    background = air_canvas.darken_frame(
        np.full((480, 640, 3), 140, dtype=np.uint8), 0.18
    )
    display = air_canvas.compose_art_layers(
        background,
        canvas.stroke_layer,
        canvas._layer_mask(canvas.stroke_layer),
        canvas.stamp_layer,
        canvas._layer_mask(canvas.stamp_layer),
    )
    canvas._draw_cursors(display)
    canvas._draw_ui(display)
    canvas._apply_save_flash(display, 0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "ux_backlog_preview.png")
        cv2.imwrite(output_path, display)
        print(f"preview={output_path}")
        print(f"sum={int(display.sum())}")
        print(f"camera_default={config.BACKGROUND_THEME}")
        print(f"themes={config.THEMES}")
        print(f"palette_dwell={config.PALETTE_DWELL_TIME}")
        print(f"stamp_shelf_height={config.STAMP_SHELF_HEIGHT}")


if __name__ == "__main__":
    main()
