import os
import tempfile
import time
import unittest
from datetime import datetime
from unittest import mock

import numpy as np

import air_canvas
import config

cv2 = air_canvas.cv2


class DrawAliveTests(unittest.TestCase):
    def test_draw_alive_transform_waits_for_idle_delay(self):
        shift_x, shift_y, scale = air_canvas.compute_draw_alive_transform(
            now=5.0,
            last_draw_time=2.5,
            delay=3.0,
        )

        self.assertEqual((shift_x, shift_y), (0, 0))
        self.assertEqual(scale, 1.0)

    def test_draw_alive_transform_stays_within_gentle_bounds(self):
        for probe in (10.25, 10.75, 11.5, 13.0):
            shift_x, shift_y, scale = air_canvas.compute_draw_alive_transform(
                now=probe,
                last_draw_time=0.0,
                delay=3.0,
            )

            self.assertLessEqual(abs(shift_x), config.DRAW_ALIVE_SHIFT_PX)
            self.assertLessEqual(abs(shift_y), config.DRAW_ALIVE_SHIFT_PX)
            self.assertGreater(scale, 0.98)
            self.assertLess(scale, 1.02)


class ShapeHuntTests(unittest.TestCase):
    def test_shape_hunt_coverage_uses_target_overlap_ratio(self):
        target_mask = np.full((10, 10), 255, dtype=np.uint8)
        user_delta_mask = np.zeros((10, 10), dtype=np.uint8)
        user_delta_mask[:, :6] = 255

        coverage = air_canvas.compute_mask_coverage(target_mask, user_delta_mask)

        self.assertAlmostEqual(coverage, 0.6, places=2)

    def test_shape_hunt_size_shrinks_without_going_below_minimum(self):
        self.assertLess(air_canvas.next_shape_hunt_size(120), 120)
        self.assertEqual(air_canvas.next_shape_hunt_size(40, min_size=48), 48)

    def test_start_shape_hunt_captures_snapshot_and_target(self):
        canvas = air_canvas.AirCanvas.__new__(air_canvas.AirCanvas)
        canvas.frame_w = 320
        canvas.frame_h = 240
        canvas.canvas = np.zeros((240, 320, 3), dtype=np.uint8)
        canvas.shape_hunt_shape_idx = 0
        canvas.shape_hunt_size = config.SHAPE_HUNT_START_SIZE

        with mock.patch.object(air_canvas.random, "randint", side_effect=[120, 100]):
            canvas._start_shape_hunt()

        self.assertTrue(canvas.shape_hunt_active)
        self.assertEqual(
            canvas.shape_hunt_shape_name, air_canvas.SHAPE_HUNT_SEQUENCE[0]
        )
        self.assertEqual(canvas.shape_hunt_center, (120, 100))
        self.assertEqual(canvas.shape_hunt_target_mask.shape, (240, 320))
        self.assertIsNotNone(canvas.shape_hunt_snapshot)

    def test_shape_hunt_success_advances_shape_and_emits_feedback(self):
        canvas = air_canvas.AirCanvas.__new__(air_canvas.AirCanvas)
        canvas.frame_w = 160
        canvas.frame_h = 120
        canvas.canvas = np.zeros((120, 160, 3), dtype=np.uint8)
        canvas.shape_hunt_active = True
        canvas.shape_hunt_shape_idx = 0
        canvas.shape_hunt_shape_name = "circle"
        canvas.shape_hunt_size = config.SHAPE_HUNT_START_SIZE
        canvas.shape_hunt_center = (80, 60)
        canvas.shape_hunt_snapshot = np.zeros((120, 160, 3), dtype=np.uint8)
        canvas.shape_hunt_target_mask = np.full((120, 160), 255, dtype=np.uint8)
        canvas.particle_system = mock.Mock()

        canvas.canvas[:] = 255

        with mock.patch.object(canvas, "_start_shape_hunt") as restart:
            coverage = canvas._evaluate_shape_hunt_progress()

        self.assertGreaterEqual(coverage, config.SHAPE_HUNT_SUCCESS_COVERAGE)
        canvas.particle_system.emit.assert_called_once()
        self.assertEqual(canvas.shape_hunt_shape_idx, 1)
        self.assertLess(canvas.shape_hunt_size, config.SHAPE_HUNT_START_SIZE)
        restart.assert_called_once()


class PrintExportTests(unittest.TestCase):
    def test_build_export_filepath_uses_print_ready_naming_contract(self):
        filepath = air_canvas.build_export_filepath(
            "/tmp/air-canvas-art",
            datetime(2026, 3, 21, 14, 5, 6),
        )

        self.assertEqual(filepath, "/tmp/air-canvas-art/art-2026-03-21-140506.png")

    def test_create_print_ready_image_upscales_and_adds_frame(self):
        art = np.full((20, 30, 3), 180, dtype=np.uint8)

        export = air_canvas.create_print_ready_image(art, "2026-03-21")

        self.assertGreater(export.shape[0], art.shape[0] * config.EXPORT_UPSCALE)
        self.assertGreater(export.shape[1], art.shape[1] * config.EXPORT_UPSCALE)
        self.assertNotEqual(tuple(export[0, 0]), (0, 0, 0))

    def test_export_print_writes_file_and_sets_overlay_timer(self):
        canvas = air_canvas.AirCanvas.__new__(air_canvas.AirCanvas)
        canvas.frame_w = 20
        canvas.frame_h = 10
        canvas.canvas = np.full((10, 20, 3), 150, dtype=np.uint8)
        canvas.save_overlay_until = 0.0
        canvas.save_flash_until = 0.0
        canvas.save_overlay_path = ""
        canvas.save_overlay_thumbnail = None
        canvas.theme_idx = 0
        canvas.themes = ["dark"]
        canvas.theme_bg_cache = {
            "dark": np.full((10, 20, 3), config.CANVAS_BG_COLOR, dtype=np.uint8)
        }

        fixed_now = datetime(2026, 3, 21, 14, 5, 6)
        frozen_time = 100.0

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(air_canvas, "EXPORT_DIR", tmpdir),
            mock.patch.object(air_canvas, "datetime") as mock_datetime,
            mock.patch.object(air_canvas.time, "time", return_value=frozen_time),
        ):
            mock_datetime.now.return_value = fixed_now
            canvas._export_print()

            expected_path = os.path.join(tmpdir, "art-2026-03-21-140506.png")
            self.assertTrue(os.path.exists(expected_path))
            self.assertGreater(canvas.save_overlay_until, frozen_time)
            self.assertGreater(canvas.save_flash_until, frozen_time)
            self.assertTrue(
                canvas.save_overlay_path.endswith("art-2026-03-21-140506.png")
            )


class LayerCompositionTests(unittest.TestCase):
    def test_compose_art_layers_applies_strokes_on_background(self):
        background = np.zeros((6, 6, 3), dtype=np.uint8)
        stroke_layer = np.zeros_like(background)
        stroke_mask = np.zeros((6, 6), dtype=np.uint8)

        stroke_layer[1, 1] = (10, 20, 30)
        stroke_mask[1, 1] = 255

        composite = air_canvas.compose_art_layers(
            background,
            stroke_layer,
            stroke_mask,
        )

        self.assertEqual(tuple(composite[1, 1]), (10, 20, 30))
        self.assertEqual(tuple(composite[0, 0]), (0, 0, 0))

    def test_compose_current_art_includes_strokes(self):
        canvas = air_canvas.AirCanvas.__new__(air_canvas.AirCanvas)
        canvas.frame_w = 6
        canvas.frame_h = 6
        canvas.stroke_layer = np.zeros((6, 6, 3), dtype=np.uint8)
        canvas.stroke_layer[1, 1] = (10, 20, 30)

        art = canvas._compose_current_art()

        self.assertEqual(tuple(art[1, 1]), (10, 20, 30))


if __name__ == "__main__":
    unittest.main()
