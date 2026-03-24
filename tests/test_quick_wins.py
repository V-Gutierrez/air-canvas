import os
import unittest
from datetime import datetime
from unittest import mock

import numpy as np

import air_canvas
import config


class QuickWinsContractTests(unittest.TestCase):
    def test_rainbow_color_changes_with_hue(self):
        color_a = air_canvas.hue_to_bgr(0)
        color_b = air_canvas.hue_to_bgr(60)

        self.assertNotEqual(color_a, color_b)
        self.assertEqual(len(color_a), 3)

    def test_particle_system_caps_and_expires_particles(self):
        system = air_canvas.ParticleSystem(max_particles=3)

        for _ in range(5):
            system.emit((20, 20), (255, 255, 255), count=1)

        self.assertEqual(len(system.particles), 3)

        for _ in range(120):
            system.update()

        self.assertEqual(len(system.particles), 0)

    def test_theme_background_generation(self):
        bg_dark = air_canvas.generate_theme_background("dark", 100, 100)
        self.assertIsNotNone(bg_dark)
        self.assertEqual(bg_dark.shape, (100, 100, 3))

        bg_space = air_canvas.generate_theme_background("space", 100, 100)
        self.assertIsNotNone(bg_space)
        self.assertGreater(bg_space.sum(), 0)

    def test_avatars_draw_on_canvas(self):
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        air_canvas.draw_penguin(canvas, (50, 50), 10)
        penguin_sum = canvas.sum()
        self.assertGreater(penguin_sum, 0)

        canvas[:] = 0
        air_canvas.draw_cat(canvas, (50, 50), 10)
        cat_sum = canvas.sum()
        self.assertGreater(cat_sum, 0)

    def test_theme_cycling_logic(self):
        themes = ["A", "B", "C"]
        idx = 0
        idx = (idx + 1) % len(themes)
        self.assertEqual(idx, 1)
        idx = (idx + 1) % len(themes)
        self.assertEqual(idx, 2)
        idx = (idx + 1) % len(themes)
        self.assertEqual(idx, 0)


class TierThreeHelperTests(unittest.TestCase):
    def test_draw_alive_transform_waits_for_idle_delay(self):
        shift_x, shift_y, scale = air_canvas.compute_draw_alive_transform(
            now=5.0,
            last_draw_time=2.5,
            delay=3.0,
        )

        self.assertEqual((shift_x, shift_y), (0, 0))
        self.assertEqual(scale, 1.0)

    def test_draw_alive_transform_stays_gentle_after_delay(self):
        shift_x, shift_y, scale = air_canvas.compute_draw_alive_transform(
            now=10.25,
            last_draw_time=0.0,
            delay=3.0,
        )

        self.assertLessEqual(abs(shift_x), 3)
        self.assertLessEqual(abs(shift_y), 3)
        self.assertGreater(scale, 0.98)
        self.assertLess(scale, 1.02)

    def test_shape_hunt_coverage_uses_only_target_and_challenge_delta_mask(self):
        target_mask = np.full((10, 10), 255, dtype=np.uint8)
        user_delta_mask = np.zeros((10, 10), dtype=np.uint8)
        user_delta_mask[:, :7] = 255

        coverage = air_canvas.compute_mask_coverage(target_mask, user_delta_mask)

        self.assertAlmostEqual(coverage, 0.7, places=2)

    def test_shape_hunt_size_shrinks_without_going_below_minimum(self):
        self.assertLess(air_canvas.next_shape_hunt_size(120), 120)
        self.assertEqual(air_canvas.next_shape_hunt_size(40, min_size=48), 48)

    def test_compose_art_layers_uses_stroke_only(self):
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

    def test_build_export_filepath_uses_print_ready_naming_contract(self):
        filepath = air_canvas.build_export_filepath(
            "/tmp/air-canvas-art",
            datetime(2026, 3, 21, 14, 5, 6),
        )

        self.assertEqual(
            filepath,
            "/tmp/air-canvas-art/art-2026-03-21-140506.png",
        )

    def test_print_ready_export_image_upscales_and_adds_frame(self):
        art = np.full((20, 30, 3), 180, dtype=np.uint8)

        export = air_canvas.create_print_ready_image(art, "2026-03-21")

        self.assertGreater(export.shape[0], art.shape[0] * 2)
        self.assertGreater(export.shape[1], art.shape[1] * 2)
        self.assertGreater(int(export.sum()), int(art.sum()))


class ThemeConfigTests(unittest.TestCase):
    def test_camera_theme_is_available(self):
        self.assertIn("camera", config.THEMES)

    def test_themes_list_starts_with_camera(self):
        self.assertEqual(config.THEMES, ["camera", "dark", "space", "forest", "ocean"])

    def test_default_background_theme_is_camera(self):
        self.assertEqual(config.BACKGROUND_THEME, "camera")

    def test_all_theme_backgrounds_return_ndarray(self):
        for theme in config.THEMES:
            bg = air_canvas.generate_theme_background(theme, 80, 60)
            self.assertIsInstance(bg, np.ndarray, f"theme={theme!r} returned None")
            self.assertEqual(bg.shape, (60, 80, 3))

    def test_darken_frame_preserves_shape(self):
        frame = np.full((20, 30, 3), 200, dtype=np.uint8)
        darkened = air_canvas.darken_frame(frame, config.CAMERA_BG_DARKEN_ALPHA)

        self.assertEqual(darkened.shape, frame.shape)
        self.assertLess(int(darkened.sum()), int(frame.sum()))

    def test_forest_and_ocean_dots_use_seeded_rng(self):
        bg1 = air_canvas.generate_theme_background("forest", 200, 150)
        bg2 = air_canvas.generate_theme_background("forest", 200, 150)
        self.assertTrue(
            np.array_equal(bg1, bg2), "forest background must be deterministic"
        )

        bg3 = air_canvas.generate_theme_background("ocean", 200, 150)
        bg4 = air_canvas.generate_theme_background("ocean", 200, 150)
        self.assertTrue(
            np.array_equal(bg3, bg4), "ocean background must be deterministic"
        )


class AvatarTests(unittest.TestCase):
    def test_avatar_config_values(self):
        self.assertEqual(config.AVATAR_LEFT, "penguin")
        self.assertEqual(config.AVATAR_RIGHT, "cat")
        self.assertTrue(config.AVATARS_ENABLED)

    def test_penguin_drawn_on_display_not_canvas(self):
        display = np.zeros((100, 100, 3), dtype=np.uint8)
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        air_canvas.draw_penguin(display, (50, 50), 12)
        self.assertGreater(display.sum(), 0)
        self.assertEqual(canvas.sum(), 0)

    def test_cat_drawn_on_display_not_canvas(self):
        display = np.zeros((100, 100, 3), dtype=np.uint8)
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        air_canvas.draw_cat(display, (50, 50), 12)
        self.assertGreater(display.sum(), 0)
        self.assertEqual(canvas.sum(), 0)


class TierThreeConfigTests(unittest.TestCase):
    def test_draw_alive_defaults_match_contract(self):
        self.assertTrue(config.DRAW_ALIVE_ENABLED)
        self.assertEqual(config.DRAW_ALIVE_DELAY, 3.0)

    def test_shape_hunt_defaults_match_contract(self):
        self.assertFalse(config.SHAPE_HUNT_ENABLED)

    def test_export_dir_defaults_to_desktop_folder(self):
        self.assertTrue(config.EXPORT_DIR.endswith("Desktop/air-canvas-art"))

    def test_palette_defaults_match_backlog_contract(self):
        self.assertEqual(config.PALETTE_DWELL_TIME, 0.3)
        self.assertEqual(config.PALETTE_CIRCLE_RADIUS, 25)

    def test_debug_defaults_off_for_kid_view(self):
        self.assertFalse(config.DEBUG)


class GestureContractTests(unittest.TestCase):
    def _make_landmarks(self, z=0.0):
        """Create 21 mock landmarks with default y=0.7, x=0.5, z=z."""
        return [mock.Mock(y=0.7, x=0.5, z=z) for _ in range(21)]

    def test_open_palm_requires_thumb_extension_for_left_handedness(self):
        landmarks = self._make_landmarks(z=0.0)
        landmarks[air_canvas.THUMB_TIP].x = 0.35
        landmarks[air_canvas.THUMB_IP].x = 0.45
        landmarks[air_canvas.INDEX_TIP].y = 0.2
        landmarks[air_canvas.INDEX_PIP].y = 0.5
        landmarks[air_canvas.MIDDLE_TIP].y = 0.2
        landmarks[air_canvas.MIDDLE_PIP].y = 0.5
        landmarks[air_canvas.RING_TIP].y = 0.2
        landmarks[air_canvas.RING_PIP].y = 0.5
        landmarks[air_canvas.PINKY_TIP].y = 0.2
        landmarks[air_canvas.PINKY_PIP].y = 0.5
        # Palm facing camera: MIDDLE_MCP.z < WRIST.z
        landmarks[air_canvas.MIDDLE_MCP].z = -0.05
        landmarks[air_canvas.WRIST].z = 0.0

        self.assertTrue(air_canvas.is_open_palm(landmarks, "Left"))

    def test_open_palm_rejects_missing_thumb_extension(self):
        landmarks = self._make_landmarks(z=0.0)
        landmarks[air_canvas.THUMB_TIP].x = 0.49
        landmarks[air_canvas.THUMB_IP].x = 0.5
        landmarks[air_canvas.INDEX_TIP].y = 0.2
        landmarks[air_canvas.INDEX_PIP].y = 0.5
        landmarks[air_canvas.MIDDLE_TIP].y = 0.2
        landmarks[air_canvas.MIDDLE_PIP].y = 0.5
        landmarks[air_canvas.RING_TIP].y = 0.2
        landmarks[air_canvas.RING_PIP].y = 0.5
        landmarks[air_canvas.PINKY_TIP].y = 0.2
        landmarks[air_canvas.PINKY_PIP].y = 0.5
        landmarks[air_canvas.MIDDLE_MCP].z = -0.05
        landmarks[air_canvas.WRIST].z = 0.0

        self.assertFalse(air_canvas.is_open_palm(landmarks, "Right"))

    def test_open_palm_rejects_palm_facing_away(self):
        landmarks = self._make_landmarks(z=0.0)
        # All fingers extended
        landmarks[air_canvas.THUMB_TIP].x = 0.35
        landmarks[air_canvas.THUMB_IP].x = 0.45
        landmarks[air_canvas.INDEX_TIP].y = 0.2
        landmarks[air_canvas.INDEX_PIP].y = 0.5
        landmarks[air_canvas.MIDDLE_TIP].y = 0.2
        landmarks[air_canvas.MIDDLE_PIP].y = 0.5
        landmarks[air_canvas.RING_TIP].y = 0.2
        landmarks[air_canvas.RING_PIP].y = 0.5
        landmarks[air_canvas.PINKY_TIP].y = 0.2
        landmarks[air_canvas.PINKY_PIP].y = 0.5
        # Palm facing AWAY: MIDDLE_MCP.z > WRIST.z
        landmarks[air_canvas.MIDDLE_MCP].z = 0.05
        landmarks[air_canvas.WRIST].z = 0.0

        self.assertFalse(air_canvas.is_open_palm(landmarks, "Left"))

    def test_is_palm_facing_camera(self):
        landmarks = self._make_landmarks(z=0.0)
        landmarks[air_canvas.MIDDLE_MCP].z = -0.05
        landmarks[air_canvas.WRIST].z = 0.0
        self.assertTrue(air_canvas.is_palm_facing_camera(landmarks))

        landmarks[air_canvas.MIDDLE_MCP].z = 0.05
        landmarks[air_canvas.WRIST].z = 0.0
        self.assertFalse(air_canvas.is_palm_facing_camera(landmarks))

    def test_pinch_active_flag_gates_color_cycling(self):
        state = air_canvas.HandState([(0, 255, 255)])
        self.assertFalse(state.pinch_active)
        state.pinch_active = True
        self.assertTrue(state.pinch_active)


class SaveFeedbackTests(unittest.TestCase):
    def test_display_path_shortens_home_directory(self):
        home = os.path.expanduser("~")
        path = os.path.join(home, "Desktop", "air-canvas-art", "art-2026.png")

        self.assertEqual(
            air_canvas.display_path(path),
            "~/Desktop/air-canvas-art/art-2026.png",
        )

    def test_build_save_thumbnail_uses_configured_size(self):
        image = np.full((60, 80, 3), 180, dtype=np.uint8)
        thumb = air_canvas.build_save_thumbnail(image)

        self.assertEqual(
            thumb.shape,
            (config.SAVE_THUMBNAIL_HEIGHT, config.SAVE_THUMBNAIL_WIDTH, 3),
        )


if __name__ == "__main__":
    unittest.main()
