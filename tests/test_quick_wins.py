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
        self.assertEqual(config.PALETTE_CIRCLE_RADIUS, 36)

    def test_debug_defaults_off_for_kid_view(self):
        self.assertFalse(config.DEBUG)

    def test_kid_mode_thresholds_are_more_permissive(self):
        self.assertEqual(config.DETECTION_CONFIDENCE, 0.5)
        self.assertEqual(config.TRACKING_CONFIDENCE, 0.4)
        self.assertEqual(config.PINCH_THRESHOLD, 0.08)
        self.assertEqual(config.MIN_PALM_SIZE, 0.055)

    def test_color_names_and_tones_match_palette_sizes(self):
        self.assertEqual(
            len(config.LEFT_HAND_COLOR_NAMES), len(config.LEFT_HAND_COLORS)
        )
        self.assertEqual(
            len(config.RIGHT_HAND_COLOR_NAMES), len(config.RIGHT_HAND_COLORS)
        )
        self.assertEqual(
            len(config.COLOR_TONE_FREQUENCIES), len(config.LEFT_HAND_COLORS)
        )

    def test_audio_defaults_enable_native_feedback(self):
        self.assertTrue(config.AUDIO_ENABLED)
        self.assertTrue(config.TTS_ENABLED)
        self.assertTrue(config.TONE_ENABLED)
        self.assertTrue(config.BG_MUSIC_ENABLED)


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

    def test_finger_extension_detects_sideways_index(self):
        landmarks = self._make_landmarks(z=0.0)
        landmarks[air_canvas.INDEX_MCP].x = 0.30
        landmarks[air_canvas.INDEX_PIP].x = 0.48
        landmarks[air_canvas.INDEX_TIP].x = 0.72
        self.assertTrue(
            air_canvas.is_finger_extended(
                landmarks, air_canvas.INDEX_TIP, air_canvas.INDEX_PIP
            )
        )

    def test_finger_extension_detects_downward_index(self):
        landmarks = self._make_landmarks(z=0.0)
        landmarks[air_canvas.INDEX_MCP].y = 0.30
        landmarks[air_canvas.INDEX_PIP].y = 0.48
        landmarks[air_canvas.INDEX_TIP].y = 0.74
        self.assertTrue(
            air_canvas.is_finger_extended(
                landmarks, air_canvas.INDEX_TIP, air_canvas.INDEX_PIP
            )
        )

    def test_finger_extension_rejects_curled_index(self):
        landmarks = self._make_landmarks(z=0.0)
        landmarks[air_canvas.INDEX_MCP].x = 0.30
        landmarks[air_canvas.INDEX_PIP].x = 0.60
        landmarks[air_canvas.INDEX_TIP].x = 0.43
        self.assertFalse(
            air_canvas.is_finger_extended(
                landmarks, air_canvas.INDEX_TIP, air_canvas.INDEX_PIP
            )
        )

    def test_open_palm_rejects_small_palms(self):
        landmarks = self._make_landmarks(z=0.0)
        landmarks[air_canvas.INDEX_MCP].x = 0.505
        landmarks[air_canvas.MIDDLE_MCP].x = 0.506
        landmarks[air_canvas.PINKY_MCP].x = 0.507
        self.assertLess(air_canvas.palm_size(landmarks), config.MIN_PALM_SIZE)


class AudioAndPaletteTests(unittest.TestCase):
    def test_hand_state_exposes_color_name_and_tone(self):
        state = air_canvas.HandState(
            config.LEFT_HAND_COLORS,
            config.LEFT_HAND_COLOR_NAMES,
            config.COLOR_TONE_FREQUENCIES,
        )
        self.assertEqual(state.color_name, "Cyan")
        self.assertEqual(state.tone_frequency, config.COLOR_TONE_FREQUENCIES[0])

    def test_set_state_color_announces_only_on_change(self):
        canvas = air_canvas.AirCanvas.__new__(air_canvas.AirCanvas)
        canvas.audio = mock.Mock()
        state = air_canvas.HandState(
            config.LEFT_HAND_COLORS,
            config.LEFT_HAND_COLOR_NAMES,
            config.COLOR_TONE_FREQUENCIES,
        )

        changed = canvas._set_state_color(state, 1, 10.0)
        self.assertTrue(changed)
        canvas.audio.speak.assert_called_once_with("Magenta")
        canvas.audio.play_tone.assert_called_once_with(config.COLOR_TONE_FREQUENCIES[1])

        canvas.audio.reset_mock()
        changed = canvas._set_state_color(state, 1, 11.0)
        self.assertFalse(changed)
        canvas.audio.speak.assert_not_called()
        canvas.audio.play_tone.assert_not_called()

    def test_palette_hover_commits_color_after_dwell(self):
        canvas = air_canvas.AirCanvas.__new__(air_canvas.AirCanvas)
        canvas.audio = mock.Mock()
        canvas.frame_w = 640
        canvas.frame_h = 480
        canvas.left_hand = air_canvas.HandState(
            config.LEFT_HAND_COLORS,
            config.LEFT_HAND_COLOR_NAMES,
            config.COLOR_TONE_FREQUENCIES,
        )
        canvas.right_hand = air_canvas.HandState(
            config.RIGHT_HAND_COLORS,
            config.RIGHT_HAND_COLOR_NAMES,
            list(reversed(config.COLOR_TONE_FREQUENCIES)),
        )
        state = canvas.left_hand
        center = canvas._palette_positions(state)[2]

        self.assertTrue(canvas._update_palette_hover(state, center, 1.0))
        self.assertEqual(state.palette_hover_idx, 2)
        self.assertEqual(state.color_idx, 0)

        self.assertTrue(
            canvas._update_palette_hover(state, center, 1.0 + config.PALETTE_DWELL_TIME)
        )
        self.assertEqual(state.color_idx, 2)
        canvas.audio.speak.assert_called_once_with(config.LEFT_HAND_COLOR_NAMES[2])

    def test_audio_manager_skips_when_not_on_darwin(self):
        manager = air_canvas.AudioManager()
        with (
            mock.patch.object(air_canvas.sys, "platform", "linux"),
            mock.patch.object(air_canvas.subprocess, "Popen") as popen,
        ):
            manager.speak("Blue")
            manager.play_tone(523)
            manager.start_music()
        popen.assert_not_called()

    def test_audio_manager_speak_uses_say(self):
        manager = air_canvas.AudioManager()
        fake_process = mock.Mock()
        fake_process.poll.return_value = 0
        with (
            mock.patch.object(air_canvas.sys, "platform", "darwin"),
            mock.patch.object(air_canvas.os.path, "exists", return_value=True),
            mock.patch.object(
                air_canvas.subprocess, "Popen", return_value=fake_process
            ) as popen,
        ):
            manager.speak("Blue")

        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/say")
        self.assertEqual(command[-1], "Blue")

    def test_audio_manager_play_tone_uses_afplay(self):
        manager = air_canvas.AudioManager()
        fake_process = mock.Mock()
        fake_process.poll.return_value = 0
        with (
            mock.patch.object(air_canvas.sys, "platform", "darwin"),
            mock.patch.object(air_canvas.os.path, "exists", return_value=True),
            mock.patch.object(
                manager, "_create_wave_file", return_value="/tmp/tone.wav"
            ) as create_wave,
            mock.patch.object(
                air_canvas.subprocess, "Popen", return_value=fake_process
            ) as popen,
        ):
            manager.play_tone(523)

        create_wave.assert_called_once()
        popen.assert_called_once_with(
            ["/usr/bin/afplay", "/tmp/tone.wav"],
            stdout=air_canvas.subprocess.DEVNULL,
            stderr=air_canvas.subprocess.DEVNULL,
        )

    def test_audio_manager_start_music_spawns_background_thread(self):
        manager = air_canvas.AudioManager()
        fake_thread = mock.Mock()
        fake_thread.is_alive.return_value = False
        with (
            mock.patch.object(air_canvas.sys, "platform", "darwin"),
            mock.patch.object(air_canvas.os.path, "exists", return_value=True),
            mock.patch.object(
                air_canvas.threading, "Thread", return_value=fake_thread
            ) as thread_cls,
        ):
            manager.start_music()

        thread_cls.assert_called_once()
        fake_thread.start.assert_called_once()


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
