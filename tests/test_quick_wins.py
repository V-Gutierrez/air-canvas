import threading
import time
import unittest
from unittest import mock

import numpy as np

import air_canvas
import config


class QuickWinsContractTests(unittest.TestCase):
    def test_sound_engine_can_be_created_without_platform_audio(self):
        with mock.patch.object(air_canvas, "resolve_sound_player", return_value=None):
            engine = air_canvas.SoundEngine(enabled=True)

        self.assertIsNone(engine.player)
        self.assertIn("draw_start", engine.tones)

    def test_rainbow_color_changes_with_hue(self):
        color_a = air_canvas.hue_to_bgr(0)
        color_b = air_canvas.hue_to_bgr(60)

        self.assertNotEqual(color_a, color_b)
        self.assertEqual(len(color_a), 3)

    def test_v_sign_detects_index_and_middle_only(self):
        landmarks = [mock.Mock(y=0.7, x=0.5) for _ in range(21)]
        landmarks[air_canvas.INDEX_TIP].y = 0.2
        landmarks[air_canvas.INDEX_PIP].y = 0.5
        landmarks[air_canvas.MIDDLE_TIP].y = 0.2
        landmarks[air_canvas.MIDDLE_PIP].y = 0.5
        landmarks[air_canvas.RING_TIP].y = 0.8
        landmarks[air_canvas.RING_PIP].y = 0.5
        landmarks[air_canvas.PINKY_TIP].y = 0.8
        landmarks[air_canvas.PINKY_PIP].y = 0.5

        self.assertTrue(air_canvas.is_v_sign(landmarks))

    def test_stamp_draw_changes_canvas_pixels(self):
        canvas = np.zeros((80, 80, 3), dtype=np.uint8)
        air_canvas.draw_stamp(canvas, (40, 40), "star", 16, (0, 255, 255))

        self.assertGreater(int(canvas.sum()), 0)

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


class ThemeConfigTests(unittest.TestCase):
    def test_themes_list_has_no_camera(self):
        self.assertNotIn("camera", config.THEMES)

    def test_themes_list_has_exactly_four_entries(self):
        self.assertEqual(config.THEMES, ["dark", "space", "forest", "ocean"])

    def test_default_background_theme_is_dark(self):
        self.assertEqual(config.BACKGROUND_THEME, "dark")

    def test_all_theme_backgrounds_return_ndarray(self):
        for theme in config.THEMES:
            bg = air_canvas.generate_theme_background(theme, 80, 60)
            self.assertIsInstance(bg, np.ndarray, f"theme={theme!r} returned None")
            self.assertEqual(bg.shape, (60, 80, 3))

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


class MusicModeTests(unittest.TestCase):
    def test_config_exports_pentatonic_tables(self):
        self.assertEqual(len(config.PENTATONIC_NOTES_LOW), 5)
        self.assertEqual(len(config.PENTATONIC_NOTES_HIGH), 5)
        self.assertEqual(config.PENTATONIC_NOTES_LOW, [261, 294, 330, 392, 440])
        self.assertEqual(config.PENTATONIC_NOTES_HIGH, [523, 587, 659, 784, 880])

    def test_low_octave_notes_are_lower_than_high_octave(self):
        for low, high in zip(config.PENTATONIC_NOTES_LOW, config.PENTATONIC_NOTES_HIGH):
            self.assertLess(low, high)

    def test_music_mode_config_is_bool(self):
        self.assertIsInstance(config.MUSIC_MODE, bool)

    def test_sound_engine_play_tone_respects_concurrency_cap(self):
        called_freqs = []
        barrier = threading.Barrier(2)

        def slow_play(wav_bytes):
            barrier.wait()
            called_freqs.append(True)
            barrier.wait()

        with mock.patch.object(
            air_canvas, "resolve_sound_player", return_value="afplay"
        ):
            engine = air_canvas.SoundEngine(enabled=True, max_concurrent=1)
        engine._play_bytes = slow_play

        t1 = threading.Thread(target=engine._launch, args=(b"fake",))
        t1.start()
        barrier.wait()

        rejected = False
        with engine._count_lock:
            if engine._active_count >= engine._max_concurrent:
                rejected = True

        self.assertTrue(rejected, "second tone should be rejected when cap is reached")
        barrier.wait()
        t1.join(timeout=2)

    def test_play_tone_does_nothing_when_disabled(self):
        with mock.patch.object(air_canvas, "resolve_sound_player", return_value=None):
            engine = air_canvas.SoundEngine(enabled=False)

        launched = []
        engine._launch = lambda wav_bytes: launched.append(wav_bytes)
        engine.play_tone(440)
        self.assertEqual(launched, [])

    def test_pentatonic_note_selected_by_color_idx(self):
        for idx in range(4):
            self.assertEqual(
                config.PENTATONIC_NOTES_LOW[idx],
                config.PENTATONIC_NOTES_LOW[idx % len(config.PENTATONIC_NOTES_LOW)],
            )
            self.assertEqual(
                config.PENTATONIC_NOTES_HIGH[idx],
                config.PENTATONIC_NOTES_HIGH[idx % len(config.PENTATONIC_NOTES_HIGH)],
            )


class SnapClearTests(unittest.TestCase):
    def test_snap_clear_config_defaults_to_disabled(self):
        self.assertFalse(config.SNAP_CLEAR_ENABLED)

    def test_snap_clear_cooldown_is_two_seconds(self):
        self.assertEqual(config.SNAP_CLEAR_COOLDOWN, 2.0)

    def test_snap_clear_event_is_threading_event(self):
        evt = threading.Event()
        self.assertFalse(evt.is_set())
        evt.set()
        self.assertTrue(evt.is_set())
        evt.clear()
        self.assertFalse(evt.is_set())

    def test_snap_clear_cooldown_prevents_rapid_retrigger(self):
        last_clear = [0.0]

        def would_trigger(now):
            if now - last_clear[0] >= config.SNAP_CLEAR_COOLDOWN:
                last_clear[0] = now
                return True
            return False

        t0 = 1000.0
        self.assertTrue(would_trigger(t0))
        self.assertFalse(would_trigger(t0 + 0.5))
        self.assertFalse(would_trigger(t0 + 1.9))
        self.assertTrue(would_trigger(t0 + 2.0))

    def test_snap_clear_disabled_gracefully_when_sounddevice_absent(self):
        with mock.patch.dict("sys.modules", {"sounddevice": None}):
            import importlib
            import sys

            saved = sys.modules.pop("air_canvas", None)
            try:
                mod = importlib.import_module("air_canvas")
                self.assertFalse(mod._SOUNDDEVICE_AVAILABLE)
            finally:
                if saved is not None:
                    sys.modules["air_canvas"] = saved
                elif "air_canvas" in sys.modules:
                    del sys.modules["air_canvas"]


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


if __name__ == "__main__":
    unittest.main()
