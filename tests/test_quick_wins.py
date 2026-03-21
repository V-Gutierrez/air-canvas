import unittest
from unittest import mock

import numpy as np

import air_canvas


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


if __name__ == "__main__":
    unittest.main()
