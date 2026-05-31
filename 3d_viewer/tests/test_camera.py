from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from render.camera import Camera


class CameraTest(unittest.TestCase):
    def test_keyframe_reset_uses_current_panorama_yaw_offset(self):
        camera = Camera()

        camera.set_keyframe(
            position=[0.0, 0.0, 0.0],
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            pano_yaw_offset_deg=0.0,
        )

        self.assertAlmostEqual(camera.yaw_deg, 90.0, places=6)

        camera.set_keyframe(
            position=[0.0, 0.0, 0.0],
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            pano_yaw_offset_deg=-90.0,
        )

        self.assertAlmostEqual(camera.yaw_deg, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
