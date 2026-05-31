from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from render.picking import find_nearest_to_mouse


class PickingTest(unittest.TestCase):
    def test_find_nearest_to_mouse_prefers_frontmost_point_within_pick_radius(self):
        points = np.array(
            [
                [0.00, 0.00, 0.60],   # exact screen hit, but farther away
                [0.16, 0.00, -0.80],  # 8px away in a 100px viewport, but in front
            ],
            dtype=np.float32,
        )
        mvp = np.eye(4, dtype=np.float32)

        idx = find_nearest_to_mouse(points, mvp, 100, 100, 50, 50, max_dist_px=10)

        self.assertEqual(idx, 1)


if __name__ == "__main__":
    unittest.main()
