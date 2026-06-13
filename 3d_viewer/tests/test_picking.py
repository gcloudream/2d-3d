from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from render.picking import (
    ScreenPointIndex,
    _project_points_to_screen_with_depth,
    find_nearest_to_mouse,
)


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

    def test_screen_point_index_reuses_projection_for_same_view(self):
        points = np.array(
            [
                [0.00, 0.00, 0.60],
                [0.16, 0.00, -0.80],
                [-0.10, 0.00, -0.20],
            ],
            dtype=np.float32,
        )
        mvp = np.eye(4, dtype=np.float32)
        index = ScreenPointIndex()

        with patch(
            "render.picking._project_points_to_screen_with_depth",
            wraps=_project_points_to_screen_with_depth,
        ) as project:
            first = index.find_nearest(points, mvp, 100, 100, 50, 50, max_dist_px=12)
            second = index.find_nearest(points, mvp, 100, 100, 58, 50, max_dist_px=12)
            third = index.find_nearest(points, mvp, 120, 100, 60, 50, max_dist_px=12)

        self.assertEqual(first, 1)
        self.assertEqual(second, 1)
        self.assertGreaterEqual(third, 0)
        self.assertEqual(project.call_count, 2)


if __name__ == "__main__":
    unittest.main()
