from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from render.orbit_camera import OrbitCamera


class OrbitCameraTest(unittest.TestCase):
    def test_fit_to_points_centers_camera_on_point_cloud_bounds(self):
        camera = OrbitCamera()
        points = np.array([
            [-2.0, -1.0, 0.0],
            [4.0, 3.0, 2.0],
        ], dtype=np.float64)

        camera.fit_to_points(points)

        np.testing.assert_allclose(camera.target, [1.0, 1.0, 1.0])
        self.assertGreater(camera.distance, 7.0)

    def test_orbit_changes_eye_position_around_same_target(self):
        camera = OrbitCamera()
        camera.fit_to_points(np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64))
        before_target = camera.target.copy()
        before_eye = camera.position.copy()

        camera.orbit(30.0, -10.0)

        np.testing.assert_allclose(camera.target, before_target)
        self.assertGreater(np.linalg.norm(camera.position - before_eye), 0.1)

    def test_zoom_changes_distance_not_field_of_view(self):
        camera = OrbitCamera()
        camera.fit_to_points(np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64))
        before_distance = camera.distance
        before_fov = camera.fov_deg

        camera.zoom(-3.0)

        self.assertLess(camera.distance, before_distance)
        self.assertEqual(camera.fov_deg, before_fov)


if __name__ == "__main__":
    unittest.main()
