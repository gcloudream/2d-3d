from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.projection import project_points_to_panorama


def reference_coordinate_to_pixel(pt: np.ndarray, img_w: int, img_h: int) -> tuple[float, float]:
    tm = np.arctan(-pt[0] / pt[1]) if pt[1] != 0 else 0.0

    if pt[0] < 0:
        if pt[1] > 0:
            u = int(img_h - img_h * tm / np.pi + 0.5) % img_w
        else:
            u = int(img_h * (-tm) / np.pi + 0.5) % img_w
    else:
        if pt[1] > 0:
            u = int(img_h - img_h * tm / np.pi + 0.5) % img_w
        else:
            u = int(img_w - img_h * tm / np.pi + 0.5) % img_w

    horizontal_dist = np.sqrt(pt[0] * pt[0] + pt[1] * pt[1])
    tm = np.arctan(horizontal_dist / pt[2]) if pt[2] != 0 else np.pi / 2
    if tm < 0:
        tm += np.pi
    v = int(img_h * tm / np.pi + 0.5) % img_h
    return float(u), float(v)


class ProjectPointsToPanoramaTest(unittest.TestCase):
    def test_matches_reference_formula_across_quadrants(self):
        img_w, img_h = 5760, 2880
        local_points = np.array([
            [-2.0, 3.0, 1.0],
            [-2.0, -3.0, 1.0],
            [2.0, 3.0, 1.0],
            [2.0, -3.0, 1.0],
            [0.0, 4.0, 2.0],
            [0.0, -4.0, -2.0],
        ], dtype=np.float64)

        got = project_points_to_panorama(
            local_points,
            cam_pos=np.zeros(3),
            R_pano=np.eye(3),
            img_w=img_w,
            img_h=img_h,
        )
        expected = np.array([
            reference_coordinate_to_pixel(p, img_w, img_h)
            for p in local_points
        ])

        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_applies_yaw_offset_like_shader_sampling(self):
        img_w, img_h = 5760, 2880
        point = np.array([[0.0, 4.0, 1.0]], dtype=np.float64)

        raw = project_points_to_panorama(
            point, np.zeros(3), np.eye(3), img_w, img_h, yaw_offset_deg=0.0,
        )
        shifted = project_points_to_panorama(
            point, np.zeros(3), np.eye(3), img_w, img_h, yaw_offset_deg=-90.0,
        )

        expected_u = (raw[0, 0] - img_w / 4.0) % img_w
        self.assertAlmostEqual(shifted[0, 0], expected_u)
        self.assertAlmostEqual(shifted[0, 1], raw[0, 1])

    def test_wraps_horizontal_offset_at_image_boundary(self):
        img_w, img_h = 5760, 2880
        point = np.array([[2.0, -3.0, 1.0]], dtype=np.float64)

        raw = project_points_to_panorama(
            point, np.zeros(3), np.eye(3), img_w, img_h, yaw_offset_deg=0.0,
        )
        shifted = project_points_to_panorama(
            point, np.zeros(3), np.eye(3), img_w, img_h, yaw_offset_deg=180.0,
        )

        self.assertGreaterEqual(shifted[0, 0], 0.0)
        self.assertLess(shifted[0, 0], img_w)
        self.assertAlmostEqual(shifted[0, 0], (raw[0, 0] + img_w / 2.0) % img_w)


if __name__ == "__main__":
    unittest.main()
