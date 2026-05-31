from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.pointcloud_extract import (
    extract_planar_region_from_seed,
    infer_label_from_dimensions,
    should_highlight_planar_region,
)


def _vertical_patch(width: float, height: float, cols: int = 14, rows: int = 18) -> np.ndarray:
    """A dense vertical rectangular patch in the x-z plane (normal ~ +y)."""
    xs = np.linspace(-width / 2.0, width / 2.0, cols)
    zs = np.linspace(1.0, 1.0 + height, rows)
    return np.array([[x, 0.0, z] for z in zs for x in xs], dtype=np.float64)


class PointcloudExtractTest(unittest.TestCase):
    def test_extracts_isolated_vertical_door_patch(self):
        points = _vertical_patch(width=0.9, height=2.0)
        seed = len(points) // 2

        selection = extract_planar_region_from_seed(points, seed)

        self.assertEqual(selection.confidence, "high")
        self.assertEqual(selection.reason, "accepted_vertical_planar_region")
        self.assertEqual(selection.label, "door")
        self.assertTrue(should_highlight_planar_region(selection))
        self.assertGreater(selection.point_count, 100)
        self.assertIsNotNone(selection.plane_normal)
        self.assertLess(abs(float(selection.plane_normal[2])), 0.2)

    def test_drops_coplanar_isolated_stragglers(self):
        # Dense door patch plus a few coplanar stragglers far from the body but
        # on the same plane (y=0). They pass the plane band yet are isolated, so
        # the statistical outlier removal must drop them. The patch is sampled
        # densely enough (like the real cloud) for the density test to engage.
        door = _vertical_patch(width=0.9, height=2.0, cols=28, rows=36)
        stragglers = np.array(
            [[2.5, 0.0, 0.5], [-2.0, 0.0, 2.5], [2.2, 0.0, 3.0]],
            dtype=np.float64,
        )
        points = np.vstack([door, stragglers])
        seed = len(door) // 2

        selection = extract_planar_region_from_seed(points, seed)

        self.assertEqual(selection.confidence, "high")
        # The three coplanar stragglers are not highlighted.
        self.assertFalse(selection.mask[len(door):].any())

    def test_does_not_bleed_onto_perpendicular_floor(self):
        wall = _vertical_patch(width=0.9, height=2.0)
        # A horizontal floor sharing the bottom edge of the wall (normal ~ +z).
        floor = np.array(
            [[x, y, 1.0] for y in np.linspace(0.04, 1.0, 16) for x in np.linspace(-0.45, 0.45, 14)],
            dtype=np.float64,
        )
        points = np.vstack([wall, floor])
        seed = len(wall) // 2

        selection = extract_planar_region_from_seed(points, seed)

        # The highlighted region is dominated by wall points, not floor points.
        wall_hits = int(selection.mask[: len(wall)].sum())
        floor_hits = int(selection.mask[len(wall):].sum())
        self.assertGreater(wall_hits, floor_hits)

    def test_rejects_scattered_blob(self):
        rng = np.random.default_rng(0)
        points = rng.uniform(-0.5, 0.5, size=(300, 3))
        seed = 0

        selection = extract_planar_region_from_seed(points, seed)

        self.assertNotEqual(selection.confidence, "high")

    def test_returns_too_few_region_points_for_sparse_seed(self):
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.02, 0.0, 1.0],
            [5.0, 5.0, 5.0],
        ], dtype=np.float64)

        selection = extract_planar_region_from_seed(points, 0)

        self.assertIn(selection.reason, {"too_few_seed_neighbors", "too_few_region_points"})
        self.assertNotEqual(selection.confidence, "high")

    def test_seed_out_of_range_returns_empty(self):
        points = _vertical_patch(width=0.9, height=2.0)

        selection = extract_planar_region_from_seed(points, 10_000)

        self.assertEqual(selection.confidence, "none")
        self.assertEqual(selection.reason, "seed_out_of_range")
        self.assertEqual(selection.point_count, 0)
        self.assertFalse(should_highlight_planar_region(selection))

    def test_infer_label_from_dimensions(self):
        self.assertEqual(infer_label_from_dimensions(0.9, 2.0), "door")
        self.assertEqual(infer_label_from_dimensions(1.0, 1.0), "window")
        self.assertEqual(infer_label_from_dimensions(5.0, 5.0), "object")
        self.assertEqual(infer_label_from_dimensions(None, None), "object")


if __name__ == "__main__":
    unittest.main()
