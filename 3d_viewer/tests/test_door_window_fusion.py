from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.door_window_fusion import (
    extract_detection_region_from_bbox,
    fuse_detection_and_pointcloud,
    frustum_candidate_mask,
    should_highlight_fused,
)
from core.projection import project_points_to_panorama, rotation_from_angle


def _vertical_patch(width: float, height: float, y: float, cols: int = 14, rows: int = 18) -> np.ndarray:
    xs = np.linspace(-width / 2.0, width / 2.0, cols)
    zs = np.linspace(1.0, 1.0 + height, rows)
    return np.array([[x, y, z] for z in zs for x in xs], dtype=np.float64)


class FusionTest(unittest.TestCase):
    def setUp(self):
        # A door panel at y=2 and a parallel wall slab behind it at y=2.6, both
        # in front of a camera at the origin looking down +y (yaw such that the
        # panorama projection places them centrally).
        self.cam = np.zeros(3, dtype=np.float64)
        self.R = rotation_from_angle(0.0, 0.0, 0.0)
        self.img_w, self.img_h = 1000, 500
        self.door = _vertical_patch(width=0.9, height=2.0, y=2.0)
        self.wall = _vertical_patch(width=3.0, height=2.4, y=2.6, cols=24, rows=22)
        self.points = np.vstack([self.door, self.wall])

        uv = project_points_to_panorama(
            self.points, self.cam, self.R, self.img_w, self.img_h, yaw_offset_deg=0.0
        )
        # Build a bbox that tightly contains the door's projected pixels.
        door_uv = uv[: len(self.door)]
        x1, y1 = door_uv.min(axis=0)
        x2, y2 = door_uv.max(axis=0)
        pad = 4.0
        self.detection = {
            "label": "door",
            "score": 0.9,
            "bbox": [x1 - pad, y1 - pad, x2 + pad, y2 + pad],
        }

    def test_frustum_mask_contains_door_pixels(self):
        mask = frustum_candidate_mask(
            self.points, self.cam, self.R, self.img_w, self.img_h, self.detection,
            yaw_offset_deg=0.0,
        )
        # Most door points fall inside the door bbox frustum.
        self.assertGreater(mask[: len(self.door)].mean(), 0.8)

    def test_fusion_restricts_growth_to_frustum_and_keeps_door_label(self):
        seed = len(self.door) // 2  # a door point
        fused = fuse_detection_and_pointcloud(
            self.points, seed, [self.detection], self.cam, self.R,
            self.img_w, self.img_h, yaw_offset_deg=0.0,
        )
        self.assertEqual(fused.source, "fused")
        self.assertEqual(fused.detection_index, 0)
        self.assertEqual(fused.label, "door")
        # The fused region is dominated by door points, not the wider wall slab.
        door_hits = int(fused.mask[: len(self.door)].sum())
        wall_hits = int(fused.mask[len(self.door):].sum())
        self.assertGreater(door_hits, wall_hits)
        self.assertTrue(should_highlight_fused(fused))

    def test_extract_detection_region_from_bbox_finds_door_without_clicked_seed(self):
        fused = extract_detection_region_from_bbox(
            self.points,
            0,
            [self.detection],
            self.cam,
            self.R,
            self.img_w,
            self.img_h,
            yaw_offset_deg=0.0,
        )

        self.assertEqual(fused.source, "fused")
        self.assertEqual(fused.detection_index, 0)
        self.assertEqual(fused.label, "door")
        door_hits = int(fused.mask[: len(self.door)].sum())
        wall_hits = int(fused.mask[len(self.door):].sum())
        self.assertGreater(door_hits, wall_hits)
        self.assertTrue(should_highlight_fused(fused))

    def test_fusion_reports_seed_outside_any_frustum(self):
        # A seed on the far wall, projected outside the (door-sized) bbox edges.
        far_seed = len(self.door)  # first wall point (a wide-slab corner)
        # Use a tiny bbox that only covers the door center so wall corners miss.
        tiny = {
            "label": "door",
            "score": 0.9,
            "bbox": [self.detection["bbox"][0] + 200, self.detection["bbox"][1],
                     self.detection["bbox"][0] + 220, self.detection["bbox"][3]],
        }
        fused = fuse_detection_and_pointcloud(
            self.points, far_seed, [tiny], self.cam, self.R,
            self.img_w, self.img_h, yaw_offset_deg=0.0,
        )
        # Either the seed is not in the frustum, or fusion produced no region.
        self.assertIn(fused.source, {"none"})

    def test_fusion_no_detections_returns_none_source(self):
        fused = fuse_detection_and_pointcloud(
            self.points, 0, [], self.cam, self.R, self.img_w, self.img_h,
        )
        self.assertEqual(fused.source, "none")
        self.assertEqual(fused.reason, "no_detections")
        self.assertFalse(should_highlight_fused(fused))

    def test_fusion_seed_out_of_range(self):
        fused = fuse_detection_and_pointcloud(
            self.points, 10_000, [self.detection], self.cam, self.R,
            self.img_w, self.img_h, yaw_offset_deg=0.0,
        )
        self.assertEqual(fused.reason, "seed_out_of_range")


if __name__ == "__main__":
    unittest.main()
