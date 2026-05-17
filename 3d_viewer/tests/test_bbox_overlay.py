from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.bbox_overlay import bbox_edge_samples, split_wrapped_bbox, uv_to_pano_local_dirs


class BboxOverlayTest(unittest.TestCase):
    def test_regular_bbox_stays_single_box(self):
        boxes = split_wrapped_bbox([10.0, 20.0, 30.0, 40.0], pano_w=100.0)

        self.assertEqual(boxes, [[10.0, 20.0, 30.0, 40.0]])

    def test_wrapped_bbox_splits_at_panorama_seam(self):
        boxes = split_wrapped_bbox([90.0, 20.0, 10.0, 40.0], pano_w=100.0)

        self.assertEqual(boxes, [
            [90.0, 20.0, 100.0, 40.0],
            [0.0, 20.0, 10.0, 40.0],
        ])

    def test_edge_samples_form_line_segments(self):
        segments = bbox_edge_samples([10.0, 20.0, 30.0, 40.0], pano_w=100.0, samples_per_edge=3)

        self.assertEqual(segments.shape, (16, 2))
        np.testing.assert_allclose(segments[0], [10.0, 20.0])
        np.testing.assert_allclose(segments[1], [20.0, 20.0])
        np.testing.assert_allclose(segments[-2], [10.0, 30.0])
        np.testing.assert_allclose(segments[-1], [10.0, 40.0])

    def test_uv_to_pano_local_dirs_maps_center_to_front(self):
        dirs = uv_to_pano_local_dirs(
            np.array([[2880.0, 1440.0]], dtype=np.float32),
            img_w=5760.0,
            img_h=2880.0,
            yaw_offset_deg=0.0,
        )

        np.testing.assert_allclose(dirs[0], [0.0, 1.0, 0.0], atol=1e-6)

    def test_uv_to_pano_local_dirs_removes_display_yaw_offset(self):
        raw = uv_to_pano_local_dirs(
            np.array([[2880.0, 1440.0]], dtype=np.float32),
            img_w=5760.0,
            img_h=2880.0,
            yaw_offset_deg=0.0,
        )
        shifted = uv_to_pano_local_dirs(
            np.array([[1440.0, 1440.0]], dtype=np.float32),
            img_w=5760.0,
            img_h=2880.0,
            yaw_offset_deg=-90.0,
        )

        np.testing.assert_allclose(shifted, raw, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
