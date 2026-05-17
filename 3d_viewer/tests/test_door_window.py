from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.door_window import match_points_to_detections, select_detection_region


class MatchPointsToDetectionsTest(unittest.TestCase):
    def test_marks_points_inside_regular_bbox(self):
        uv = np.array([
            [15.0, 15.0],
            [25.0, 25.0],
            [45.0, 20.0],
        ])
        dets = [{"label": "door", "score": 0.8, "bbox": [10.0, 10.0, 30.0, 30.0]}]

        result = match_points_to_detections(uv, dets, pano_w=100.0)

        self.assertEqual(result.match_indices.tolist(), [0, 0, -1])
        self.assertEqual(result.hit_mask.tolist(), [True, True, False])

    def test_handles_bbox_that_wraps_panorama_seam(self):
        uv = np.array([
            [95.0, 20.0],
            [5.0, 20.0],
            [50.0, 20.0],
            [95.0, 50.0],
        ])
        dets = [{"label": "window", "score": 0.7, "bbox": [90.0, 10.0, 10.0, 30.0]}]

        result = match_points_to_detections(uv, dets, pano_w=100.0)

        self.assertEqual(result.match_indices.tolist(), [0, 0, -1, -1])

    def test_prefers_smallest_containing_bbox_for_nested_detections(self):
        uv = np.array([[50.0, 50.0]])
        dets = [
            {"label": "door", "score": 0.95, "bbox": [0.0, 0.0, 100.0, 100.0]},
            {"label": "window", "score": 0.50, "bbox": [40.0, 40.0, 60.0, 60.0]},
        ]

        result = match_points_to_detections(uv, dets, pano_w=100.0)

        self.assertEqual(result.match_indices.tolist(), [1])

    def test_selects_all_points_in_clicked_detection(self):
        uv = np.array([
            [12.0, 12.0],
            [18.0, 18.0],
            [60.0, 60.0],
        ])
        dets = [{"label": "window", "score": 0.8, "bbox": [10.0, 10.0, 30.0, 30.0]}]

        selection = select_detection_region(
            clicked_idx=1,
            uv=uv,
            detections=dets,
            pano_w=100.0,
        )

        self.assertEqual(selection.detection_index, 0)
        self.assertEqual(selection.label, "window")
        self.assertEqual(selection.point_count, 2)
        self.assertEqual(selection.mask.tolist(), [True, True, False])

    def test_returns_empty_selection_when_clicked_point_misses_all_detections(self):
        uv = np.array([
            [12.0, 12.0],
            [60.0, 60.0],
        ])
        dets = [{"label": "door", "score": 0.9, "bbox": [10.0, 10.0, 30.0, 30.0]}]

        selection = select_detection_region(
            clicked_idx=1,
            uv=uv,
            detections=dets,
            pano_w=100.0,
        )

        self.assertEqual(selection.detection_index, -1)
        self.assertEqual(selection.point_count, 0)
        self.assertEqual(selection.mask.tolist(), [False, False])


if __name__ == "__main__":
    unittest.main()
