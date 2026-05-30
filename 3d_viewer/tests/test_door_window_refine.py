from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.door_window_refine import (
    connected_component_from_seed,
    filter_by_seed_depth,
    refine_detection_selection,
)


class DoorWindowRefineTest(unittest.TestCase):
    def test_connected_component_keeps_seed_component_inside_candidate_mask(self):
        points = np.array([
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [2.00, 0.00, 0.00],
            [2.05, 0.00, 0.00],
            [5.00, 0.00, 0.00],
        ], dtype=np.float64)
        candidate = np.array([True, True, True, True, True, False])

        mask = connected_component_from_seed(
            points=points,
            candidate_mask=candidate,
            seed_idx=1,
            radius=0.11,
        )

        self.assertEqual(mask.tolist(), [True, True, True, False, False, False])

    def test_depth_filter_keeps_points_near_seed_range(self):
        points = np.array([
            [1.00, 0.00, 0.00],
            [1.05, 0.00, 0.00],
            [1.20, 0.00, 0.00],
            [1.80, 0.00, 0.00],
        ], dtype=np.float64)
        component = np.array([True, True, True, True])
        cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        mask = filter_by_seed_depth(
            points=points,
            component_mask=component,
            seed_idx=0,
            cam_pos=cam_pos,
            max_delta=0.25,
        )

        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_refine_selection_returns_refined_component_for_clicked_bbox(self):
        points = np.array([
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [2.00, 0.00, 0.00],
            [2.05, 0.00, 0.00],
            [4.00, 0.00, 0.00],
        ], dtype=np.float64)
        uv = np.array([
            [12.0, 12.0],
            [13.0, 12.0],
            [14.0, 12.0],
            [15.0, 12.0],
            [16.0, 12.0],
            [90.0, 90.0],
        ], dtype=np.float64)
        detections = [{"label": "window", "score": 0.8, "bbox": [10.0, 10.0, 20.0, 20.0]}]

        selection = refine_detection_selection(
            points=points,
            uv=uv,
            clicked_idx=1,
            detections=detections,
            pano_w=100.0,
            cam_pos=np.zeros(3, dtype=np.float64),
            component_radius=0.11,
            depth_delta=0.30,
            min_refined_points=3,
        )

        self.assertEqual(selection.detection_index, 0)
        self.assertEqual(selection.label, "window")
        self.assertEqual(selection.confidence, "medium")
        self.assertEqual(selection.reason, "seed_component_depth_filtered")
        self.assertEqual(selection.coarse_count, 5)
        self.assertEqual(selection.point_count, 3)
        self.assertEqual(selection.coarse_mask.tolist(), [True, True, True, True, True, False])
        self.assertEqual(selection.refined_mask.tolist(), [True, True, True, False, False, False])

    def test_refine_selection_reports_no_bbox_hit(self):
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        uv = np.array([[80.0, 80.0]], dtype=np.float64)
        detections = [{"label": "door", "score": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]}]

        selection = refine_detection_selection(
            points=points,
            uv=uv,
            clicked_idx=0,
            detections=detections,
            pano_w=100.0,
            cam_pos=np.zeros(3, dtype=np.float64),
        )

        self.assertEqual(selection.detection_index, -1)
        self.assertEqual(selection.confidence, "none")
        self.assertEqual(selection.reason, "no_bbox_hit")
        self.assertEqual(selection.point_count, 0)
        self.assertEqual(selection.refined_mask.tolist(), [False])

    def test_refine_selection_reports_too_few_refined_points(self):
        points = np.array([
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [2.00, 0.00, 0.00],
        ], dtype=np.float64)
        uv = np.array([
            [12.0, 12.0],
            [13.0, 12.0],
            [14.0, 12.0],
        ], dtype=np.float64)
        detections = [{"label": "door", "score": 0.7, "bbox": [10.0, 10.0, 20.0, 20.0]}]

        selection = refine_detection_selection(
            points=points,
            uv=uv,
            clicked_idx=0,
            detections=detections,
            pano_w=100.0,
            cam_pos=np.zeros(3, dtype=np.float64),
            component_radius=0.11,
            depth_delta=0.20,
            min_refined_points=3,
        )

        self.assertEqual(selection.detection_index, 0)
        self.assertEqual(selection.confidence, "low")
        self.assertEqual(selection.reason, "too_few_refined_points")
        self.assertEqual(selection.point_count, 2)
        self.assertEqual(selection.refined_mask.tolist(), [True, True, False])


if __name__ == "__main__":
    unittest.main()
