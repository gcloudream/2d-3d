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
    fit_plane_least_squares,
    refine_detection_selection,
    score_door_window_geometry,
)


def _vertical_patch(width: float, height: float, cols: int = 6, rows: int = 5) -> np.ndarray:
    xs = np.linspace(-width / 2.0, width / 2.0, cols)
    zs = np.linspace(1.0, 1.0 + height, rows)
    return np.array([[x, 0.0, z] for z in zs for x in xs], dtype=np.float64)


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

    def test_plane_fit_returns_horizontal_normal_for_vertical_patch(self):
        points = _vertical_patch(width=1.2, height=0.9)

        plane = fit_plane_least_squares(points)

        self.assertLess(abs(float(plane.normal[2])), 0.05)
        self.assertLess(plane.rms_error, 1e-9)

    def test_geometry_score_accepts_plausible_vertical_window_patch(self):
        points = _vertical_patch(width=1.2, height=0.9)

        score = score_door_window_geometry(points, "window")

        self.assertEqual(score.confidence, "high")
        self.assertEqual(score.reason, "accepted_vertical_window_geometry")
        self.assertGreater(score.width_m, 1.0)
        self.assertGreater(score.height_m, 0.8)

    def test_geometry_score_rejects_window_patch_that_is_too_small(self):
        points = _vertical_patch(width=0.12, height=0.12)

        score = score_door_window_geometry(points, "window")

        self.assertEqual(score.confidence, "low")
        self.assertEqual(score.reason, "rejected_implausible_window_size")

    def test_refine_selection_uses_geometry_reason_for_plausible_window(self):
        points = _vertical_patch(width=1.2, height=0.9)
        uv = np.array([[50.0 + p[0] * 10.0, 50.0 - p[2] * 10.0] for p in points], dtype=np.float64)
        detections = [{"label": "window", "score": 0.8, "bbox": [35.0, 20.0, 65.0, 45.0]}]

        selection = refine_detection_selection(
            points=points,
            uv=uv,
            clicked_idx=len(points) // 2,
            detections=detections,
            pano_w=100.0,
            cam_pos=np.array([0.0, -3.0, 1.4], dtype=np.float64),
            component_radius=0.35,
            depth_delta=3.0,
            min_refined_points=20,
        )

        self.assertEqual(selection.confidence, "high")
        self.assertEqual(selection.reason, "accepted_vertical_window_geometry")
        self.assertIsNotNone(selection.plane_normal)
        self.assertGreater(selection.width_m, 1.0)
        self.assertGreater(selection.height_m, 0.8)


if __name__ == "__main__":
    unittest.main()
