from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.door_window_refine import (
    DEFAULT_PLANE_DISPLAY_THICKNESS,
    clip_bbox_cone_by_seed_depth,
    connected_component_from_seed,
    estimate_seed_plane,
    filter_by_normal_consistency,
    filter_by_seed_depth,
    filter_by_seed_plane,
    fit_plane_least_squares,
    refine_detection_selection,
    remove_isolated_points,
    score_door_window_geometry,
    select_plane_band,
    should_highlight_refined_selection,
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

    def test_geometry_score_rejects_line_like_window_patch(self):
        xs = np.linspace(-0.6, 0.6, 30)
        points = np.array([[x, 0.0, 1.0 + (x + 0.6) * 0.75] for x in xs], dtype=np.float64)

        score = score_door_window_geometry(points, "window")

        self.assertEqual(score.confidence, "low")
        self.assertEqual(score.reason, "rejected_low_rectangular_coverage")

    def test_geometry_score_accepts_window_patch_with_sparse_outliers(self):
        patch = _vertical_patch(width=1.2, height=0.9)
        outliers = np.array(
            [[x, 0.7, z] for x, z in zip(np.linspace(-0.6, 0.6, 8), np.linspace(1.0, 1.9, 8))],
            dtype=np.float64,
        )

        score = score_door_window_geometry(np.vstack([patch, outliers]), "window")

        self.assertEqual(score.confidence, "high")
        self.assertEqual(score.reason, "accepted_vertical_window_geometry")
        self.assertGreater(score.inlier_ratio, 0.7)

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

    def test_refine_selection_returns_only_geometry_inliers_when_outliers_pass_depth_filter(self):
        patch = _vertical_patch(width=1.2, height=0.9)
        outliers = np.array(
            [[x, 0.7, z] for x, z in zip(np.linspace(-0.6, 0.6, 8), np.linspace(1.0, 1.9, 8))],
            dtype=np.float64,
        )
        points = np.vstack([patch, outliers])
        uv = np.array([[50.0 + p[0] * 10.0, 50.0 - p[2] * 10.0] for p in points], dtype=np.float64)
        detections = [{"label": "window", "score": 0.8, "bbox": [35.0, 20.0, 65.0, 45.0]}]

        selection = refine_detection_selection(
            points=points,
            uv=uv,
            clicked_idx=len(patch) // 2,
            detections=detections,
            pano_w=100.0,
            cam_pos=np.array([0.0, -3.0, 1.4], dtype=np.float64),
            component_radius=1.0,
            depth_delta=3.0,
            min_refined_points=20,
        )

        self.assertEqual(selection.confidence, "high")
        self.assertEqual(selection.point_count, len(patch))
        self.assertTrue(selection.refined_mask[:len(patch)].all())
        self.assertFalse(selection.refined_mask[len(patch):].any())

    def test_rejected_geometry_selection_should_not_use_normal_highlight(self):
        points = np.array([[x, 0.0, 1.0 + (x + 0.6) * 0.75] for x in np.linspace(-0.6, 0.6, 30)])
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

        self.assertEqual(selection.reason, "rejected_low_rectangular_coverage")
        self.assertFalse(should_highlight_refined_selection(selection))

    def test_select_plane_band_keeps_only_points_near_plane(self):
        plane_point = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        points = np.array([
            [0.0, 0.00, 1.0],   # on plane
            [0.5, 0.02, 1.5],   # 0.02 m off -> keep
            [0.5, 0.09, 1.5],   # 0.09 m off -> drop
            [-0.5, -0.30, 0.8],  # 0.30 m off -> drop
        ], dtype=np.float64)

        band = select_plane_band(points, plane_point, plane_normal, thickness=0.04)

        self.assertEqual(band.tolist(), [True, True, False, False])

    def test_clip_bbox_cone_removes_far_surface_behind_seed(self):
        # A near door surface (~2 m) and a far wall (~5 m) share the same bbox
        # cone; clipping around the near seed must drop the far wall.
        points = np.array([
            [0.0, 2.0, 0.0],   # seed surface
            [0.1, 2.1, 0.2],   # near
            [0.0, 5.0, 0.0],   # far wall
            [0.1, 5.1, 0.2],   # far wall
        ], dtype=np.float64)
        candidate = np.array([True, True, True, True])
        cam = np.zeros(3, dtype=np.float64)

        clipped = clip_bbox_cone_by_seed_depth(
            points, candidate, seed_idx=0, cam_pos=cam, depth_window=0.6,
        )

        self.assertEqual(clipped.tolist(), [True, True, False, False])

    def test_filter_by_seed_plane_flattens_wall_halo(self):
        # Door sheet at y=0 plus a halo 0.2 m behind it on the same wall: a
        # camera-distance shell barely separates them, the plane filter does.
        plane_point = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        plane_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        points = np.array([
            [0.0, 0.00, 1.0],   # seed, on plane
            [0.4, 0.03, 1.4],   # on door
            [0.4, 0.20, 1.4],   # halo behind door
            [-0.4, 0.25, 0.7],  # halo
        ], dtype=np.float64)
        component = np.array([True, True, True, True])

        mask = filter_by_seed_plane(
            points, component, seed_idx=0,
            plane_point=plane_point, plane_normal=plane_normal, max_delta=0.08,
        )

        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_estimate_seed_plane_returns_horizontal_normal_for_vertical_wall(self):
        patch = _vertical_patch(width=1.0, height=1.0, cols=6, rows=6)
        mask = np.ones(len(patch), dtype=bool)

        result = estimate_seed_plane(patch, mask, seed_idx=len(patch) // 2, radius=2.0)

        self.assertIsNotNone(result)
        _, normal = result
        self.assertLess(abs(float(normal[2])), 0.05)

    def test_filter_by_normal_consistency_drops_perpendicular_floor(self):
        # Vertical door wall (normal ~ +y) connected to a horizontal floor
        # (normal ~ +z). Region growth would include both; the normal filter
        # must keep the wall and drop the floor.
        wall = np.array(
            [[x, 0.0, z] for z in np.linspace(1.0, 2.0, 8) for x in np.linspace(-0.4, 0.4, 8)],
            dtype=np.float64,
        )
        floor = np.array(
            [[x, y, 1.0] for y in np.linspace(0.05, 0.6, 8) for x in np.linspace(-0.4, 0.4, 8)],
            dtype=np.float64,
        )
        points = np.vstack([wall, floor])
        component = np.ones(len(points), dtype=bool)
        ref_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        mask = filter_by_normal_consistency(
            points, component, seed_idx=len(wall) // 2,
            reference_normal=ref_normal, radius=0.3, max_angle_deg=35.0, min_neighbors=4,
        )

        # most wall points kept, most floor points dropped
        self.assertGreater(mask[:len(wall)].mean(), 0.8)
        self.assertLess(mask[len(wall):].mean(), 0.2)

    def test_remove_isolated_points_drops_disconnected_stragglers(self):
        # Dense body of points plus a few far-flung isolated stragglers.
        body = np.array(
            [[x, 0.0, z] for z in np.linspace(0.0, 0.3, 9) for x in np.linspace(0.0, 0.3, 9)],
            dtype=np.float64,
        )
        stragglers = np.array([[3.0, 0.0, 3.0], [-2.5, 0.0, 2.0], [4.0, 0.0, -1.0]], dtype=np.float64)
        points = np.vstack([body, stragglers])
        mask = np.ones(len(points), dtype=bool)

        kept = remove_isolated_points(points, mask, radius=0.1, min_neighbors=4)

        self.assertGreater(kept[: len(body)].mean(), 0.9)  # body retained
        self.assertFalse(kept[len(body):].any())           # stragglers removed

    def test_remove_isolated_points_keeps_seed_even_if_isolated(self):
        body = np.array(
            [[x, 0.0, z] for z in np.linspace(0.0, 0.3, 9) for x in np.linspace(0.0, 0.3, 9)],
            dtype=np.float64,
        )
        lone = np.array([[5.0, 0.0, 5.0]], dtype=np.float64)
        points = np.vstack([body, lone])
        mask = np.ones(len(points), dtype=bool)
        seed = len(points) - 1  # the lone point

        kept = remove_isolated_points(points, mask, seed_idx=seed, radius=0.1, min_neighbors=4)

        self.assertTrue(kept[seed])  # seed protected from removal

    def test_remove_isolated_points_returns_small_selection_unchanged(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        mask = np.array([True, True, True])

        kept = remove_isolated_points(points, mask, radius=0.1, min_neighbors=4)

        # too few points to judge density -> unchanged
        self.assertEqual(kept.tolist(), [True, True, True])

    def test_refine_selection_thins_accepted_set_to_plane_band(self):
        # A clean vertical door plane (dense) plus a sparse parallel "halo" sheet
        # 0.055 m behind it (e.g. the wall the door is flush against). The halo is
        # inside the RANSAC inlier band (0.06 m) so it survives plane fitting, but
        # the displayed highlight must be thinned to the display band (0.04 m) so
        # it does not fan out in the oblique global view.
        patch = _vertical_patch(width=0.9, height=2.0, cols=8, rows=10)
        halo = patch[::7] + np.array([0.0, 0.055, 0.0])
        points = np.vstack([patch, halo])
        uv = np.array([[50.0 + p[0] * 10.0, 50.0 - p[2] * 10.0] for p in points], dtype=np.float64)
        detections = [{"label": "door", "score": 0.9, "bbox": [20.0, 0.0, 80.0, 60.0]}]

        selection = refine_detection_selection(
            points=points,
            uv=uv,
            clicked_idx=len(patch) // 2,
            detections=detections,
            pano_w=100.0,
            cam_pos=np.array([0.0, -3.0, 1.4], dtype=np.float64),
            component_radius=1.0,
            depth_delta=3.0,
            min_refined_points=20,
        )

        self.assertEqual(selection.confidence, "high")
        highlighted = points[selection.refined_mask]
        plane_offsets = np.abs(
            (highlighted - selection.plane_point.reshape(1, 3)) @ selection.plane_normal
        )
        # Every highlighted point sits within the display band of the plane.
        self.assertTrue(np.all(plane_offsets <= DEFAULT_PLANE_DISPLAY_THICKNESS + 1e-9))
        # The sparse 0.05 m halo sheet is excluded, so the highlight is one sheet.
        self.assertLess(selection.point_count, len(points))
        self.assertEqual(selection.point_count, len(patch))


if __name__ == "__main__":
    unittest.main()
