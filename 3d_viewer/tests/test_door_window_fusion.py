from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.door_window_fusion import (
    FusedSelection,
    FrustumProjectionCache,
    extract_detection_region_from_bbox,
    fuse_detection_and_pointcloud,
    frustum_candidate_mask,
    should_highlight_fused,
)
from core.pointcloud_extract import PlanarRegionSelection
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

    def test_extract_detection_region_from_bbox_prefers_region_near_clicked_uv(self):
        left = _vertical_patch(width=0.7, height=1.0, y=2.0)
        left[:, 0] -= 0.8
        right = _vertical_patch(width=0.7, height=1.0, y=2.0)
        right[:, 0] += 0.8
        points = np.vstack([left, right])
        uv = project_points_to_panorama(
            points, self.cam, self.R, self.img_w, self.img_h, yaw_offset_deg=0.0
        )
        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)
        detection = {
            "label": "window",
            "score": 0.95,
            "bbox": [x1 - 4.0, y1 - 4.0, x2 + 4.0, y2 + 4.0],
        }
        left_uv = tuple(np.median(uv[: len(left)], axis=0))
        right_uv = tuple(np.median(uv[len(left):], axis=0))

        selected_left = extract_detection_region_from_bbox(
            points,
            0,
            [detection],
            self.cam,
            self.R,
            self.img_w,
            self.img_h,
            yaw_offset_deg=0.0,
            click_uv=left_uv,
            max_seed_count=6,
        )
        selected_right = extract_detection_region_from_bbox(
            points,
            0,
            [detection],
            self.cam,
            self.R,
            self.img_w,
            self.img_h,
            yaw_offset_deg=0.0,
            click_uv=right_uv,
            max_seed_count=6,
        )

        self.assertGreater(selected_left.mask[: len(left)].sum(), selected_left.mask[len(left):].sum())
        self.assertGreater(selected_right.mask[len(left):].sum(), selected_right.mask[: len(left)].sum())

    def test_extract_detection_region_from_bbox_reuses_projection_for_auto_seeds(self):
        with patch(
            "core.door_window_fusion.project_points_to_panorama",
            wraps=project_points_to_panorama,
        ) as project:
            fused = extract_detection_region_from_bbox(
                self.points,
                0,
                [self.detection],
                self.cam,
                self.R,
                self.img_w,
                self.img_h,
                yaw_offset_deg=0.0,
                max_seed_count=4,
            )

        self.assertGreater(fused.point_count, 0)
        self.assertEqual(project.call_count, 1)

    def test_extract_detection_region_from_bbox_reuses_shared_projection_cache(self):
        cache = FrustumProjectionCache()
        with patch(
            "core.door_window_fusion.project_points_to_panorama",
            wraps=project_points_to_panorama,
        ) as project:
            first = extract_detection_region_from_bbox(
                self.points,
                0,
                [self.detection],
                self.cam,
                self.R,
                self.img_w,
                self.img_h,
                yaw_offset_deg=0.0,
                max_seed_count=4,
                cache=cache,
            )
            second = extract_detection_region_from_bbox(
                self.points,
                0,
                [self.detection],
                self.cam,
                self.R,
                self.img_w,
                self.img_h,
                yaw_offset_deg=0.0,
                max_seed_count=4,
                cache=cache,
            )

        self.assertGreater(first.point_count, 0)
        self.assertGreater(second.point_count, 0)
        self.assertEqual(project.call_count, 1)
        self.assertTrue(second.diagnostics["projection_cache_hit"])
        self.assertTrue(second.diagnostics["candidate_cache_hit"])

    def test_projection_cache_key_distinguishes_array_strides(self):
        cache = FrustumProjectionCache()
        points = np.arange(9, dtype=np.float64).reshape(3, 3)
        transposed = points.T

        with patch(
            "core.door_window_fusion.project_points_to_panorama",
            wraps=project_points_to_panorama,
        ) as project:
            cache.project(points, self.cam, self.R, self.img_w, self.img_h, 0.0)
            cache.project(transposed, self.cam, self.R, self.img_w, self.img_h, 0.0)

        self.assertEqual(project.call_count, 2)

    def test_fusion_seed_path_projects_points_once(self):
        with patch(
            "core.door_window_fusion.project_points_to_panorama",
            wraps=project_points_to_panorama,
        ) as project:
            fused = fuse_detection_and_pointcloud(
                self.points,
                len(self.door) // 2,
                [self.detection],
                self.cam,
                self.R,
                self.img_w,
                self.img_h,
                yaw_offset_deg=0.0,
            )

        self.assertGreater(fused.point_count, 0)
        self.assertEqual(project.call_count, 1)

    def test_extract_detection_region_from_bbox_reports_diagnostics(self):
        with self.assertLogs("3d_viewer.core.door_window_fusion", level="INFO") as logs:
            fused = extract_detection_region_from_bbox(
                self.points,
                0,
                [self.detection],
                self.cam,
                self.R,
                self.img_w,
                self.img_h,
                yaw_offset_deg=0.0,
                max_seed_count=5,
            )

        diagnostics = fused.diagnostics
        self.assertIsInstance(diagnostics, dict)
        self.assertGreater(diagnostics["bbox_candidate_count"], 0)
        self.assertGreater(diagnostics["selected_depth_candidate_count"], 0)
        self.assertGreaterEqual(diagnostics["seed_attempt_count"], 1)
        self.assertIn("total_ms", diagnostics)
        self.assertIn("extract_detection_region_from_bbox", "\n".join(logs.output))

    def test_extract_detection_region_from_bbox_exposes_debug_masks(self):
        fused = extract_detection_region_from_bbox(
            self.points,
            0,
            [self.detection],
            self.cam,
            self.R,
            self.img_w,
            self.img_h,
            yaw_offset_deg=0.0,
            max_seed_count=5,
        )

        self.assertIsInstance(fused.debug_masks, dict)
        for key in ("bbox", "depth", "final"):
            self.assertIn(key, fused.debug_masks)
            self.assertEqual(len(fused.debug_masks[key]), len(self.points))
        self.assertEqual(int(fused.debug_masks["bbox"].sum()), fused.diagnostics["bbox_candidate_count"])
        self.assertEqual(int(fused.debug_masks["depth"].sum()), fused.diagnostics["selected_depth_candidate_count"])
        np.testing.assert_array_equal(fused.debug_masks["final"], fused.mask)

    def test_extract_detection_region_from_bbox_clips_final_region_to_bbox(self):
        leaked_region = PlanarRegionSelection(
            mask=np.ones(len(self.points), dtype=bool),
            point_count=len(self.points),
            confidence="high",
            reason="accepted_vertical_planar_region",
            label="door",
            plane_point=np.array([0.0, 2.0, 1.0], dtype=np.float64),
            plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            width_m=0.9,
            height_m=2.0,
        )
        expected = frustum_candidate_mask(
            self.points,
            self.cam,
            self.R,
            self.img_w,
            self.img_h,
            self.detection,
            yaw_offset_deg=0.0,
            depth_window=None,
        )

        with patch("core.door_window_fusion.extract_planar_region_from_seed", return_value=leaked_region):
            fused = extract_detection_region_from_bbox(
                self.points,
                0,
                [self.detection],
                self.cam,
                self.R,
                self.img_w,
                self.img_h,
                yaw_offset_deg=0.0,
                max_seed_count=3,
            )

        np.testing.assert_array_equal(fused.mask, expected)
        self.assertEqual(fused.point_count, int(expected.sum()))

    def test_extract_detection_region_from_bbox_prefers_bbox_center_when_no_click_uv(self):
        edge = _vertical_patch(width=0.2, height=1.0, y=2.0, cols=5, rows=10)
        edge[:, 0] -= 1.2
        center = _vertical_patch(width=0.2, height=1.0, y=2.0, cols=5, rows=10)
        points = np.vstack([edge, center])
        uv = project_points_to_panorama(
            points, self.cam, self.R, self.img_w, self.img_h, yaw_offset_deg=0.0
        )
        detection = {
            "label": "window",
            "score": 0.95,
            "bbox": [250.0, float(uv[:, 1].min() - 4.0), 750.0, float(uv[:, 1].max() + 4.0)],
        }
        edge_mask = np.zeros(len(points), dtype=bool)
        edge_mask[: len(edge)] = True
        center_mask = np.zeros(len(points), dtype=bool)
        center_mask[len(edge):] = True

        def fake_extract(_points, seed_idx, **_kwargs):
            is_edge_seed = int(seed_idx) < len(edge)
            mask = edge_mask if is_edge_seed else center_mask
            return PlanarRegionSelection(
                mask=mask,
                point_count=int(mask.sum()),
                confidence="high",
                reason="accepted_vertical_planar_region",
                label="window",
                plane_point=np.array([0.0, 2.0, 1.0], dtype=np.float64),
                plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                width_m=0.2,
                height_m=1.0,
            )

        with patch("core.door_window_fusion.extract_planar_region_from_seed", side_effect=fake_extract):
            fused = extract_detection_region_from_bbox(
                points,
                0,
                [detection],
                self.cam,
                self.R,
                self.img_w,
                self.img_h,
                yaw_offset_deg=0.0,
                max_seed_count=len(points),
            )

        self.assertGreater(fused.mask[len(edge):].sum(), fused.mask[: len(edge)].sum())

    def test_extract_detection_region_from_bbox_prefers_bbox_center_over_point_count(self):
        edge = _vertical_patch(width=0.2, height=1.0, y=2.0, cols=18, rows=18)
        edge[:, 0] -= 1.2
        center = _vertical_patch(width=0.2, height=1.0, y=2.0, cols=5, rows=6)
        points = np.vstack([edge, center])
        uv = project_points_to_panorama(
            points, self.cam, self.R, self.img_w, self.img_h, yaw_offset_deg=0.0
        )
        detection = {
            "label": "window",
            "score": 0.95,
            "bbox": [250.0, float(uv[:, 1].min() - 4.0), 750.0, float(uv[:, 1].max() + 4.0)],
        }
        edge_mask = np.zeros(len(points), dtype=bool)
        edge_mask[: len(edge)] = True
        center_mask = np.zeros(len(points), dtype=bool)
        center_mask[len(edge):] = True

        def fake_extract(_points, seed_idx, **_kwargs):
            is_edge_seed = int(seed_idx) < len(edge)
            mask = edge_mask if is_edge_seed else center_mask
            return PlanarRegionSelection(
                mask=mask,
                point_count=int(mask.sum()),
                confidence="high",
                reason="accepted_vertical_planar_region",
                label="window",
                plane_point=np.array([0.0, 2.0, 1.0], dtype=np.float64),
                plane_normal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                width_m=0.2,
                height_m=1.0,
            )

        with patch("core.door_window_fusion.extract_planar_region_from_seed", side_effect=fake_extract):
            fused = extract_detection_region_from_bbox(
                points,
                0,
                [detection],
                self.cam,
                self.R,
                self.img_w,
                self.img_h,
                yaw_offset_deg=0.0,
                max_seed_count=len(points),
            )

        self.assertGreater(fused.mask[len(edge):].sum(), fused.mask[: len(edge)].sum())

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

    def test_frustum_only_rejected_geometry_is_not_highlighted(self):
        fused = extract_detection_region_from_bbox(
            self.points,
            0,
            [self.detection],
            self.cam,
            self.R,
            self.img_w,
            self.img_h,
            yaw_offset_deg=0.0,
            max_seed_count=1,
        )
        rejected = PlanarRegionSelection(
            mask=fused.mask,
            point_count=max(1, fused.point_count),
            confidence="low",
            reason="rejected_not_vertical_plane",
            label="door",
            plane_point=np.array([0.0, 2.0, 1.0], dtype=np.float64),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            width_m=0.9,
            height_m=2.0,
        )

        with patch("core.door_window_fusion.extract_planar_region_from_seed", return_value=rejected):
            selection = extract_detection_region_from_bbox(
                self.points,
                0,
                [self.detection],
                self.cam,
                self.R,
                self.img_w,
                self.img_h,
                yaw_offset_deg=0.0,
                max_seed_count=1,
            )

        self.assertEqual(selection.reason, "frustum_only_rejected_not_vertical_plane")
        self.assertFalse(should_highlight_fused(selection))

    def test_frustum_only_incomplete_geometry_is_not_highlighted(self):
        selection = FusedSelection(
            mask=np.asarray([True, False, True]),
            point_count=2,
            confidence="medium",
            reason="frustum_only_too_few_region_points",
            label="window",
            source="fused",
            detection_index=0,
            score=0.9,
            plane_point=None,
            plane_normal=None,
            width_m=None,
            height_m=None,
        )

        self.assertFalse(should_highlight_fused(selection))

    def test_fusion_seed_out_of_range(self):
        fused = fuse_detection_and_pointcloud(
            self.points, 10_000, [self.detection], self.cam, self.R,
            self.img_w, self.img_h, yaw_offset_deg=0.0,
        )
        self.assertEqual(fused.reason, "seed_out_of_range")


if __name__ == "__main__":
    unittest.main()
