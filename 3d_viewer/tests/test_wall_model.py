from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.wall_openings import wall_opening_events_path
from core.wall_model import (
    WallSegment,
    complete_boundary_corner_gaps,
    complete_parallel_return_bridges,
    extract_axis_aligned_wall_segments,
    extract_boundary_outline_wall_segments,
    extract_contour_wall_segments,
    filter_wall_segments_by_network,
    generate_wall_model,
    recover_short_return_wall_segments,
    snap_wall_segment_endpoints,
    verify_wall_segments_with_points,
    wall_mask_world_bounds,
    wall_segments_to_mesh,
)


class WallModelTest(unittest.TestCase):
    def test_extracts_axis_aligned_segments_from_mask(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[4:34, 8] = True
        mask[20, 8:32] = True

        segments = extract_axis_aligned_wall_segments(
            mask,
            x_min=0.0,
            y_min=0.0,
            resolution_m=0.1,
            min_wall_length_m=1.5,
        )

        orientations = {segment.orientation for segment in segments}
        self.assertIn("vertical", orientations)
        self.assertIn("horizontal", orientations)

    def test_does_not_turn_filled_interior_blob_into_center_wall(self):
        mask = np.zeros((80, 80), dtype=bool)
        mask[8:72, 8] = True
        mask[8:72, 71] = True
        mask[8, 8:72] = True
        mask[71, 8:72] = True
        mask[26:58, 34:50] = True

        segments = extract_axis_aligned_wall_segments(
            mask,
            x_min=0.0,
            y_min=0.0,
            resolution_m=0.1,
            min_wall_length_m=2.0,
        )

        interior = [
            segment
            for segment in segments
            if 3.0 <= segment.x1 <= 5.2
            and 2.4 <= segment.y1 <= 6.0
            and 2.4 <= segment.y2 <= 6.0
        ]
        self.assertEqual(interior, [])

    def test_merges_parallel_ridges_from_same_wall_thickness(self):
        mask = np.zeros((50, 80), dtype=bool)
        mask[20, 10:70] = True
        mask[23, 10:70] = True

        segments = extract_axis_aligned_wall_segments(
            mask,
            x_min=0.0,
            y_min=0.0,
            resolution_m=0.1,
            min_wall_length_m=2.0,
        )

        horizontal = [segment for segment in segments if segment.orientation == "horizontal"]
        self.assertEqual(len(horizontal), 1)

    def test_extracts_short_wall_candidate_when_requested(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[6:19, 10] = True

        segments = extract_axis_aligned_wall_segments(
            mask,
            x_min=0.0,
            y_min=0.0,
            resolution_m=0.1,
            min_wall_length_m=1.0,
        )

        self.assertEqual(len(segments), 1)
        self.assertGreaterEqual(segments[0].length_m, 1.0)

    def test_boundary_outline_extracts_edges_from_filled_wall_evidence(self):
        mask = np.zeros((70, 90), dtype=bool)
        mask[20:45, 15:65] = True

        segments = extract_boundary_outline_wall_segments(
            mask,
            x_min=0.0,
            y_min=0.0,
            resolution_m=0.1,
            min_wall_length_m=2.0,
        )

        horizontal_ys = [segment.y1 for segment in segments if segment.orientation == "horizontal"]
        self.assertTrue(any(abs(y - 2.05) < 1e-6 for y in horizontal_ys))
        self.assertTrue(any(abs(y - 4.45) < 1e-6 for y in horizontal_ys))
        self.assertFalse(any(abs(y - 3.2) < 0.2 for y in horizontal_ys))

    def test_contour_extraction_keeps_close_parallel_notch_edge(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:30, 25] = True
        mask[18:25, 22] = True

        segments = extract_contour_wall_segments(
            mask,
            x_min=0.0,
            y_min=0.0,
            resolution_m=0.1,
            min_wall_length_m=0.45,
        )

        short_edges = [
            segment
            for segment in segments
            if segment.orientation == "vertical"
            and abs(segment.x1 - 2.25) < 1e-6
            and segment.length_m < 1.0
        ]
        self.assertEqual(len(short_edges), 1)

    def test_filters_isolated_interior_short_segments_from_wall_network(self):
        main_wall = WallSegment("horizontal", 0.0, 3.0, 6.0, 3.0, 0.0, 2.6, 6.0, 300, 2.6)
        boundary_wall = WallSegment("vertical", 0.0, 0.0, 0.0, 6.0, 0.0, 2.6, 6.0, 300, 2.6)
        connected_partition = WallSegment("vertical", 2.5, 1.8, 2.5, 3.05, 0.0, 2.6, 1.25, 140, 2.6)
        furniture_line = WallSegment("horizontal", 2.0, 1.2, 3.6, 1.2, 0.0, 2.6, 1.6, 120, 2.6)

        filtered = filter_wall_segments_by_network(
            [main_wall, boundary_wall, connected_partition, furniture_line],
            x_min=0.0,
            x_max=6.0,
            y_min=0.0,
            y_max=6.0,
            boundary_tolerance_m=0.25,
            connection_tolerance_m=0.25,
            min_major_wall_length_m=4.0,
            min_connected_wall_length_m=1.0,
        )

        self.assertIn(main_wall, filtered)
        self.assertIn(boundary_wall, filtered)
        self.assertIn(connected_partition, filtered)
        self.assertNotIn(furniture_line, filtered)

    def test_keeps_short_partition_across_small_scan_gap(self):
        main_wall = WallSegment("horizontal", 0.0, 3.0, 6.0, 3.0, 0.0, 2.6, 6.0, 300, 2.6)
        gapped_partition = WallSegment("vertical", 2.5, 1.35, 2.5, 2.4, 0.0, 2.6, 1.05, 120, 2.6)
        furniture_line = WallSegment("horizontal", 2.0, 1.0, 3.2, 1.0, 0.0, 2.6, 1.2, 120, 2.6)

        filtered = filter_wall_segments_by_network(
            [main_wall, gapped_partition, furniture_line],
            x_min=0.0,
            x_max=6.0,
            y_min=0.0,
            y_max=6.0,
        )

        self.assertIn(gapped_partition, filtered)
        self.assertNotIn(furniture_line, filtered)

    def test_snaps_short_partition_endpoint_to_anchor_wall(self):
        main_wall = WallSegment("horizontal", 0.0, 3.0, 6.0, 3.0, 0.0, 2.6, 6.0, 300, 2.6)
        gapped_partition = WallSegment("vertical", 2.5, 1.35, 2.5, 2.4, 0.0, 2.6, 1.05, 120, 2.6)

        snapped = snap_wall_segment_endpoints([main_wall, gapped_partition], tolerance_m=0.65)

        partition = next(segment for segment in snapped if segment.orientation == "vertical")
        self.assertEqual(partition.y2, 3.0)
        self.assertAlmostEqual(partition.length_m, 1.65)

    def test_recovers_short_return_wall_but_rejects_isolated_clutter(self):
        main_wall = WallSegment("horizontal", 0.0, 3.0, 6.0, 3.0, 0.0, 2.6, 6.0, 300, 2.6)
        existing_return = WallSegment("vertical", 3.0, 1.2, 3.0, 3.0, 0.0, 2.6, 1.8, 450, 2.6)
        return_wall = WallSegment("vertical", 2.0, 1.2, 2.0, 2.95, 0.0, 2.6, 1.75, 450, 2.6)
        duplicate = WallSegment("horizontal", 0.1, 3.05, 5.9, 3.05, 0.0, 2.6, 5.8, 500, 2.6)
        clutter = WallSegment("vertical", 4.0, 0.6, 4.0, 1.8, 0.0, 2.6, 1.2, 450, 2.6)

        recovered = recover_short_return_wall_segments(
            [return_wall, duplicate, clutter],
            [main_wall, existing_return],
            x_min=0.0,
            x_max=6.0,
            y_min=0.0,
            y_max=6.0,
        )

        self.assertEqual(recovered, [return_wall])

    def test_recovers_tiny_boundary_notch_return_only_near_boundary_anchor(self):
        boundary_wall = WallSegment("vertical", 6.0, 0.0, 6.0, 3.0, 0.0, 2.6, 3.0, 500, 2.6)
        notch_return = WallSegment("vertical", 5.4, 2.8, 5.4, 3.35, 0.0, 2.6, 0.55, 260, 2.0)
        interior_short = WallSegment("vertical", 3.0, 2.8, 3.0, 3.35, 0.0, 2.6, 0.55, 260, 2.0)

        recovered = recover_short_return_wall_segments(
            [notch_return, interior_short],
            [boundary_wall],
            x_min=0.0,
            x_max=6.0,
            y_min=0.0,
            y_max=6.0,
        )

        self.assertEqual(recovered, [notch_return])

    def test_rejects_tiny_boundary_noise_along_continuous_anchor_span(self):
        boundary_wall = WallSegment("horizontal", 0.0, 0.0, 6.0, 0.0, 0.0, 2.6, 6.0, 500, 2.6)
        mid_span_noise = WallSegment("horizontal", 2.0, 0.5, 2.6, 0.5, 0.0, 2.6, 0.6, 260, 2.0)

        recovered = recover_short_return_wall_segments(
            [mid_span_noise],
            [boundary_wall],
            x_min=0.0,
            x_max=6.0,
            y_min=0.0,
            y_max=6.0,
        )

        self.assertEqual(recovered, [])

    def test_completes_bridge_between_parallel_return_wall_ends(self):
        left = WallSegment("vertical", 1.0, 1.0, 1.0, 3.0, 0.0, 2.6, 2.0, 400, 2.6)
        right = WallSegment("vertical", 2.0, 1.1, 2.0, 3.0, 0.0, 2.6, 1.9, 380, 2.6)
        main = WallSegment("horizontal", 0.0, 3.0, 4.0, 3.0, 0.0, 2.6, 4.0, 800, 2.6)

        completed = complete_parallel_return_bridges([main, left, right])

        bridges = [
            segment
            for segment in completed
            if segment.orientation == "horizontal"
            and abs(segment.y1 - 1.05) < 1e-6
            and abs(segment.x1 - 1.0) < 1e-6
            and abs(segment.x2 - 2.0) < 1e-6
        ]
        self.assertEqual(len(bridges), 1)

    def test_completes_boundary_corner_gap_when_points_support_wall(self):
        bottom = WallSegment("horizontal", 0.0, 0.0, 4.0, 0.0, 0.0, 2.6, 4.0, 500, 2.6)
        right = WallSegment("vertical", 4.0, 1.4, 4.0, 6.0, 0.0, 2.6, 4.6, 500, 2.6)
        points = np.asarray(
            [[4.0, y, z] for y in np.linspace(0.0, 1.4, 18) for z in np.linspace(0.0, 2.4, 8)],
            dtype=np.float64,
        )

        completed = complete_boundary_corner_gaps(
            [bottom, right],
            points,
            x_min=0.0,
            x_max=4.0,
            y_min=0.0,
            y_max=6.0,
            fallback_z_min=0.0,
            fallback_z_max=2.4,
            strip_width_m=0.12,
            min_point_count=50,
            min_height_span_m=1.0,
        )

        gap_segments = [
            segment
            for segment in completed
            if segment.orientation == "vertical"
            and abs(segment.x1 - 4.0) < 1e-6
            and abs(segment.y1 - 0.0) < 1e-6
            and abs(segment.y2 - 1.4) < 1e-6
        ]
        self.assertEqual(len(gap_segments), 1)

    def test_completes_boundary_corner_gap_at_supported_offset_endpoint(self):
        bottom = WallSegment("horizontal", 0.0, 0.0, 3.8, 0.0, 0.0, 2.6, 3.8, 500, 2.6)
        right = WallSegment("vertical", 4.0, 1.4, 4.0, 6.0, 0.0, 2.6, 4.6, 500, 2.6)
        points = np.asarray(
            [[3.8, y, z] for y in np.linspace(0.0, 1.4, 18) for z in np.linspace(0.0, 2.4, 8)],
            dtype=np.float64,
        )

        completed = complete_boundary_corner_gaps(
            [bottom, right],
            points,
            x_min=0.0,
            x_max=4.0,
            y_min=0.0,
            y_max=6.0,
            fallback_z_min=0.0,
            fallback_z_max=2.4,
            strip_width_m=0.12,
            min_point_count=50,
            min_height_span_m=1.0,
        )

        gap_segments = [
            segment
            for segment in completed
            if segment.orientation == "vertical"
            and abs(segment.x1 - 3.8) < 1e-6
            and abs(segment.y1 - 0.0) < 1e-6
            and abs(segment.y2 - 1.4) < 1e-6
        ]
        self.assertEqual(len(gap_segments), 1)

    def test_verify_segments_can_keep_all_contour_candidates(self):
        segments = []
        points = []
        for idx in range(82):
            x = idx * 0.2
            segments.append(WallSegment("vertical", x, 0.0, x, 1.0, 0.0, 2.0, 1.0, 0, 0.0))
            for y in np.linspace(0.0, 1.0, 5):
                for z in np.linspace(0.0, 2.0, 18):
                    points.append([x, y, z])

        verified = verify_wall_segments_with_points(
            segments,
            np.asarray(points, dtype=np.float64),
            fallback_z_min=0.0,
            fallback_z_max=2.0,
            strip_width_m=0.05,
            max_segments=None,
        )

        self.assertEqual(len(verified), 82)

    def test_wall_mask_bounds_use_preserved_wall_evidence_not_sparse_cloud_bounds(self):
        mask = np.zeros((100, 120), dtype=bool)
        mask[10:70, 20] = True
        mask[10:70, 90] = True
        mask[10, 20:91] = True
        mask[69, 20:91] = True

        bounds = wall_mask_world_bounds(mask, x_min=-10.0, y_min=-20.0, resolution_m=0.1)

        self.assertEqual(bounds, (-7.95, -0.95, -18.95, -13.05))

    def test_converts_wall_segment_to_box_mesh(self):
        segment = WallSegment(
            "horizontal",
            0.0,
            1.0,
            2.0,
            1.0,
            0.0,
            3.0,
            2.0,
            100,
            3.0,
        )

        vertices, faces = wall_segments_to_mesh([segment], wall_thickness_m=0.2)

        self.assertEqual(len(vertices), 8)
        self.assertEqual(len(faces), 6)

    def test_matches_wall_opening_to_vertical_segment(self):
        from core.wall_openings import WallOpening
        from core.wall_model import match_wall_openings_to_segments

        wall = WallSegment("vertical", 1.0, 0.0, 1.0, 3.0, 0.0, 2.6, 3.0, 100, 2.6)
        opening = WallOpening(
            id="window-0001",
            label="window",
            source_image="608.jpg",
            seed_index=1,
            point_count=20,
            confidence="high",
            reason="accepted_vertical_planar_region",
            center=(1.05, 1.2, 1.2),
            normal=(1.0, 0.0, 0.0),
            bbox_min=(1.02, 0.8, 0.9),
            bbox_max=(1.08, 1.6, 1.5),
            width_m=0.8,
            height_m=0.6,
            z_min=0.9,
            z_max=1.5,
            detection_index=-1,
            score=None,
        )

        matched, unmatched = match_wall_openings_to_segments([opening], [wall])

        self.assertEqual(len(matched), 1)
        self.assertEqual(unmatched, [])
        self.assertEqual(matched[0].opening_id, "window-0001")
        self.assertEqual(matched[0].segment_index, 0)
        self.assertEqual(matched[0].axis_min, 0.8)
        self.assertEqual(matched[0].axis_max, 1.6)

    def test_rejects_opening_far_from_wall_segment(self):
        from core.wall_openings import WallOpening
        from core.wall_model import match_wall_openings_to_segments

        wall = WallSegment("horizontal", 0.0, 0.0, 3.0, 0.0, 0.0, 2.6, 3.0, 100, 2.6)
        opening = WallOpening(
            id="door-0001",
            label="door",
            source_image="608.jpg",
            seed_index=1,
            point_count=20,
            confidence="high",
            reason="accepted_vertical_planar_region",
            center=(1.0, 2.0, 1.0),
            normal=(0.0, 1.0, 0.0),
            bbox_min=(0.6, 1.9, 0.0),
            bbox_max=(1.4, 2.1, 2.0),
            width_m=0.8,
            height_m=2.0,
            z_min=0.0,
            z_max=2.0,
            detection_index=-1,
            score=None,
        )

        matched, unmatched = match_wall_openings_to_segments(
            [opening],
            [wall],
            max_plane_distance_m=0.25,
        )

        self.assertEqual(matched, [])
        self.assertEqual(unmatched[0]["id"], "door-0001")
        self.assertEqual(unmatched[0]["reason"], "no_matching_wall_segment")

    def test_topdown_preview_draws_opening_marker(self):
        from core.wall_model import WallOpeningMarker, render_wall_model_topdown_preview

        grid = np.zeros((80, 80), dtype=np.uint8)
        wall_mask = np.zeros_like(grid, dtype=bool)
        segment = WallSegment("vertical", 1.0, 0.0, 1.0, 3.0, 0.0, 2.6, 3.0, 100, 2.6)
        marker = WallOpeningMarker(
            opening_id="window-0001",
            label="window",
            segment_index=0,
            orientation="vertical",
            wall_coord=1.0,
            axis_min=1.0,
            axis_max=1.5,
            z_min=0.9,
            z_max=1.5,
            side=1.0,
        )

        image = render_wall_model_topdown_preview(
            grid,
            wall_mask,
            [segment],
            x_min=0.0,
            y_min=0.0,
            resolution_m=0.05,
            opening_markers=[marker],
        )

        pixels = np.asarray(image)
        green_pixels = (pixels[:, :, 1] > 180) & (pixels[:, :, 0] < 80)
        self.assertTrue(green_pixels.any())

    def test_topdown_preview_draws_projected_unmatched_opening(self):
        from core.wall_openings import WallOpening
        from core.wall_model import render_wall_model_topdown_preview

        grid = np.zeros((80, 80), dtype=np.uint8)
        wall_mask = np.zeros_like(grid, dtype=bool)
        opening = WallOpening(
            id="window-0001",
            label="window",
            source_image="608.jpg",
            seed_index=1,
            point_count=20,
            confidence="medium",
            reason="frustum_only_rejected_not_vertical_plane",
            center=(1.2, 1.2, 0.5),
            normal=(1.0, 0.0, 0.0),
            bbox_min=(1.0, 1.0, 0.2),
            bbox_max=(1.6, 1.5, 0.8),
            width_m=0.6,
            height_m=0.6,
            z_min=0.2,
            z_max=0.8,
            detection_index=-1,
            score=None,
        )

        image = render_wall_model_topdown_preview(
            grid,
            wall_mask,
            [],
            x_min=0.0,
            y_min=0.0,
            resolution_m=0.05,
            projected_openings=[opening],
        )

        pixels = np.asarray(image)
        orange_pixels = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] > 90) & (pixels[:, :, 1] < 190)
        self.assertTrue(orange_pixels.any())

    def test_opening_marker_frames_create_extra_mesh(self):
        from core.wall_model import WallOpeningMarker, opening_markers_to_mesh

        marker = WallOpeningMarker(
            opening_id="door-0001",
            label="door",
            segment_index=0,
            orientation="horizontal",
            wall_coord=2.0,
            axis_min=0.5,
            axis_max=1.3,
            z_min=0.0,
            z_max=2.1,
            side=1.0,
        )

        vertices, faces = opening_markers_to_mesh([marker], wall_thickness_m=0.08)

        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)
        self.assertEqual(len(vertices) % 8, 0)
        self.assertEqual(len(faces) % 6, 0)

    def test_generates_wall_model_artifacts_for_synthetic_room(self):
        xs = np.linspace(0.0, 4.0, 45)
        ys = np.linspace(0.0, 3.0, 35)
        zs = np.linspace(0.0, 2.4, 12)
        points = []
        for z in zs:
            for x in xs:
                points.append([x, 0.0, z])
                points.append([x, 3.0, z])
            for y in ys:
                points.append([0.0, y, z])
                points.append([4.0, y, z])
        points = np.asarray(points, dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "synthetic_room"
            data_root.mkdir()

            result = generate_wall_model(
                workspace,
                data_root,
                points,
                resolution_m=0.1,
                min_wall_length_m=1.5,
            )

            self.assertTrue(result.obj_path.exists())
            self.assertTrue(result.metadata_path.exists())
            self.assertTrue(result.preview_path.exists())
            self.assertGreaterEqual(result.segment_count, 4)

    def test_generate_wall_model_includes_opening_metadata(self):
        from core.wall_openings import WallOpening

        xs = np.linspace(0.0, 4.0, 45)
        ys = np.linspace(0.0, 3.0, 35)
        zs = np.linspace(0.0, 2.4, 12)
        points = []
        for z in zs:
            for x in xs:
                points.append([x, 0.0, z])
                points.append([x, 3.0, z])
            for y in ys:
                points.append([0.0, y, z])
                points.append([4.0, y, z])
        points = np.asarray(points, dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            opening = WallOpening(
                id="window-0001",
                label="window",
                source_image="608.jpg",
                seed_index=1,
                point_count=20,
                confidence="high",
                reason="accepted_vertical_planar_region",
                center=(0.02, 2.0, 1.2),
                normal=(1.0, 0.0, 0.0),
                bbox_min=(0.0, 1.6, 0.9),
                bbox_max=(0.08, 2.4, 1.5),
                width_m=0.8,
                height_m=0.6,
                z_min=0.9,
                z_max=1.5,
                detection_index=-1,
                score=None,
            )

            result = generate_wall_model(
                workspace,
                data_root,
                points,
                openings=[opening],
                resolution_m=0.1,
            )

            payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["matched_opening_count"], 1)
            self.assertEqual(payload["unmatched_opening_count"], 0)
            self.assertEqual(payload["projected_opening_count"], 0)
            self.assertEqual(result.projected_opening_count, 0)
            self.assertEqual(payload["opening_markers"][0]["opening_id"], "window-0001")
            obj_text = result.obj_path.read_text(encoding="utf-8")
            mtl_path = result.obj_path.with_suffix(".mtl")
            self.assertTrue(mtl_path.exists())
            self.assertIn(f"mtllib {mtl_path.name}", obj_text)
            self.assertIn("usemtl wall", obj_text)
            self.assertIn("usemtl opening_window", obj_text)
            mtl_text = mtl_path.read_text(encoding="utf-8")
            self.assertIn("newmtl wall", mtl_text)
            self.assertIn("newmtl opening_window", mtl_text)

    def test_obj_loader_uses_material_colors_for_opening_faces(self):
        from wall_model_tool import OBJ_MATERIAL_COLORS, load_obj_triangle_mesh

        with tempfile.TemporaryDirectory() as tmp:
            obj_path = Path(tmp) / "colored.obj"
            obj_path.write_text(
                "\n".join(
                    [
                        "v 0 0 0",
                        "v 1 0 0",
                        "v 1 1 0",
                        "v 0 1 0",
                        "v 0 0 1",
                        "v 1 0 1",
                        "v 1 1 1",
                        "v 0 1 1",
                        "usemtl wall",
                        "f 1 2 3 4",
                        "usemtl opening_window",
                        "f 5 6 7 8",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            mesh = load_obj_triangle_mesh(obj_path)

            self.assertEqual(mesh.triangles.shape, (12, 3))
            self.assertEqual(mesh.colors.shape, (12, 4))
            self.assertTrue(np.allclose(mesh.colors[0], OBJ_MATERIAL_COLORS["wall"]))
            self.assertTrue(np.allclose(mesh.colors[-1], OBJ_MATERIAL_COLORS["opening_window"]))
            self.assertFalse(np.allclose(mesh.colors[0], mesh.colors[-1]))

    def test_generate_wall_model_projects_reliable_unmatched_opening_metadata(self):
        from core.wall_openings import WallOpening

        xs = np.linspace(0.0, 4.0, 45)
        ys = np.linspace(0.0, 3.0, 35)
        zs = np.linspace(0.0, 2.4, 12)
        points = []
        for z in zs:
            for x in xs:
                points.append([x, 0.0, z])
                points.append([x, 3.0, z])
            for y in ys:
                points.append([0.0, y, z])
                points.append([4.0, y, z])
        points = np.asarray(points, dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            opening = WallOpening(
                id="window-0001",
                label="window",
                source_image="608.jpg",
                seed_index=1,
                point_count=20,
                confidence="high",
                reason="fused_frustum_and_planar_geometry",
                center=(2.0, 1.5, 0.5),
                normal=(1.0, 0.0, 0.0),
                bbox_min=(1.8, 1.3, 0.2),
                bbox_max=(2.2, 1.7, 0.8),
                width_m=0.4,
                height_m=0.6,
                z_min=0.2,
                z_max=0.8,
                detection_index=-1,
                score=None,
            )

            result = generate_wall_model(
                workspace,
                data_root,
                points,
                openings=[opening],
                resolution_m=0.1,
            )

            payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["matched_opening_count"], 0)
            self.assertEqual(payload["unmatched_opening_count"], 1)
            self.assertEqual(payload["projected_opening_count"], 1)
            self.assertEqual(result.projected_opening_count, 1)
            self.assertEqual(payload["projected_openings"][0]["id"], "window-0001")
            self.assertEqual(payload["opening_markers"], [])

            events = [
                json.loads(line)
                for line in wall_opening_events_path(workspace, data_root).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(events[-1]["event"], "generate_wall_model_opening_match")
            self.assertEqual(events[-1]["matched_opening_count"], 0)
            self.assertEqual(events[-1]["projected_opening_count"], 1)
            self.assertEqual(events[-1]["unmatched_opening_count"], 1)
            self.assertEqual(events[-1]["unmatched_openings"][0]["id"], "window-0001")

    def test_generate_wall_model_does_not_project_rejected_unmatched_opening(self):
        from core.wall_openings import WallOpening

        xs = np.linspace(0.0, 4.0, 45)
        ys = np.linspace(0.0, 3.0, 35)
        zs = np.linspace(0.0, 2.4, 12)
        points = []
        for z in zs:
            for x in xs:
                points.append([x, 0.0, z])
                points.append([x, 3.0, z])
            for y in ys:
                points.append([0.0, y, z])
                points.append([4.0, y, z])
        points = np.asarray(points, dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            opening = WallOpening(
                id="window-0001",
                label="window",
                source_image="608.jpg",
                seed_index=1,
                point_count=20,
                confidence="medium",
                reason="frustum_only_rejected_not_vertical_plane",
                center=(2.0, 1.5, 0.5),
                normal=(1.0, 0.0, 0.0),
                bbox_min=(1.8, 1.3, 0.2),
                bbox_max=(2.2, 1.7, 0.8),
                width_m=0.4,
                height_m=0.6,
                z_min=0.2,
                z_max=0.8,
                detection_index=-1,
                score=None,
            )

            result = generate_wall_model(
                workspace,
                data_root,
                points,
                openings=[opening],
                resolution_m=0.1,
            )

            payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["matched_opening_count"], 0)
            self.assertEqual(payload["unmatched_opening_count"], 1)
            self.assertEqual(payload["projected_opening_count"], 0)
            self.assertEqual(result.projected_opening_count, 0)
            self.assertEqual(payload["projected_openings"], [])


if __name__ == "__main__":
    unittest.main()
