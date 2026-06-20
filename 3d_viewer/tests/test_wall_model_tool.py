from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from core.wall_model import WallLineDraft, WallOpeningMarker, WallSegment, render_wall_line_draft_preview
from core.wall_openings import WallOpening, save_wall_openings
from wall_model_tool import (
    WallLineEditor,
    WallModelWorkbench,
    _ObjModelGLWindow,
    build_arg_parser,
    load_obj_triangles,
)


class WallModelToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_parser_accepts_workspace_sampling_and_auto_generate(self):
        parser = build_arg_parser()

        args = parser.parse_args([
            "--workspace",
            "/tmp/scan",
            "--max-points",
            "12345",
            "--auto-generate",
        ])

        self.assertEqual(args.workspace, Path("/tmp/scan"))
        self.assertEqual(args.max_points, 12345)
        self.assertTrue(args.auto_generate)

    def test_load_obj_triangles_triangulates_quad_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = Path(tmp) / "wall.obj"
            obj_path.write_text(
                "\n".join([
                    "v 0 0 0",
                    "v 1 0 0",
                    "v 1 1 0",
                    "v 0 1 0",
                    "f 1 2 3 4",
                ])
                + "\n",
                encoding="utf-8",
            )

            triangles = load_obj_triangles(obj_path)

        self.assertEqual(triangles.shape, (6, 3))
        self.assertEqual(triangles[0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(triangles[5].tolist(), [0.0, 1.0, 0.0])

    def test_obj_preview_shows_mesh_edges_by_default(self):
        view = _ObjModelGLWindow()
        self.addCleanup(view.close)

        self.assertTrue(view.show_mesh_edges)

    def test_workbench_splits_wall_line_generation_from_obj_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            workbench = WallModelWorkbench(Path(tmp), max_points=10_000)
        self.addCleanup(workbench.close)

        self.assertEqual(workbench.generate_lines_button.text(), "生成墙体线")
        self.assertEqual(workbench.export_obj_button.text(), "从墙体线生成 OBJ")
        self.assertFalse(workbench.export_obj_button.isEnabled())
        self.assertIsInstance(workbench.line_editor, WallLineEditor)
        self.assertEqual(workbench.zoom_in_button.text(), "放大")
        self.assertEqual(workbench.zoom_out_button.text(), "缩小")
        self.assertEqual(workbench.reset_zoom_button.text(), "重置视图")

    def test_wall_line_editor_deletes_and_adds_axis_aligned_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = WallLineDraft(
                wall_lines_path=Path(tmp) / "scan_wall_lines.json",
                topdown_preview_path=Path(tmp) / "scan_wall_lines_topdown.png",
                segments=[
                    WallSegment("horizontal", 0.0, 1.0, 2.0, 1.0, 0.0, 2.4, 2.0, 10, 2.4),
                ],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
            )
            editor = WallLineEditor()
            self.addCleanup(editor.close)

            editor.set_draft(draft)
            editor.select_segment(0)
            editor.delete_selected_segment()
            self.assertEqual(editor.wall_segments(), [])

            editor.add_axis_aligned_segment((0.2, 0.0), (0.4, 2.0), z_min=0.0, z_max=2.4)
            segments = editor.wall_segments()

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].orientation, "vertical")
        self.assertAlmostEqual(segments[0].x1, segments[0].x2)
        self.assertAlmostEqual(segments[0].length_m, 2.0)

    def test_wall_line_editor_emits_one_finished_event_per_discrete_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = WallLineDraft(
                wall_lines_path=Path(tmp) / "scan_wall_lines.json",
                topdown_preview_path=Path(tmp) / "scan_wall_lines_topdown.png",
                segments=[
                    WallSegment("horizontal", 0.0, 1.0, 2.0, 1.0, 0.0, 2.4, 2.0, 10, 2.4),
                ],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
            )
            editor = WallLineEditor()
            self.addCleanup(editor.close)
            events: list[str] = []
            editor.edit_finished.connect(events.append)

            editor.set_draft(draft)
            editor.select_segment(0)
            editor.delete_selected_segment()
            editor.add_axis_aligned_segment((0.2, 0.0), (0.4, 2.0), z_min=0.0, z_max=2.4)

        self.assertEqual(events, ["delete_segment", "add_segment"])

    def test_wall_line_editor_does_not_keep_deleted_line_from_preview_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            segment = WallSegment("horizontal", 0.0, 1.0, 2.0, 1.0, 0.0, 2.4, 2.0, 10, 2.4)
            grid = np.zeros((41, 41), dtype=np.uint32)
            wall_mask = np.zeros((41, 41), dtype=bool)
            draft = WallLineDraft(
                wall_lines_path=Path(tmp) / "scan_wall_lines.json",
                topdown_preview_path=Path(tmp) / "scan_wall_lines_topdown.png",
                segments=[segment],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
                grid=grid,
                wall_mask=wall_mask,
            )
            render_wall_line_draft_preview(draft).save(draft.topdown_preview_path)
            editor = WallLineEditor()
            self.addCleanup(editor.close)
            editor.resize(220, 220)

            editor.set_draft(draft)
            editor.select_segment(0)
            editor.delete_selected_segment()

            image = QImage(editor.size(), QImage.Format_ARGB32)
            image.fill(0)
            editor.render(image)

        red_pixels = 0
        for y in range(image.height()):
            for x in range(image.width()):
                color = image.pixelColor(x, y)
                if color.red() > 230 and 35 <= color.green() <= 90 and color.blue() < 80:
                    red_pixels += 1
        self.assertEqual(red_pixels, 0)

    def test_wall_line_editor_draws_opening_marker_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = WallLineDraft(
                wall_lines_path=Path(tmp) / "scan_wall_lines.json",
                topdown_preview_path=Path(tmp) / "scan_wall_lines_topdown.png",
                segments=[
                    WallSegment("vertical", 1.0, 0.0, 1.0, 3.0, 0.0, 2.4, 3.0, 10, 2.4),
                ],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
            )
            marker = WallOpeningMarker(
                opening_id="window-0001",
                label="window",
                segment_index=0,
                orientation="vertical",
                wall_coord=1.0,
                axis_min=1.0,
                axis_max=2.0,
                z_min=0.5,
                z_max=1.8,
                side=1.0,
            )
            editor = WallLineEditor()
            self.addCleanup(editor.close)
            editor.resize(220, 220)
            editor.set_draft(draft)

            editor.set_opening_overlays([marker], [])
            image = QImage(editor.size(), QImage.Format_ARGB32)
            image.fill(0)
            editor.render(image)

        green_pixels = 0
        for y in range(image.height()):
            for x in range(image.width()):
                color = image.pixelColor(x, y)
                if color.green() > 200 and color.red() < 80 and color.blue() < 140:
                    green_pixels += 1
        self.assertGreater(green_pixels, 0)
        self.assertEqual(editor.opening_overlay_counts(), (1, 0))

    def test_workbench_loads_recorded_openings_for_wall_line_editor(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan-a"
            data_root.mkdir()
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
            save_wall_openings(workspace, data_root, [opening])
            draft = WallLineDraft(
                wall_lines_path=workspace / "out" / "wall_models" / "scan-a_wall_lines.json",
                topdown_preview_path=workspace / "out" / "wall_models" / "scan-a_wall_lines_topdown.png",
                segments=[
                    WallSegment("vertical", 1.0, 0.0, 1.0, 3.0, 0.0, 2.4, 3.0, 10, 2.4),
                ],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
            )
            workbench = WallModelWorkbench(workspace, max_points=10_000)
            self.addCleanup(workbench.close)
            workbench.dataset = SimpleNamespace(data_root=data_root)

            workbench._show_wall_lines(draft)

        self.assertEqual(workbench.line_editor.opening_overlay_counts(), (1, 0))
        self.assertIn("门窗标记: 1 matched", workbench.info.text())

    def test_wall_line_editor_prefers_endpoint_hit_over_line_body_near_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = WallLineDraft(
                wall_lines_path=Path(tmp) / "scan_wall_lines.json",
                topdown_preview_path=Path(tmp) / "scan_wall_lines_topdown.png",
                segments=[
                    WallSegment("horizontal", 0.0, 1.0, 2.0, 1.0, 0.0, 2.4, 2.0, 10, 2.4),
                ],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
            )
            editor = WallLineEditor()
            self.addCleanup(editor.close)
            editor.resize(220, 220)
            editor.set_draft(draft)

            rect = editor._image_rect()
            start = editor._world_to_widget(0.0, 1.0, rect)
            pos = start + (editor._world_to_widget(0.2, 1.0, rect) - start) * 0.35
            index, part = editor._hit_test(pos)

        self.assertEqual(index, 0)
        self.assertEqual(part, "start")

    def test_wall_line_editor_dragging_horizontal_endpoint_makes_free_segment_and_keeps_other_endpoint_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = WallLineDraft(
                wall_lines_path=Path(tmp) / "scan_wall_lines.json",
                topdown_preview_path=Path(tmp) / "scan_wall_lines_topdown.png",
                segments=[
                    WallSegment("horizontal", 0.0, 1.0, 2.0, 1.0, 0.0, 2.4, 2.0, 10, 2.4),
                ],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
            )
            editor = WallLineEditor()
            self.addCleanup(editor.close)
            editor.set_draft(draft)
            editor.select_segment(0)
            editor._drag_part = "start"
            editor._drag_last_world = (0.0, 1.0)

            editor._move_selected((0.5, 1.8))
            segment = editor.wall_segments()[0]

        self.assertEqual(segment.orientation, "free")
        self.assertAlmostEqual(segment.x1, 0.5)
        self.assertAlmostEqual(segment.y1, 1.8)
        self.assertAlmostEqual(segment.x2, 2.0)
        self.assertAlmostEqual(segment.y2, 1.0)

    def test_wall_line_editor_dragging_vertical_endpoint_makes_free_segment_and_keeps_other_endpoint_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = WallLineDraft(
                wall_lines_path=Path(tmp) / "scan_wall_lines.json",
                topdown_preview_path=Path(tmp) / "scan_wall_lines_topdown.png",
                segments=[
                    WallSegment("vertical", 1.0, 0.0, 1.0, 2.0, 0.0, 2.4, 2.0, 10, 2.4),
                ],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
            )
            editor = WallLineEditor()
            self.addCleanup(editor.close)
            editor.set_draft(draft)
            editor.select_segment(0)
            editor._drag_part = "end"
            editor._drag_last_world = (1.0, 2.0)

            editor._move_selected((1.5, 2.2))
            segment = editor.wall_segments()[0]

        self.assertEqual(segment.orientation, "free")
        self.assertAlmostEqual(segment.x1, 1.0)
        self.assertAlmostEqual(segment.x2, 1.5)
        self.assertAlmostEqual(segment.y1, 0.0)
        self.assertAlmostEqual(segment.y2, 2.2)

    def test_wall_line_editor_snaps_dragged_endpoint_to_nearby_endpoint_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = WallLineDraft(
                wall_lines_path=Path(tmp) / "scan_wall_lines.json",
                topdown_preview_path=Path(tmp) / "scan_wall_lines_topdown.png",
                segments=[
                    WallSegment("horizontal", 0.0, 1.0, 2.0, 1.0, 0.0, 2.4, 2.0, 10, 2.4),
                    WallSegment("vertical", 3.0, 1.0, 3.0, 2.0, 0.0, 2.4, 1.0, 8, 2.4),
                ],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
            )
            editor = WallLineEditor()
            self.addCleanup(editor.close)
            editor.resize(220, 220)
            editor.set_draft(draft)
            editor.select_segment(0)
            editor._drag_part = "end"
            editor._drag_last_world = (2.0, 1.0)

            editor._move_selected((2.96, 1.04))
            segment = editor.wall_segments()[0]

        self.assertAlmostEqual(segment.x2, 3.0)
        self.assertAlmostEqual(segment.y2, 1.0)
        self.assertEqual(segment.orientation, "horizontal")

    def test_wall_line_editor_does_not_snap_endpoint_outside_tight_snap_radius(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = WallLineDraft(
                wall_lines_path=Path(tmp) / "scan_wall_lines.json",
                topdown_preview_path=Path(tmp) / "scan_wall_lines_topdown.png",
                segments=[
                    WallSegment("horizontal", 0.0, 1.0, 2.0, 1.0, 0.0, 2.4, 2.0, 10, 2.4),
                    WallSegment("vertical", 3.0, 1.0, 3.0, 2.0, 0.0, 2.4, 1.0, 8, 2.4),
                ],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
            )
            editor = WallLineEditor()
            self.addCleanup(editor.close)
            editor.resize(220, 220)
            editor.set_draft(draft)
            editor.select_segment(0)
            editor._drag_part = "end"
            editor._drag_last_world = (2.0, 1.0)

            editor._move_selected((2.90, 1.04))
            segment = editor.wall_segments()[0]

        self.assertAlmostEqual(segment.x2, 2.90)
        self.assertAlmostEqual(segment.y2, 1.04)
        self.assertAlmostEqual(segment.x1, 0.0)
        self.assertAlmostEqual(segment.y1, 1.0)
        self.assertEqual(segment.orientation, "free")

    def test_wall_line_editor_zoom_preserves_world_position_under_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = WallLineDraft(
                wall_lines_path=Path(tmp) / "scan_wall_lines.json",
                topdown_preview_path=Path(tmp) / "scan_wall_lines_topdown.png",
                segments=[
                    WallSegment("horizontal", 0.0, 1.0, 2.0, 1.0, 0.0, 2.4, 2.0, 10, 2.4),
                ],
                x_min=0.0,
                y_min=0.0,
                x_max=4.0,
                y_max=4.0,
                resolution_m=0.1,
                grid_shape=(41, 41),
            )
            editor = WallLineEditor()
            self.addCleanup(editor.close)
            editor.resize(220, 220)
            editor.set_draft(draft)
            anchor = QPointF(110.0, 110.0)
            world_before = editor._widget_to_world(anchor)
            rect_before = editor._image_rect()

            editor.zoom_by(2.0, anchor=anchor)
            world_after = editor._widget_to_world(anchor)
            rect_after = editor._image_rect()

        self.assertGreater(rect_after.width(), rect_before.width())
        self.assertAlmostEqual(editor.zoom_factor(), 2.0)
        self.assertAlmostEqual(world_after[0], world_before[0])
        self.assertAlmostEqual(world_after[1], world_before[1])


if __name__ == "__main__":
    unittest.main()
