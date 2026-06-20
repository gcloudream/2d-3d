from __future__ import annotations

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from core.dataset import CameraPose, Dataset
from core.app_logging import operation_events_path
from core.detection_cache import annotation_output_path
from core.wall_openings import (
    WallOpening,
    append_wall_opening_event,
    load_wall_openings,
    save_wall_openings,
    wall_opening_events_path,
    wall_openings_path,
)
from ui.main_window import MainWindow


class MainWindowWallOpeningsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_startup_schedules_dataset_directory_picker_without_default_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with patch("ui.main_window.find_default_dataset") as find_default, \
                patch("ui.main_window.QTimer.singleShot") as single_shot:
                win = MainWindow(workspace, prompt_for_dataset=True)
            self.addCleanup(win.close)

        find_default.assert_not_called()
        self.assertIsNone(win.dataset)
        single_shot.assert_called_once()
        self.assertEqual(single_shot.call_args.args[0], 0)

    def test_dataset_picker_loads_selected_directory_instead_of_default_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "chosen_scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "CAM" / "camera_pos.cam",
                image_dir=data_root / "CAM",
                pointcloud_file=data_root / "LAS_Rgb" / "chosen_scan_rgb_0.las",
                poses=[],
                points=np.asarray([[1.0, 0.0, 0.8]], dtype=np.float64),
                colors=np.zeros((1, 3), dtype=np.uint8),
                total_points=1,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=-90.0,
            )
            cfg = SimpleNamespace(data_root=data_root)

            with patch("ui.main_window.find_default_dataset") as find_default, \
                patch("ui.main_window.dataset_config_from_root", return_value=cfg) as selected_config, \
                patch("ui.main_window.load_dataset", return_value=dataset) as load, \
                patch("ui.main_window.QFileDialog.getExistingDirectory", return_value=str(data_root)):
                win = MainWindow(workspace, prompt_for_dataset=False)
                win._prompt_for_dataset_directory()
            self.addCleanup(win.close)

        find_default.assert_not_called()
        selected_config.assert_called_once_with(data_root)
        load.assert_called_once_with(cfg, max_points=300_000)
        self.assertEqual(win.dataset.data_root, data_root)

    def test_record_current_opening_saves_last_highlight(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[],
                points=np.asarray([
                    [1.0, 0.0, 0.8],
                    [1.0, 0.4, 1.0],
                    [1.0, 0.8, 1.6],
                ], dtype=np.float64),
                colors=np.zeros((3, 3), dtype=np.uint8),
                total_points=3,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=0.0,
            )
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win.current_idx = -1
            win._last_opening_candidate = {
                "mask": np.asarray([True, True, True]),
                "label": "window",
                "seed_index": 1,
                "confidence": "high",
                "reason": "accepted_vertical_planar_region",
                "plane_point": np.asarray([1.0, 0.4, 1.0]),
                "plane_normal": np.asarray([1.0, 0.0, 0.0]),
                "width_m": 0.8,
                "height_m": 0.8,
                "detection_index": 2,
                "score": 0.8,
            }

            win._record_current_opening()

            openings = load_wall_openings(workspace, data_root)
            self.assertEqual(len(openings), 1)
            self.assertEqual(openings[0].label, "window")
            self.assertEqual(openings[0].source_image, "")

            log_path = wall_opening_events_path(workspace, data_root)
            events = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event["event"], "record_opening_saved")
            self.assertEqual(event["opening_id"], "window-0001")
            self.assertEqual(event["seed_index"], 1)
            self.assertEqual(event["point_count"], 3)
            self.assertEqual(event["candidate_point_count"], 3)
            self.assertEqual(event["highlight_point_count"], 0)
            self.assertEqual(event["record_mask_source"], "latest_highlight")
            self.assertEqual(event["reason"], "accepted_vertical_planar_region")
            self.assertEqual(event["bbox_min"], [1.0, 0.0, 0.8])

    def test_record_current_opening_rejects_failed_geometry_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[],
                points=np.asarray([
                    [1.0, 0.0, 0.8],
                    [1.0, 0.4, 1.0],
                    [1.0, 0.8, 1.6],
                ], dtype=np.float64),
                colors=np.zeros((3, 3), dtype=np.uint8),
                total_points=3,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=0.0,
            )
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win.current_idx = -1
            win._last_opening_candidate = {
                "mask": np.asarray([True, True, True]),
                "label": "window",
                "seed_index": 1,
                "confidence": "medium",
                "reason": "frustum_only_rejected_not_vertical_plane",
                "plane_point": np.asarray([1.0, 0.4, 1.0]),
                "plane_normal": np.asarray([0.0, 0.0, 1.0]),
                "width_m": 0.8,
                "height_m": 0.8,
                "detection_index": 2,
                "score": 0.8,
            }

            win._record_current_opening()

            self.assertEqual(load_wall_openings(workspace, data_root), [])
            self.assertFalse(wall_opening_events_path(workspace, data_root).exists())

    def test_record_current_opening_rejects_incomplete_fused_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[],
                points=np.asarray([
                    [1.0, 0.0, 0.8],
                    [1.0, 0.4, 1.0],
                    [1.0, 0.8, 1.6],
                ], dtype=np.float64),
                colors=np.zeros((3, 3), dtype=np.uint8),
                total_points=3,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=0.0,
            )
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win.current_idx = -1
            win._last_opening_candidate = {
                "mask": np.asarray([True, True, True]),
                "label": "window",
                "seed_index": 1,
                "confidence": "medium",
                "reason": "frustum_only_too_few_region_points",
                "plane_point": np.asarray([1.0, 0.4, 1.0]),
                "plane_normal": np.asarray([1.0, 0.0, 0.0]),
                "width_m": None,
                "height_m": None,
                "detection_index": 2,
                "score": 0.8,
            }

            win._record_current_opening()

            self.assertEqual(load_wall_openings(workspace, data_root), [])
            self.assertFalse(wall_opening_events_path(workspace, data_root).exists())

    def test_selected_dataset_load_resets_wall_opening_records_for_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[],
                points=np.asarray([[1.0, 0.0, 0.8]], dtype=np.float64),
                colors=np.zeros((1, 3), dtype=np.uint8),
                total_points=1,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=0.0,
            )
            opening = WallOpening(
                id="window-0001",
                label="window",
                source_image="608.jpg",
                seed_index=1,
                point_count=3,
                confidence="high",
                reason="accepted_vertical_planar_region",
                center=(1.0, 0.4, 1.0),
                normal=(1.0, 0.0, 0.0),
                bbox_min=(1.0, 0.0, 0.8),
                bbox_max=(1.0, 0.8, 1.6),
                width_m=0.8,
                height_m=0.8,
                z_min=0.8,
                z_max=1.6,
                detection_index=2,
                score=0.8,
            )
            save_wall_openings(workspace, data_root, [opening])
            append_wall_opening_event(workspace, data_root, {"event": "record_opening_saved"})
            other_root = workspace / "other_scan"
            other_root.mkdir()
            save_wall_openings(workspace, other_root, [opening])
            append_wall_opening_event(workspace, other_root, {"event": "record_opening_saved"})

            cfg = SimpleNamespace(data_root=data_root)
            with patch("ui.main_window.dataset_config_from_root", return_value=cfg), \
                patch("ui.main_window.load_dataset", return_value=dataset):
                win = MainWindow(workspace, prompt_for_dataset=False)
                win._load_selected_dataset(data_root)
            self.addCleanup(win.close)

            self.assertEqual(load_wall_openings(workspace, data_root), [])
            self.assertFalse(wall_opening_events_path(workspace, data_root).exists())
            self.assertTrue(wall_openings_path(workspace, other_root).exists())
            self.assertTrue(wall_opening_events_path(workspace, other_root).exists())

    def test_load_defaults_panorama_yaw_offset_to_minus_90(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[],
                points=np.asarray([[1.0, 0.0, 0.8]], dtype=np.float64),
                colors=np.zeros((1, 3), dtype=np.uint8),
                total_points=1,
                sample_step=1,
                pano_calibration=SimpleNamespace(default_yaw_offset_deg=-92.819),
                pano_yaw_offset_deg=-92.819,
            )

            with patch("ui.main_window.find_default_dataset", return_value=SimpleNamespace(data_root=data_root)), \
                patch("ui.main_window.load_dataset", return_value=dataset):
                win = MainWindow(workspace, prompt_for_dataset=False)
                win._load()
            self.addCleanup(win.close)

            self.assertAlmostEqual(float(win.yaw_offset.currentData()), -90.0)

    def test_load_enables_depth_test_for_primary_highlight(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[],
                points=np.asarray([[1.0, 0.0, 0.8]], dtype=np.float64),
                colors=np.zeros((1, 3), dtype=np.uint8),
                total_points=1,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=-90.0,
            )

            with patch("ui.main_window.find_default_dataset", return_value=SimpleNamespace(data_root=data_root)), \
                patch("ui.main_window.load_dataset", return_value=dataset):
                win = MainWindow(workspace, prompt_for_dataset=False)
                win._load()
            self.addCleanup(win.close)

            self.assertTrue(win.scene.gl_window._selected_depth_test)

    def test_interaction_mode_configures_primary_and_global_views(self):
        with patch.object(MainWindow, "_load", lambda self: None):
            win = MainWindow(ROOT.parent)
        self.addCleanup(win.close)

        def choose(mode: str):
            for i in range(win.interaction_mode.count()):
                if win.interaction_mode.itemData(i) == mode:
                    win.interaction_mode.setCurrentIndex(i)
                    return
            raise AssertionError(f"missing interaction mode {mode}")

        choose("point")
        self.assertEqual(win.scene.gl_window._interaction_mode, "point")
        self.assertEqual(win.cloud_scene.gl_window._interaction_mode, "point")

        choose("detection")
        self.assertEqual(win.scene.gl_window._interaction_mode, "detection")
        self.assertEqual(win.cloud_scene.gl_window._interaction_mode, "navigate")

        choose("navigate")
        self.assertEqual(win.scene.gl_window._interaction_mode, "navigate")
        self.assertEqual(win.cloud_scene.gl_window._interaction_mode, "navigate")

    def test_main_window_uses_manual_annotation_workflow_without_auto_detection_controls(self):
        with patch.object(MainWindow, "_load", lambda self: None):
            win = MainWindow(ROOT.parent)
        self.addCleanup(win.close)

        self.assertFalse(hasattr(win, "btn_detect_current"))
        self.assertFalse(hasattr(win, "det_mode"))
        self.assertTrue(hasattr(win, "btn_edit_current"))

    def test_main_window_uses_single_panorama_visibility_checkbox(self):
        with patch.object(MainWindow, "_load", lambda self: None):
            win = MainWindow(ROOT.parent)
        self.addCleanup(win.close)

        calls: list[bool] = []
        win.scene.set_show_pano = lambda on: calls.append(bool(on))

        self.assertFalse(hasattr(win, "btn_toggle_pano"))
        win.cb_pano.setChecked(False)
        win.cb_pano.setChecked(True)

        self.assertEqual(calls, [False, True])

    def test_xray_highlight_toggle_updates_both_views_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.current_idx = 3

            win._set_highlight_xray(False)
            self.assertTrue(win.scene.gl_window._selected_depth_test)
            self.assertTrue(win.cloud_scene.gl_window._selected_depth_test)

            win.cb_xray_highlight.setChecked(True)

            self.assertFalse(win.scene.gl_window._selected_depth_test)
            self.assertFalse(win.cloud_scene.gl_window._selected_depth_test)
            events = [
                json.loads(line)
                for line in operation_events_path(workspace).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(events[-1]["event"], "highlight_xray_changed")
            self.assertTrue(events[-1]["xray_enabled"])

    def test_clear_highlight_clears_opening_candidate(self):
        with patch.object(MainWindow, "_load", lambda self: None):
            win = MainWindow(ROOT.parent)
        self.addCleanup(win.close)
        win._last_opening_candidate = {"label": "window"}

        win._clear_highlight()

        self.assertIsNone(win._last_opening_candidate)

    def test_detection_click_runs_bbox_extraction_and_highlights_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                points=np.asarray([
                    [1.0, 0.0, 0.8],
                    [1.0, 0.4, 1.0],
                    [1.0, 0.8, 1.6],
                ], dtype=np.float64),
                colors=np.zeros((3, 3), dtype=np.uint8),
                total_points=3,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=-90.0,
            )
            selection = SimpleNamespace(
                mask=np.asarray([True, True, True]),
                point_count=3,
                confidence="high",
                reason="fused_frustum_and_planar_geometry",
                label="window",
                source="fused",
                detection_index=0,
                score=0.9,
                plane_point=np.asarray([1.0, 0.4, 1.0]),
                plane_normal=np.asarray([1.0, 0.0, 0.0]),
                width_m=0.8,
                height_m=0.8,
                diagnostics={
                    "bbox_candidate_count": 120,
                    "selected_depth_candidate_count": 64,
                    "seed_attempt_count": 4,
                    "total_ms": 12.345,
                },
            )

            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win.current_idx = 0
            win.current_detections = [{"label": "window", "score": 0.9, "bbox": [10, 10, 20, 20]}]
            win.current_image_size = (100, 50)

            with patch("ui.main_window.extract_detection_region_from_bbox", return_value=selection) as extract:
                win._on_detection_clicked(0, 12.5, 18.5)

            extract.assert_called_once()
            self.assertEqual(extract.call_args.args[1], 0)
            self.assertEqual(extract.call_args.kwargs["click_uv"], (12.5, 18.5))
            self.assertIs(extract.call_args.kwargs["cache"], win._fusion_cache)
            self.assertEqual(int(win._highlight_mask.sum()), 3)
            self.assertEqual(win._last_opening_candidate["label"], "window")
            self.assertEqual(win._last_opening_candidate["seed_index"], -1)
            self.assertIn("diag: bbox=120", win.lbl_detection.text())
            events = [
                json.loads(line)
                for line in operation_events_path(workspace).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(events[-1]["event"], "detection_bbox_extract")
            self.assertEqual(events[-1]["detection_index"], 0)
            self.assertEqual(events[-1]["diagnostics"]["bbox_candidate_count"], 120)

    def test_debug_layer_switch_displays_intermediate_mask_without_replacing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            final_mask = np.asarray([True, False, False, True])
            bbox_mask = np.asarray([True, True, True, True])
            depth_mask = np.asarray([True, True, False, True])
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                points=np.asarray([
                    [1.0, 0.0, 0.8],
                    [1.0, 0.4, 1.0],
                    [1.0, 0.8, 1.6],
                    [1.0, 1.2, 1.8],
                ], dtype=np.float64),
                colors=np.zeros((4, 3), dtype=np.uint8),
                total_points=4,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=-90.0,
            )
            selection = SimpleNamespace(
                mask=final_mask,
                point_count=2,
                confidence="high",
                reason="fused_frustum_and_planar_geometry",
                label="window",
                source="fused",
                detection_index=0,
                score=0.9,
                plane_point=np.asarray([1.0, 0.4, 1.0]),
                plane_normal=np.asarray([1.0, 0.0, 0.0]),
                width_m=0.8,
                height_m=0.8,
                diagnostics={
                    "bbox_candidate_count": 4,
                    "selected_depth_candidate_count": 3,
                    "seed_attempt_count": 2,
                    "total_ms": 5.0,
                },
                debug_masks={"bbox": bbox_mask, "depth": depth_mask, "final": final_mask},
            )
            displayed = []
            styles = []

            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win.current_idx = 0
            win.current_detections = [{"label": "window", "score": 0.9, "bbox": [10, 10, 20, 20]}]
            win.current_image_size = (100, 50)

            with patch("ui.main_window.extract_detection_region_from_bbox", return_value=selection), \
                patch("ui.main_window.set_scene_pair_highlight_mask", side_effect=lambda _a, _b, mask: displayed.append(None if mask is None else np.asarray(mask, dtype=bool).copy())), \
                patch("ui.main_window.set_scene_pair_highlight_style", side_effect=lambda _a, _b, ring, fill: styles.append((tuple(ring), tuple(fill)))):
                win._on_detection_clicked(0, 12.5, 18.5)
                self.assertTrue(np.array_equal(win._highlight_mask, final_mask))
                self.assertTrue(np.array_equal(win._last_opening_candidate["mask"], final_mask))

                win._set_debug_layer("bbox")
                self.assertTrue(np.array_equal(displayed[-1], bbox_mask))
                self.assertEqual(styles[-1], ((0.0, 0.85, 1.0), (0.0, 0.2, 1.0)))
                self.assertTrue(np.array_equal(win._highlight_mask, final_mask))
                self.assertTrue(np.array_equal(win._last_opening_candidate["mask"], final_mask))
                self.assertIn("调试层: bbox", win.lbl_detection.text())

                win._set_debug_layer("off")
                self.assertTrue(np.array_equal(displayed[-1], final_mask))
                self.assertEqual(styles[-1], ((1.0, 1.0, 0.0), (1.0, 0.0, 0.0)))
            events = [
                json.loads(line)
                for line in operation_events_path(workspace).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(events[-1]["event"], "debug_layer_changed")
            self.assertEqual(events[-1]["debug_layer"], "off")

    def test_top_point_click_uses_detection_fallback_when_seed_misses_frustum(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                points=np.asarray([
                    [1.0, 0.0, 0.8],
                    [1.0, 0.4, 1.0],
                    [1.0, 0.8, 1.6],
                ], dtype=np.float64),
                colors=np.zeros((3, 3), dtype=np.uint8),
                total_points=3,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=-90.0,
            )
            bbox_selection = SimpleNamespace(
                mask=np.asarray([True, True, False]),
                point_count=2,
                confidence="high",
                reason="fused_frustum_and_planar_geometry",
                label="window",
                source="fused",
                detection_index=0,
                score=0.9,
                plane_point=np.asarray([1.0, 0.2, 0.9]),
                plane_normal=np.asarray([1.0, 0.0, 0.0]),
                width_m=0.4,
                height_m=0.8,
                diagnostics={"bbox_candidate_count": 2, "seed_attempt_count": 2},
            )
            pure_selection = SimpleNamespace(
                mask=np.asarray([True, True, True]),
                point_count=3,
                confidence="high",
                reason="accepted_vertical_planar_region",
                label="object",
                source="",
                detection_index=-1,
                score=None,
                plane_point=np.asarray([1.0, 0.4, 1.0]),
                plane_normal=np.asarray([1.0, 0.0, 0.0]),
                width_m=3.8,
                height_m=3.0,
                diagnostics={},
            )
            missed_fusion = SimpleNamespace(source="none", reason="seed_not_in_any_frustum")

            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win.current_idx = 0
            win.current_detections = [{"label": "window", "score": 0.9, "bbox": [10, 10, 20, 20]}]
            win.current_image_size = (100, 50)
            win.scene.consume_point_detection_hit = lambda: (0, 12.5, 18.5)

            with patch.object(win, "_try_fused_extraction", return_value=missed_fusion), \
                patch("ui.main_window.extract_detection_region_from_bbox", return_value=bbox_selection) as extract, \
                patch("ui.main_window.extract_planar_region_from_seed", return_value=pure_selection) as pure_extract:
                win._on_point_clicked(43949)

            pure_extract.assert_not_called()
            extract.assert_called_once()
            self.assertEqual(extract.call_args.args[1], 0)
            self.assertEqual(extract.call_args.kwargs["click_uv"], (12.5, 18.5))
            self.assertEqual(int(win._highlight_mask.sum()), 2)
            self.assertEqual(win._last_opening_candidate["detection_index"], 0)
            events = [
                json.loads(line)
                for line in operation_events_path(workspace).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(events[-1]["event"], "pointcloud_detection_fallback_extract")
            self.assertEqual(events[-1]["seed_index"], 43949)
            self.assertEqual(events[-1]["fallback_detection_index"], 0)

    def test_top_point_click_uses_detection_fallback_when_seed_fusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                points=np.asarray([
                    [1.0, 0.0, 0.8],
                    [1.0, 0.4, 1.0],
                    [1.0, 0.8, 1.6],
                ], dtype=np.float64),
                colors=np.zeros((3, 3), dtype=np.uint8),
                total_points=3,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=-90.0,
            )
            bbox_selection = SimpleNamespace(
                mask=np.asarray([False, True, True]),
                point_count=2,
                confidence="high",
                reason="fused_frustum_and_planar_geometry",
                label="window",
                source="fused",
                detection_index=0,
                score=0.9,
                plane_point=np.asarray([1.0, 0.6, 1.3]),
                plane_normal=np.asarray([1.0, 0.0, 0.0]),
                width_m=0.4,
                height_m=0.8,
                diagnostics={"bbox_candidate_count": 2, "seed_attempt_count": 2},
            )
            rejected_fusion = SimpleNamespace(
                mask=np.asarray([True, False, False]),
                point_count=1,
                confidence="medium",
                reason="frustum_only_rejected_not_vertical_plane",
                label="window",
                source="fused",
                detection_index=0,
                score=0.9,
                plane_point=np.asarray([1.0, 0.0, 0.8]),
                plane_normal=np.asarray([0.0, 0.0, 1.0]),
                width_m=0.2,
                height_m=0.2,
                diagnostics={},
            )

            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win.current_idx = 0
            win.current_detections = [{"label": "window", "score": 0.9, "bbox": [10, 10, 20, 20]}]
            win.current_image_size = (100, 50)
            win.scene.consume_point_detection_hit = lambda: (0, 12.5, 18.5)

            with patch.object(win, "_try_fused_extraction", return_value=rejected_fusion), \
                patch("ui.main_window.extract_detection_region_from_bbox", return_value=bbox_selection) as extract, \
                patch("ui.main_window.extract_planar_region_from_seed") as pure_extract:
                win._on_point_clicked(291867)

            pure_extract.assert_not_called()
            extract.assert_called_once()
            self.assertEqual(int(win._highlight_mask.sum()), 2)
            events = [
                json.loads(line)
                for line in operation_events_path(workspace).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(events[-1]["event"], "pointcloud_detection_fallback_extract")
            self.assertEqual(events[-1]["fallback_reason"], "frustum_only_rejected_not_vertical_plane")

    def test_record_current_opening_uses_accumulated_highlight_in_supplement_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[],
                points=np.asarray([
                    [1.0, 0.0, 0.8],
                    [1.0, 0.4, 1.0],
                    [1.0, 0.8, 1.6],
                    [1.0, 1.2, 1.8],
                ], dtype=np.float64),
                colors=np.zeros((4, 3), dtype=np.uint8),
                total_points=4,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=0.0,
            )
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win.current_idx = -1
            win.cb_supplement.setChecked(True)
            win._set_highlight_mask(np.asarray([True, True, False, False]))
            first_selection = SimpleNamespace(
                label="window",
                confidence="medium",
                reason="seed_component_depth_filtered",
                point_count=2,
                detection_index=4,
                score=1.0,
                width_m=0.4,
                height_m=0.7,
                source="fused",
                plane_point=np.asarray([1.0, 0.0, 1.0]),
                plane_normal=np.asarray([1.0, 0.0, 0.0]),
            )
            win._last_opening_candidate = win._opening_candidate_from_selection(
                10,
                first_selection,
                np.asarray([True, True, False, False]),
            )
            second_selection = SimpleNamespace(
                label="window",
                confidence="medium",
                reason="seed_component_depth_filtered",
                point_count=2,
                detection_index=4,
                score=1.0,
                width_m=0.4,
                height_m=0.7,
                source="fused",
                plane_point=np.asarray([1.0, 0.8, 1.6]),
                plane_normal=np.asarray([1.0, 0.0, 0.0]),
            )

            effective = win._apply_extraction_highlight(np.asarray([False, False, True, True]))
            win._last_opening_candidate = win._opening_candidate_from_selection(11, second_selection, effective)
            win._record_current_opening()

            openings = load_wall_openings(workspace, data_root)
            self.assertEqual(len(openings), 1)
            self.assertEqual(openings[0].point_count, 4)
            self.assertEqual(openings[0].bbox_min, (1.0, 0.0, 0.8))
            self.assertEqual(openings[0].bbox_max, (1.0, 1.2, 1.8))
            events = [
                json.loads(line)
                for line in wall_opening_events_path(workspace, data_root).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(events[-1]["record_mask_source"], "accumulated_highlight")

    def test_logs_extraction_candidate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[],
                points=np.asarray([
                    [1.0, 0.0, 0.8],
                    [1.0, 0.4, 1.0],
                ], dtype=np.float64),
                colors=np.zeros((2, 3), dtype=np.uint8),
                total_points=2,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=0.0,
            )
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win._highlight_mask = np.asarray([True, True])
            selection = SimpleNamespace(
                label="window",
                confidence="medium",
                reason="frustum_only_rejected_not_vertical_plane",
                point_count=1,
                detection_index=4,
                score=1.0,
                width_m=0.4,
                height_m=0.7,
                source="fused",
                diagnostics={
                    "bbox_candidate_count": 10,
                    "selected_depth_candidate_count": 6,
                    "seed_attempt_count": 2,
                    "total_ms": 3.21,
                },
            )

            win._log_opening_selection_event(
                "extract_opening_candidate",
                0,
                selection,
                np.asarray([True, False]),
            )

            events = [
                json.loads(line)
                for line in wall_opening_events_path(workspace, data_root).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event["event"], "extract_opening_candidate")
            self.assertEqual(event["seed_index"], 0)
            self.assertEqual(event["candidate_point_count"], 1)
            self.assertEqual(event["highlight_point_count"], 2)
            self.assertEqual(event["event_mask_source"], "latest_highlight")
            self.assertEqual(event["reason"], "frustum_only_rejected_not_vertical_plane")
            self.assertEqual(event["candidate_bbox_min"], [1.0, 0.0, 0.8])
            self.assertEqual(event["extraction_diagnostics"]["bbox_candidate_count"], 10)
            self.assertEqual(event["extraction_diagnostics"]["selected_depth_candidate_count"], 6)

    def test_failed_supplement_click_preserves_highlight_but_clears_recordable_candidate(self):
        with patch.object(MainWindow, "_load", lambda self: None):
            win = MainWindow(ROOT.parent)
        self.addCleanup(win.close)
        win.cb_supplement.setChecked(True)
        old_mask = np.asarray([True, False, False])
        win._set_highlight_mask(old_mask)
        win._last_opening_candidate = {"mask": old_mask, "label": "window"}
        selection = SimpleNamespace(
            label="window",
            confidence="low",
            reason="rejected_not_vertical_plane",
            point_count=0,
            detection_index=4,
            score=1.0,
            width_m=None,
            height_m=None,
            source="fused",
        )

        effective = win._set_opening_candidate_from_extraction(7, selection, None)

        self.assertTrue(np.array_equal(effective, old_mask))
        self.assertIsNone(win._last_opening_candidate)

    def test_load_detections_filters_invalid_detection_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            pose = CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            out_path = annotation_output_path(workspace, pose.image_name)
            out_path.parent.mkdir(parents=True)
            out_path.write_text(
                json.dumps(
                    {
                        "width": 100,
                        "height": 50,
                        "detections": [
                            {"label": "window", "score": 0.9, "bbox": [10, 10, 20, 20]},
                            {"label": "door"},
                            {"label": "window", "bbox": [1, 2, 3]},
                            {"label": "window", "score": "bad", "bbox": [30, 10, 40, 20]},
                            {"label": "window", "bbox": [5, 25, 15, 20]},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)

            win._load_detections(pose)

            self.assertEqual(len(win.current_detections), 2)
            self.assertEqual(win.current_detections[0]["bbox"], [10.0, 10.0, 20.0, 20.0])
            self.assertEqual(win.current_detections[0]["score"], 0.9)
            self.assertNotIn("score", win.current_detections[1])

    def test_load_detections_handles_invalid_image_size_and_logs_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            pose = CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            out_path = annotation_output_path(workspace, pose.image_name)
            out_path.parent.mkdir(parents=True)
            out_path.write_text(
                json.dumps({"width": "bad", "height": 50, "detections": []}),
                encoding="utf-8",
            )
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)

            win._load_detections(pose)

            self.assertEqual(win.current_detections, [])
            self.assertIsNone(win.current_image_size)
            self.assertIn("尺寸无效", win.lbl_detection.text())
            events = [
                json.loads(line)
                for line in operation_events_path(workspace).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(events[-1]["event"], "detection_load_failed")
            self.assertEqual(events[-1]["reason"], "invalid_image_size")

    def test_missing_detection_json_prompts_manual_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            pose = CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)

            win._load_detections(pose)

            self.assertIn("编辑当前全景框", win.lbl_detection.text())


if __name__ == "__main__":
    unittest.main()
