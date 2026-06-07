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

from core.dataset import Dataset
from core.wall_openings import load_wall_openings, wall_opening_events_path
from ui.main_window import MainWindow


class MainWindowWallOpeningsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

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
            self.assertEqual(event["reason"], "accepted_vertical_planar_region")
            self.assertEqual(event["bbox_min"], [1.0, 0.0, 0.8])

    def test_clear_highlight_clears_opening_candidate(self):
        with patch.object(MainWindow, "_load", lambda self: None):
            win = MainWindow(ROOT.parent)
        self.addCleanup(win.close)
        win._last_opening_candidate = {"label": "window"}

        win._clear_highlight()

        self.assertIsNone(win._last_opening_candidate)

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
            self.assertEqual(event["reason"], "frustum_only_rejected_not_vertical_plane")
            self.assertEqual(event["candidate_bbox_min"], [1.0, 0.0, 0.8])


if __name__ == "__main__":
    unittest.main()
