from __future__ import annotations

import os
import sys
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
from ui.main_window import MainWindow


class MainWindowWallModelEntryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_wall_model_button_switches_to_embedded_workbench(self):
        with patch.object(MainWindow, "_load", lambda self: None):
            win = MainWindow(ROOT.parent)
        self.addCleanup(win.close)

        self.assertIs(win.view_stack.currentWidget(), win.normal_view)
        self.assertTrue(hasattr(win, "btn_wall_model"))
        self.assertTrue(hasattr(win, "btn_main_view"))
        self.assertFalse(win.side_panel.isHidden())
        self.assertEqual(win.wall_model_view.__class__.__name__, "WallModelWorkbench")

        win.btn_wall_model.click()

        self.assertIs(win.view_stack.currentWidget(), win.wall_model_view)
        self.assertTrue(win.side_panel.isHidden())

        win.wall_model_view.back_button.click()

        self.assertIs(win.view_stack.currentWidget(), win.normal_view)
        self.assertFalse(win.side_panel.isHidden())

    def test_wall_model_entry_reuses_current_main_dataset(self):
        with patch.object(MainWindow, "_load", lambda self: None):
            win = MainWindow(ROOT.parent)
        self.addCleanup(win.close)
        data_root = ROOT.parent / "scan"
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
            pano_calibration=SimpleNamespace(default_yaw_offset_deg=-90.0),
            pano_yaw_offset_deg=-90.0,
        )
        win.dataset = dataset

        win._show_wall_model()

        self.assertIs(win.wall_model_view.dataset, dataset)


if __name__ == "__main__":
    unittest.main()
