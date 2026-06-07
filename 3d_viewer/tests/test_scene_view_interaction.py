from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from core.dataset import CameraPose
from render.scene_view import _SceneGLWindow


class SceneViewInteractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_pick_detection_maps_mouse_position_to_panorama_bbox(self):
        window = _SceneGLWindow()
        self.addCleanup(window.close)
        window.resize(1000, 500)
        window.camera.yaw_deg = 90.0
        window.camera.pitch_deg = 0.0
        window.camera.fov_deg = 90.0
        window._current_pose = CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        window._detections = [{"label": "window", "score": 0.9, "bbox": [490.0, 240.0, 510.0, 260.0]}]
        window._detection_image_size = (1000, 500)
        window._show_bboxes = True
        window._global_view_mode = False

        self.assertEqual(window._pick_detection(QPoint(500, 250)), 0)
        self.assertEqual(window._pick_detection(QPoint(50, 50)), -1)

    def test_pick_detection_hit_returns_panorama_click_uv(self):
        window = _SceneGLWindow()
        self.addCleanup(window.close)
        window.resize(1000, 500)
        window.camera.yaw_deg = 90.0
        window.camera.pitch_deg = 0.0
        window.camera.fov_deg = 90.0
        window._current_pose = CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        window._detections = [{"label": "window", "score": 0.9, "bbox": [490.0, 240.0, 510.0, 260.0]}]
        window._detection_image_size = (1000, 500)
        window._show_bboxes = True
        window._global_view_mode = False

        det_idx, click_u, click_v = window._pick_detection_hit(QPoint(500, 250))

        self.assertEqual(det_idx, 0)
        self.assertAlmostEqual(click_u, 500.0, delta=1.0)
        self.assertAlmostEqual(click_v, 250.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
