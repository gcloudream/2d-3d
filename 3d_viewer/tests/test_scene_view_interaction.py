from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
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

    def test_pick_detection_hit_uses_yaw_offset_set_before_gl_initializes(self):
        window = _SceneGLWindow()
        self.addCleanup(window.close)
        window.resize(1000, 500)
        window.camera.yaw_deg = 90.0
        window.camera.pitch_deg = 0.0
        window.camera.fov_deg = 90.0
        window.set_pano_yaw_offset(-90.0)
        window._current_pose = CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        window._detections = [{"label": "window", "score": 0.9, "bbox": [240.0, 240.0, 260.0, 260.0]}]
        window._detection_image_size = (1000, 500)
        window._show_bboxes = True
        window._global_view_mode = False

        det_idx, click_u, click_v = window._pick_detection_hit(QPoint(500, 250))

        self.assertEqual(det_idx, 0)
        self.assertAlmostEqual(click_u, 250.0, delta=1.0)
        self.assertAlmostEqual(click_v, 250.0, delta=1.0)

    def test_point_interaction_mode_prefers_point_seed_inside_detection_box(self):
        window = _SceneGLWindow()
        self.addCleanup(window.close)
        emitted_points: list[int] = []
        emitted_detections: list[tuple[int, float, float]] = []
        window.point_clicked.connect(emitted_points.append)
        window.detection_clicked.connect(
            lambda idx, u, v: emitted_detections.append((idx, u, v))
        )
        window.set_interaction_mode("point")
        window._show_bboxes = True
        window._global_view_mode = False
        window._current_pose = CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        window._detections = [{"label": "window", "score": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]}]
        window._detection_image_size = (100, 50)
        window._pick_detection_hit = lambda _pos: (0, 15.0, 15.0)
        window._pick_point = lambda _pos, max_dist_px: 42

        class FakeMouseEvent:
            def button(self):
                return Qt.LeftButton

            def position(self):
                return QPointF(50, 50)

        window.mousePressEvent(FakeMouseEvent())

        self.assertEqual(emitted_points, [42])
        self.assertEqual(emitted_detections, [])

    def test_detection_interaction_mode_uses_detection_even_when_point_is_nearby(self):
        window = _SceneGLWindow()
        self.addCleanup(window.close)
        emitted_points: list[int] = []
        emitted_detections: list[tuple[int, float, float]] = []
        window.point_clicked.connect(emitted_points.append)
        window.detection_clicked.connect(
            lambda idx, u, v: emitted_detections.append((idx, u, v))
        )
        window.set_interaction_mode("detection")
        window._show_bboxes = True
        window._global_view_mode = False
        window._current_pose = CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        window._detections = [{"label": "window", "score": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]}]
        window._detection_image_size = (100, 50)
        window._pick_detection_hit = lambda _pos: (0, 15.0, 15.0)
        window._pick_point = lambda _pos, max_dist_px: 42

        class FakeMouseEvent:
            def button(self):
                return Qt.LeftButton

            def position(self):
                return QPointF(50, 50)

        window.mousePressEvent(FakeMouseEvent())

        self.assertEqual(emitted_points, [])
        self.assertEqual(emitted_detections, [(0, 15.0, 15.0)])

    def test_global_view_keyframe_can_skip_panorama_image_load(self):
        window = _SceneGLWindow()
        self.addCleanup(window.close)
        calls: list[str] = []
        window._ctx = object()
        window._pano = SimpleNamespace(
            load_image=lambda _path: calls.append("load_image"),
            set_pose=lambda *_args: calls.append("set_pose"),
        )
        window._bbox_overlay = SimpleNamespace(clear=lambda: calls.append("clear"))
        window.set_global_view_mode(True)
        pose = CameraPose("608.jpg", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        window.set_keyframe(pose, Path("missing.jpg"), load_pano=False)

        self.assertNotIn("load_image", calls)
        self.assertIn("set_pose", calls)
        self.assertIs(window._current_pose, pose)


if __name__ == "__main__":
    unittest.main()
