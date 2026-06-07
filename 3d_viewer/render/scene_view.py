"""主 OpenGL 视图：Camera + PanoSphere + PointCloud + 鼠标交互。

这里不用 QOpenGLWidget，而是用 QOpenGLWindow 包一层 QWidget 容器。
原因：QOpenGLWidget 在 macOS/Qt 下通过隐藏 FBO 合成，ModernGL 容易画到错误
framebuffer，表现为全景逃出左侧视图区域。QOpenGLWindow 是 native surface，
更适合和 ModernGL 直接配合。
"""
from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QTimer, Qt, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QVBoxLayout, QWidget

from core.dataset import CameraPose
from core.door_window import match_points_to_detections
from core.projection import project_points_to_panorama, rotation_from_angle
from render.bbox_overlay import BboxOverlay
from render.camera import Camera
from render.orbit_camera import OrbitCamera
from render.pano_sphere import PanoSphere
from render.point_cloud import PointCloud
from render.picking import find_nearest_to_mouse


class SceneView(QWidget):
    """QWidget facade used by MainWindow.

    Public methods mirror the old QOpenGLWidget-based SceneView, while actual
    rendering happens in the embedded QOpenGLWindow.
    """

    hover_changed = Signal(object)  # 发出 dict 或 None
    point_clicked = Signal(int)
    detection_clicked = Signal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gl_window = _SceneGLWindow()
        self.gl_window.hover_changed.connect(self.hover_changed)
        self.gl_window.point_clicked.connect(self.point_clicked)
        self.gl_window.detection_clicked.connect(self.detection_clicked)

        self.container = QWidget.createWindowContainer(self.gl_window, self)
        self.container.setFocusPolicy(Qt.StrongFocus)
        self.container.setMouseTracking(True)
        self.container.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.container)

    def set_world_points(self, points: np.ndarray, colors: np.ndarray):
        self.gl_window.set_world_points(points, colors)

    def set_keyframe(self, pose: CameraPose, image_path: Path):
        self.gl_window.set_keyframe(pose, image_path)

    def set_show_pano(self, on: bool):
        self.gl_window.set_show_pano(on)

    def set_show_pc(self, on: bool):
        self.gl_window.set_show_pc(on)

    def set_point_size(self, sz: float):
        self.gl_window.set_point_size(sz)

    def set_pano_yaw_offset(self, degrees: float):
        self.gl_window.set_pano_yaw_offset(degrees)

    def reset_view(self):
        self.gl_window.reset_view()

    def set_highlight(self, idx: int):
        self.gl_window.set_highlight(idx)

    def set_highlight_mask(self, mask: np.ndarray | None):
        self.gl_window.set_highlight_mask(mask)

    def set_selected_depth_test(self, on: bool):
        self.gl_window.set_selected_depth_test(on)

    def set_pick_mode(self, on: bool):
        self.gl_window.set_pick_mode(on)

    def set_show_bboxes(self, on: bool):
        self.gl_window.set_show_bboxes(on)

    def set_detections(self, detections: list[dict], img_w: int, img_h: int):
        self.gl_window.set_detections(detections, img_w, img_h)

    def set_selected_detection(self, idx: int):
        self.gl_window.set_selected_detection(idx)

    def set_global_view_mode(self, on: bool):
        self.gl_window.set_global_view_mode(on)

    def eventFilter(self, obj, event):
        if obj is self.container and event.type() == QEvent.KeyPress:
            if self.gl_window.handle_key(event.key()):
                return True
        return super().eventFilter(obj, event)


class _SceneGLWindow(QOpenGLWindow):
    hover_changed = Signal(object)
    point_clicked = Signal(int)
    detection_clicked = Signal(int, float, float)

    def __init__(self):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        super().__init__()
        self.setFormat(fmt)

        self.camera = Camera()
        self.orbit_camera = OrbitCamera()
        self._global_view_mode = False
        self._ctx: moderngl.Context | None = None
        self._pano: PanoSphere | None = None
        self._pc: PointCloud | None = None
        self._bbox_overlay: BboxOverlay | None = None
        self._show_pano = True
        self._show_pc = True
        self._show_bboxes = True
        self._pick_mode = False
        self._selected_depth_test = False
        self._current_pose: CameraPose | None = None
        self._detections: list[dict] = []
        self._detection_image_size = (0, 0)
        self._selected_detection = -1

        self._dragging = False
        self._last_pos = QPoint()
        self._cursor_pos = QPoint(-1, -1)

        self._pending_points: tuple[np.ndarray, np.ndarray] | None = None
        self._pending_pose: tuple[CameraPose, Path] | None = None

        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(30)
        self._hover_timer.timeout.connect(self._do_hover)

    # ------------- public API -------------

    def set_world_points(self, points: np.ndarray, colors: np.ndarray):
        if self._ctx is None:
            self._pending_points = (points, colors)
        else:
            self._pc.upload(points, colors)
            if self._global_view_mode:
                self.orbit_camera.fit_to_points(points)
            self.update()

    def set_keyframe(self, pose: CameraPose, image_path: Path):
        if self._ctx is None:
            self._pending_pose = (pose, image_path)
            return
        self._clear_hover()
        self._pano.load_image(image_path)
        self._pano.set_pose(pose.roll, pose.pitch, pose.yaw)
        if not self._global_view_mode:
            self.camera.set_keyframe(
                pose.position,
                pose.roll,
                pose.pitch,
                pose.yaw,
                self._pano.yaw_offset_deg,
            )
        self._current_pose = pose
        self._selected_detection = -1
        self._refresh_bbox_overlay()
        self.update()

    def set_show_pano(self, on: bool):
        self._show_pano = on
        self.update()

    def set_show_pc(self, on: bool):
        self._show_pc = on
        self.update()

    def set_point_size(self, sz: float):
        if self._pc is not None:
            self._pc.point_size = float(sz)
            self.update()

    def set_pano_yaw_offset(self, degrees: float):
        if self._pano is not None:
            self._pano.set_yaw_offset(degrees)
            if self._current_pose is not None and not self._global_view_mode:
                self.camera.update_keyframe_yaw_reference(
                    self._current_pose.roll,
                    self._current_pose.pitch,
                    self._current_pose.yaw,
                    degrees,
                )
            self._refresh_bbox_overlay()
            self.update()

    def reset_view(self):
        if self._global_view_mode:
            self.orbit_camera.reset_view()
        else:
            self.camera.yaw_deg = self.camera._keyframe_yaw_deg
            self.camera.pitch_deg = 0.0
            self.camera.fov_deg = 75.0
        self._clear_hover()
        self.update()

    def set_global_view_mode(self, on: bool):
        self._global_view_mode = bool(on)
        if self._global_view_mode and self._pc is not None and self._pc.points is not None:
            self.orbit_camera.fit_to_points(self._pc.points)
        self.update()

    def set_highlight(self, idx: int):
        if self._pc is not None:
            self._pc.highlight = int(idx)
            self.update()

    def set_highlight_mask(self, mask: np.ndarray | None):
        if self._pc is not None:
            self._pc.set_highlight_mask(mask)
            self.update()

    def set_selected_depth_test(self, on: bool):
        self._selected_depth_test = bool(on)
        if self._pc is not None:
            self._pc.set_selected_depth_test(self._selected_depth_test)
            self.update()

    def set_pick_mode(self, on: bool):
        self._pick_mode = bool(on)
        self._clear_hover()

    def set_show_bboxes(self, on: bool):
        self._show_bboxes = bool(on)
        self.update()

    def set_detections(self, detections: list[dict], img_w: int, img_h: int):
        self._detections = list(detections)
        self._detection_image_size = (int(img_w), int(img_h))
        self._selected_detection = -1
        self._refresh_bbox_overlay()
        self.update()

    def set_selected_detection(self, idx: int):
        self._selected_detection = int(idx)
        self._refresh_bbox_overlay()
        self.update()

    def _refresh_bbox_overlay(self):
        if self._bbox_overlay is None:
            return
        if self._current_pose is None or not self._detections:
            self._bbox_overlay.clear()
            return
        img_w, img_h = self._detection_image_size
        if img_w <= 0 or img_h <= 0:
            self._bbox_overlay.clear()
            return
        R = rotation_from_angle(
            self._current_pose.roll,
            self._current_pose.pitch,
            self._current_pose.yaw,
        )
        yaw_offset = self._pano.yaw_offset_deg if self._pano is not None else 0.0
        self._bbox_overlay.set_detections(
            self._detections,
            img_w,
            img_h,
            R,
            yaw_offset,
            selected_index=self._selected_detection,
        )

    # ------------- OpenGL lifecycle -------------

    def initializeGL(self):
        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self._pano = PanoSphere(self._ctx)
        self._pc = PointCloud(self._ctx)
        self._pc.set_selected_depth_test(self._selected_depth_test)
        self._bbox_overlay = BboxOverlay(self._ctx)

        if self._pending_points is not None:
            self._pc.upload(*self._pending_points)
            if self._global_view_mode:
                self.orbit_camera.fit_to_points(self._pending_points[0])
            self._pending_points = None
        if self._pending_pose is not None:
            pose, img = self._pending_pose
            self._pano.load_image(img)
            self._pano.set_pose(pose.roll, pose.pitch, pose.yaw)
            if not self._global_view_mode:
                self.camera.set_keyframe(
                    pose.position,
                    pose.roll,
                    pose.pitch,
                    pose.yaw,
                    self._pano.yaw_offset_deg,
                )
            self._current_pose = pose
            self._pending_pose = None
        self._refresh_bbox_overlay()

    def resizeGL(self, w: int, h: int):
        if self._ctx is None:
            return
        self._ctx.viewport = (0, 0, *self._framebuffer_size())

    def paintGL(self):
        if self._ctx is None:
            return
        logical_w, logical_h = self._logical_size()
        fb_w, fb_h = self._framebuffer_size()
        self._ctx.viewport = (0, 0, fb_w, fb_h)
        self._ctx.clear(0.07, 0.07, 0.09, 1.0)

        active_camera = self.orbit_camera if self._global_view_mode else self.camera
        proj = active_camera.proj_matrix(logical_w / logical_h)
        view = active_camera.view_matrix()
        mvp = proj @ view

        if self._show_pano and self._pano is not None:
            self._pano.render(mvp, active_camera.position)
        if self._show_pc and self._pc is not None:
            self._pc.render(mvp)
        if self._show_bboxes and self._bbox_overlay is not None:
            self._bbox_overlay.render(mvp, active_camera.position)

    def _logical_size(self) -> tuple[int, int]:
        return max(1, self.width()), max(1, self.height())

    def _framebuffer_size(self) -> tuple[int, int]:
        ratio = max(1.0, float(self.devicePixelRatio()))
        w, h = self._logical_size()
        return max(1, round(w * ratio)), max(1, round(h * ratio))

    # ------------- mouse / keyboard -------------

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._pick_mode:
            det_idx, click_u, click_v = self._pick_detection_hit(e.position().toPoint())
            if det_idx >= 0:
                self.detection_clicked.emit(det_idx, click_u, click_v)
                return
            idx = self._pick_point(e.position().toPoint(), max_dist_px=18)
            self.point_clicked.emit(idx)
            if idx >= 0:
                self.set_highlight(idx)
            return
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = e.position().toPoint()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = False

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        if self._dragging:
            dx = pos.x() - self._last_pos.x()
            dy = pos.y() - self._last_pos.y()
            self._last_pos = pos
            active_camera = self.orbit_camera if self._global_view_mode else self.camera
            active_camera.orbit(-dx * 0.2, -dy * 0.2)
            self._clear_hover()
            self.update()
            return

        self._cursor_pos = pos
        self._hover_timer.start()

    def leaveEvent(self, e):
        self._cursor_pos = QPoint(-1, -1)
        self._clear_hover()

    def wheelEvent(self, e):
        delta = -e.angleDelta().y() / 120.0
        active_camera = self.orbit_camera if self._global_view_mode else self.camera
        if self._global_view_mode:
            active_camera.zoom(delta)
        else:
            active_camera.zoom(delta * 4.0)
        self._clear_hover()
        self.update()

    def keyPressEvent(self, e):
        if not self.handle_key(e.key()):
            super().keyPressEvent(e)

    def handle_key(self, key: int) -> bool:
        if key == Qt.Key_R:
            self.reset_view()
            return True
        if key == Qt.Key_1:
            self.set_show_pano(not self._show_pano)
            return True
        if key == Qt.Key_2:
            self.set_show_pc(not self._show_pc)
            return True
        return False

    # ------------- hover -------------

    def _clear_hover(self):
        if self._pc is not None and self._pc.highlight >= 0:
            self._pc.highlight = -1
            self.hover_changed.emit(None)
            self.update()

    def _do_hover(self):
        if self._pc is None or self._pc.points is None or self._pc.n == 0:
            return
        if self._cursor_pos.x() < 0:
            return
        w, h = self._logical_size()
        active_camera = self.orbit_camera if self._global_view_mode else self.camera
        proj = active_camera.proj_matrix(w / h)
        view = active_camera.view_matrix()
        mvp = (proj @ view).astype(np.float32)
        idx = find_nearest_to_mouse(
            self._pc.points, mvp, w, h,
            self._cursor_pos.x(), self._cursor_pos.y(), max_dist_px=14,
        )
        if idx == self._pc.highlight:
            return
        self._pc.highlight = idx
        if idx >= 0 and self._pc.colors is not None:
            xyz = self._pc.points[idx]
            rgb = self._pc.colors[idx]
            self.hover_changed.emit({
                "index": idx,
                "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]),
                "r": int(rgb[0]), "g": int(rgb[1]), "b": int(rgb[2]),
            })
        else:
            self.hover_changed.emit(None)
        self.update()

    def _pick_point(self, pos: QPoint, max_dist_px: int) -> int:
        if self._pc is None or self._pc.points is None or self._pc.n == 0:
            return -1
        w, h = self._logical_size()
        active_camera = self.orbit_camera if self._global_view_mode else self.camera
        proj = active_camera.proj_matrix(w / h)
        view = active_camera.view_matrix()
        mvp = (proj @ view).astype(np.float32)
        return find_nearest_to_mouse(
            self._pc.points, mvp, w, h,
            pos.x(), pos.y(), max_dist_px=max_dist_px,
        )

    def _pick_detection(self, pos: QPoint) -> int:
        return self._pick_detection_hit(pos)[0]

    def _pick_detection_hit(self, pos: QPoint) -> tuple[int, float, float]:
        if self._global_view_mode or not self._show_bboxes:
            return -1, float("nan"), float("nan")
        if self._current_pose is None or not self._detections:
            return -1, float("nan"), float("nan")
        img_w, img_h = self._detection_image_size
        if img_w <= 0 or img_h <= 0:
            return -1, float("nan"), float("nan")
        ray = self._mouse_world_ray(pos)
        if ray is None:
            return -1, float("nan"), float("nan")
        R = rotation_from_angle(
            self._current_pose.roll,
            self._current_pose.pitch,
            self._current_pose.yaw,
        )
        yaw_offset = self._pano.yaw_offset_deg if self._pano is not None else 0.0
        uv = project_points_to_panorama(
            (self._current_pose.position + ray).reshape(1, 3),
            self._current_pose.position,
            R,
            img_w,
            img_h,
            yaw_offset_deg=yaw_offset,
        )
        matches = match_points_to_detections(uv, self._detections, float(img_w))
        det_idx = int(matches.match_indices[0])
        if det_idx < 0:
            return -1, float("nan"), float("nan")
        return det_idx, float(uv[0, 0]), float(uv[0, 1])

    def _mouse_world_ray(self, pos: QPoint) -> np.ndarray | None:
        w, h = self._logical_size()
        if w <= 0 or h <= 0:
            return None
        aspect = w / h
        ndc_x = (2.0 * float(pos.x()) / float(w)) - 1.0
        ndc_y = 1.0 - (2.0 * float(pos.y()) / float(h))
        tan_half_fov = np.tan(np.deg2rad(self.camera.fov_deg) / 2.0)
        camera_dir = np.array([
            ndc_x * tan_half_fov * aspect,
            ndc_y * tan_half_fov,
            -1.0,
        ], dtype=np.float64)
        camera_dir /= max(float(np.linalg.norm(camera_dir)), 1e-9)
        view = self.camera.view_matrix().astype(np.float64)
        world_dir = view[:3, :3].T @ camera_dir
        world_dir /= max(float(np.linalg.norm(world_dir)), 1e-9)
        return world_dir
