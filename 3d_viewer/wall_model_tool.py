"""Standalone UI for generating and previewing wall model artifacts."""
from __future__ import annotations

import argparse
from io import BytesIO
from dataclasses import dataclass
import sys
from pathlib import Path

import moderngl
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from core.app_logging import log_operation
from core.dataset import Dataset, find_default_dataset, load_dataset
from core.wall_model import (
    OBJ_MATERIAL_COLORS,
    WallLineDraft,
    WallModelResult,
    WallOpeningMarker,
    WallSegment,
    export_wall_model_from_wall_lines,
    generate_wall_lines as build_wall_lines,
    opening_overlays_for_wall_segments,
    render_wall_line_draft_background,
    render_wall_line_draft_preview,
    save_wall_line_draft,
)
from core.wall_openings import WallOpening, load_wall_openings
from render.orbit_camera import OrbitCamera


WORKSPACE = HERE.parent

_VERT_SRC = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec4 in_color;
out vec4 v_color;
void main() {
    v_color = in_color;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

_FRAG_SRC = """
#version 330
in vec4 v_color;
out vec4 frag;
void main() {
    frag = v_color;
}
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open an isolated wall-model generation UI.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE,
        help="Workspace containing the scan dataset and output directory.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=300_000,
        help="Maximum sampled points loaded from the LAS file.",
    )
    parser.add_argument(
        "--auto-generate",
        action="store_true",
        help="Load the default point cloud and generate the wall model after the window opens.",
    )
    return parser


@dataclass(frozen=True)
class ObjTriangleMesh:
    triangles: np.ndarray
    colors: np.ndarray


def load_obj_triangle_mesh(path: Path) -> ObjTriangleMesh:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    current_material = "wall"
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "usemtl" and len(parts) >= 2:
            current_material = parts[1]
            continue
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            continue
        if parts[0] != "f" or len(parts) < 4:
            continue
        material_color = OBJ_MATERIAL_COLORS.get(current_material, OBJ_MATERIAL_COLORS["wall"])
        indices = [_parse_face_index(token, len(vertices)) for token in parts[1:]]
        for i in range(1, len(indices) - 1):
            for index in (indices[0], indices[i], indices[i + 1]):
                triangles.append(vertices[index])
                colors.append(material_color)
    return ObjTriangleMesh(
        triangles=np.asarray(triangles, dtype=np.float32).reshape((-1, 3)),
        colors=np.asarray(colors, dtype=np.float32).reshape((-1, 4)),
    )


def load_obj_triangles(path: Path) -> np.ndarray:
    return load_obj_triangle_mesh(path).triangles


def _parse_face_index(token: str, vertex_count: int) -> int:
    raw = token.split("/", 1)[0]
    index = int(raw)
    if index < 0:
        return vertex_count + index
    return index - 1


def _triangle_edges(triangles: np.ndarray) -> np.ndarray:
    tris = np.asarray(triangles, dtype=np.float32).reshape((-1, 3, 3))
    if len(tris) == 0:
        return np.empty((0, 3), dtype=np.float32)
    edges = np.concatenate(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]],
        axis=1,
    )
    return np.ascontiguousarray(edges.reshape((-1, 3)))


def _pil_image_to_pixmap(image) -> QPixmap:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    return pixmap


class ObjModelView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gl_window = _ObjModelGLWindow()
        self.container = QWidget.createWindowContainer(self.gl_window, self)
        self.container.setFocusPolicy(Qt.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.container)

    def load_obj(self, path: Path):
        self.gl_window.load_obj(path)


class _ObjModelGLWindow(QOpenGLWindow):
    def __init__(self):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        super().__init__()
        self.setFormat(fmt)
        self.camera = OrbitCamera()
        self._ctx: moderngl.Context | None = None
        self._prog: moderngl.Program | None = None
        self._vbo: moderngl.Buffer | None = None
        self._vao: moderngl.VertexArray | None = None
        self._edge_vbo: moderngl.Buffer | None = None
        self._edge_vao: moderngl.VertexArray | None = None
        self._triangles: np.ndarray | None = None
        self._pending_mesh: ObjTriangleMesh | None = None
        self.show_mesh_edges = True
        self._dragging = False
        self._last_pos = QPoint()

    def load_obj(self, path: Path):
        mesh = load_obj_triangle_mesh(path)
        if self._ctx is None:
            self._pending_mesh = mesh
            return
        self._upload(mesh)
        self.update()

    def initializeGL(self):
        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._prog = self._ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)
        if self._pending_mesh is not None:
            self._upload(self._pending_mesh)
            self._pending_mesh = None

    def resizeGL(self, w: int, h: int):
        if self._ctx is not None:
            self._ctx.viewport = (0, 0, *self._framebuffer_size())

    def paintGL(self):
        if self._ctx is None:
            return
        fb_w, fb_h = self._framebuffer_size()
        logical_w, logical_h = self._logical_size()
        self._ctx.viewport = (0, 0, fb_w, fb_h)
        self._ctx.clear(0.06, 0.06, 0.07, 1.0)
        if self._vao is None or self._prog is None:
            return
        mvp = self.camera.proj_matrix(logical_w / logical_h) @ self.camera.view_matrix()
        self._prog["mvp"].write(mvp.T.tobytes())
        self._vao.render(mode=moderngl.TRIANGLES)
        if self.show_mesh_edges and self._edge_vao is not None:
            self._ctx.disable(moderngl.DEPTH_TEST)
            self._edge_vao.render(mode=moderngl.LINES)
            self._ctx.enable(moderngl.DEPTH_TEST)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.position().toPoint()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        self.camera.orbit(-dx * 0.25, -dy * 0.25)
        self.update()

    def wheelEvent(self, event):
        delta = -event.angleDelta().y() / 120.0
        self.camera.zoom(delta)
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.camera.reset_view()
            self.update()
            return
        super().keyPressEvent(event)

    def _upload(self, mesh: ObjTriangleMesh):
        self._release_buffers()
        positions = np.asarray(mesh.triangles, dtype=np.float32).reshape((-1, 3))
        colors = np.asarray(mesh.colors, dtype=np.float32).reshape((-1, 4))
        if len(colors) != len(positions):
            colors = np.tile(np.asarray(OBJ_MATERIAL_COLORS["wall"], dtype=np.float32), (len(positions), 1))
        self._triangles = positions
        if len(positions) == 0 or self._ctx is None or self._prog is None:
            return
        self.camera.fit_to_points(positions)
        data = np.ascontiguousarray(np.column_stack([positions, colors]).astype(np.float32))
        self._vbo = self._ctx.buffer(data.tobytes())
        self._vao = self._ctx.vertex_array(self._prog, [(self._vbo, "3f 4f", "in_pos", "in_color")])
        edges = _triangle_edges(positions)
        edge_colors = np.tile(np.asarray((0.05, 0.08, 0.09, 1.0), dtype=np.float32), (len(edges), 1))
        edge_data = np.ascontiguousarray(np.column_stack([edges, edge_colors]).astype(np.float32))
        self._edge_vbo = self._ctx.buffer(edge_data.tobytes())
        self._edge_vao = self._ctx.vertex_array(self._prog, [(self._edge_vbo, "3f 4f", "in_pos", "in_color")])

    def _release_buffers(self):
        for obj in (self._vao, self._edge_vao, self._vbo, self._edge_vbo):
            if obj is not None:
                obj.release()
        self._vao = None
        self._edge_vao = None
        self._vbo = None
        self._edge_vbo = None

    def _logical_size(self) -> tuple[int, int]:
        return max(1, self.width()), max(1, self.height())

    def _framebuffer_size(self) -> tuple[int, int]:
        ratio = max(1.0, float(self.devicePixelRatio()))
        w, h = self._logical_size()
        return max(1, round(w * ratio)), max(1, round(h * ratio))


class WallLineEditor(QWidget):
    lines_changed = Signal()
    edit_finished = Signal(str)
    endpoint_hit_radius_px = 12.0
    line_hit_radius_px = 14.0
    endpoint_snap_radius_px = 5.0
    min_zoom = 0.25
    max_zoom = 8.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("wallLineEditor")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._draft: WallLineDraft | None = None
        self._segments: list[WallSegment] = []
        self._selected_index: int | None = None
        self._background = QPixmap()
        self._drag_part: str | None = None
        self._drag_last_world: tuple[float, float] | None = None
        self._add_mode = False
        self._add_start_world: tuple[float, float] | None = None
        self._add_current_world: tuple[float, float] | None = None
        self._snap_target_world: tuple[float, float] | None = None
        self._opening_markers: list[WallOpeningMarker] = []
        self._projected_openings: list[WallOpening] = []
        self._zoom_scale = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_last_pos = QPointF(0.0, 0.0)

    def set_draft(self, draft: WallLineDraft):
        self._draft = draft
        self._segments = list(draft.segments)
        self._selected_index = None
        self._drag_part = None
        self._add_start_world = None
        self._add_current_world = None
        self._snap_target_world = None
        self._opening_markers = []
        self._projected_openings = []
        self.reset_zoom(update=False)
        self._background = _pil_image_to_pixmap(render_wall_line_draft_background(draft))
        self.update()

    def draft(self) -> WallLineDraft | None:
        return self._draft

    def wall_segments(self) -> list[WallSegment]:
        return list(self._segments)

    def set_opening_overlays(
        self,
        opening_markers: list[WallOpeningMarker],
        projected_openings: list[WallOpening],
    ):
        self._opening_markers = list(opening_markers)
        self._projected_openings = list(projected_openings)
        self.update()

    def opening_overlay_counts(self) -> tuple[int, int]:
        return len(self._opening_markers), len(self._projected_openings)

    def select_segment(self, index: int | None):
        if index is None or index < 0 or index >= len(self._segments):
            self._selected_index = None
        else:
            self._selected_index = int(index)
        self.update()

    def delete_selected_segment(self):
        if self._selected_index is None:
            return
        del self._segments[self._selected_index]
        self._selected_index = None
        self.lines_changed.emit()
        self.edit_finished.emit("delete_segment")
        self.update()

    def set_add_mode(self, enabled: bool):
        self._add_mode = bool(enabled)
        self._add_start_world = None
        self._add_current_world = None
        self.setCursor(Qt.CrossCursor if self._add_mode else Qt.ArrowCursor)
        self.update()

    def add_axis_aligned_segment(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        z_min: float = 0.0,
        z_max: float = 2.6,
    ):
        segment = self._make_axis_aligned_segment(start, end, z_min=z_min, z_max=z_max)
        if segment is None:
            return
        self._segments.append(segment)
        self._selected_index = len(self._segments) - 1
        self.lines_changed.emit()
        self.edit_finished.emit("add_segment")
        self.update()

    def zoom_factor(self) -> float:
        return float(self._zoom_scale)

    def zoom_by(self, factor: float, *, anchor: QPointF | QPoint | None = None):
        if factor <= 0.0:
            return
        if anchor is None:
            anchor_pos = QPointF(self.width() / 2.0, self.height() / 2.0)
        else:
            anchor_pos = QPointF(float(anchor.x()), float(anchor.y()))
        anchor_world = self._widget_to_world(anchor_pos)
        next_scale = max(self.min_zoom, min(self.max_zoom, self._zoom_scale * float(factor)))
        if abs(next_scale - self._zoom_scale) <= 1e-9:
            return
        self._zoom_scale = next_scale
        if anchor_world is not None:
            after = self._world_to_widget(anchor_world[0], anchor_world[1])
            if after is not None:
                self._pan_offset += anchor_pos - after
        self.update()

    def reset_zoom(self, *, update: bool = True):
        self._zoom_scale = 1.0
        self._pan_offset = QPointF(0.0, 0.0)
        if update:
            self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b0d0f"))
        image_rect = self._image_rect()
        if not image_rect.isNull():
            if not self._background.isNull():
                painter.drawPixmap(image_rect.toRect(), self._background)
            else:
                painter.fillRect(image_rect, QColor("#020304"))
        self._draw_segments(painter, image_rect)
        self._draw_opening_overlays(painter, image_rect)
        if self._snap_target_world is not None:
            target = self._world_to_widget(*self._snap_target_world, image_rect)
            if target is not None:
                snap_pen = QPen(QColor("#32d583"), 2)
                snap_pen.setCosmetic(True)
                painter.setPen(snap_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(target, 8, 8)
        if self._add_mode and self._add_start_world and self._add_current_world:
            preview = self._make_axis_aligned_segment(self._add_start_world, self._add_current_world)
            if preview is not None:
                self._draw_segment(painter, image_rect, preview, selected=True, preview=True)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.setFocus(Qt.MouseFocusReason)
            self._panning = True
            self._pan_last_pos = QPointF(event.position())
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.LeftButton or self._draft is None:
            super().mousePressEvent(event)
            return
        world = self._widget_to_world(event.position())
        if world is None:
            return
        self.setFocus(Qt.MouseFocusReason)
        if self._add_mode:
            self._add_start_world = world
            self._add_current_world = world
            self.update()
            return
        index, part = self._hit_test(event.position())
        self.select_segment(index)
        self._drag_part = part
        self._drag_last_world = world if index is not None else None

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            pos = QPointF(event.position())
            self._pan_offset += pos - self._pan_last_pos
            self._pan_last_pos = pos
            self.update()
            event.accept()
            return
        if self._draft is None:
            return
        world = self._widget_to_world(event.position())
        if world is None:
            return
        if self._add_mode and self._add_start_world:
            self._add_current_world = world
            self.update()
            return
        if self._selected_index is None or self._drag_part is None or self._drag_last_world is None:
            return
        self._move_selected(world)
        self._drag_last_world = world
        self.lines_changed.emit()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = False
            self.setCursor(Qt.CrossCursor if self._add_mode else Qt.ArrowCursor)
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._add_mode and self._add_start_world and self._add_current_world:
            z_min, z_max = self._default_z_range()
            self.add_axis_aligned_segment(self._add_start_world, self._add_current_world, z_min=z_min, z_max=z_max)
            self._add_start_world = None
            self._add_current_world = None
            self.set_add_mode(False)
            return
        if self._drag_part is not None:
            self.edit_finished.emit("drag_segment")
        self._drag_part = None
        self._drag_last_world = None
        self._snap_target_world = None
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_segment()
            return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_by(1.2)
            return
        if event.key() == Qt.Key_Minus:
            self.zoom_by(1.0 / 1.2)
            return
        if event.key() == Qt.Key_0:
            self.reset_zoom()
            return
        if event.key() == Qt.Key_Escape:
            self.set_add_mode(False)
            self.select_segment(None)
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        self.zoom_by(1.2 if delta > 0 else 1.0 / 1.2, anchor=event.position())
        event.accept()

    def _draw_segments(self, painter: QPainter, image_rect: QRectF):
        for index, segment in enumerate(self._segments):
            self._draw_segment(
                painter,
                image_rect,
                segment,
                selected=index == self._selected_index,
                preview=False,
            )

    def _draw_segment(
        self,
        painter: QPainter,
        image_rect: QRectF,
        segment: WallSegment,
        *,
        selected: bool,
        preview: bool,
    ):
        p1 = self._world_to_widget(segment.x1, segment.y1, image_rect)
        p2 = self._world_to_widget(segment.x2, segment.y2, image_rect)
        if p1 is None or p2 is None:
            return
        color = QColor("#ffd43b") if selected else QColor("#ff4628")
        if preview:
            color = QColor("#32d583")
        pen = QPen(color, 4 if selected else 3, Qt.DashLine if preview else Qt.SolidLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        if selected:
            handle_pen = QPen(QColor("#101418"), 2)
            handle_pen.setCosmetic(True)
            painter.setPen(handle_pen)
            painter.setBrush(QColor("#ffd43b"))
            painter.drawEllipse(p1, 5, 5)
            painter.drawEllipse(p2, 5, 5)

    def _draw_opening_overlays(self, painter: QPainter, image_rect: QRectF):
        for opening in self._projected_openings:
            p1 = self._world_to_widget(opening.bbox_min[0], opening.bbox_min[1], image_rect)
            p2 = self._world_to_widget(opening.bbox_max[0], opening.bbox_max[1], image_rect)
            if p1 is None or p2 is None:
                continue
            rect = QRectF(p1, p2).normalized()
            outer_pen = QPen(QColor("#ff9623"), 4)
            outer_pen.setCosmetic(True)
            painter.setPen(outer_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
            inner_pen = QPen(QColor("#140f05"), 1)
            inner_pen.setCosmetic(True)
            painter.setPen(inner_pen)
            painter.drawRect(rect)

        for marker in self._opening_markers:
            if marker.orientation == "vertical":
                p1 = self._world_to_widget(marker.wall_coord, marker.axis_min, image_rect)
                p2 = self._world_to_widget(marker.wall_coord, marker.axis_max, image_rect)
            else:
                p1 = self._world_to_widget(marker.axis_min, marker.wall_coord, image_rect)
                p2 = self._world_to_widget(marker.axis_max, marker.wall_coord, image_rect)
            if p1 is None or p2 is None:
                continue
            color = QColor("#28e66e") if marker.label == "window" else QColor("#ffcd28")
            outer_pen = QPen(color, 7)
            outer_pen.setCosmetic(True)
            painter.setPen(outer_pen)
            painter.drawLine(p1, p2)
            inner_pen = QPen(QColor("#0a140f"), 2)
            inner_pen.setCosmetic(True)
            painter.setPen(inner_pen)
            painter.drawLine(p1, p2)

    def _image_rect(self) -> QRectF:
        base = self._base_image_rect()
        if self._draft is None or base.isNull():
            return base
        width = base.width() * self._zoom_scale
        height = base.height() * self._zoom_scale
        center = base.center() + self._pan_offset
        return QRectF(center.x() - width / 2.0, center.y() - height / 2.0, width, height)

    def _base_image_rect(self) -> QRectF:
        if self._draft is None:
            return QRectF(self.rect())
        width_px = max(1, int(self._draft.grid_shape[1]))
        height_px = max(1, int(self._draft.grid_shape[0]))
        available = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        if available.width() <= 0.0 or available.height() <= 0.0:
            return QRectF()
        scale = min(available.width() / width_px, available.height() / height_px)
        width = width_px * scale
        height = height_px * scale
        return QRectF(
            available.x() + (available.width() - width) / 2.0,
            available.y() + (available.height() - height) / 2.0,
            width,
            height,
        )

    def _world_to_widget(self, x: float, y: float, image_rect: QRectF | None = None) -> QPointF | None:
        if self._draft is None:
            return None
        rect = image_rect or self._image_rect()
        if rect.isNull():
            return None
        width_px = max(1, int(self._draft.grid_shape[1]))
        height_px = max(1, int(self._draft.grid_shape[0]))
        px = (float(x) - self._draft.x_min) / self._draft.resolution_m
        py = height_px - 1 - (float(y) - self._draft.y_min) / self._draft.resolution_m
        return QPointF(
            rect.x() + px / width_px * rect.width(),
            rect.y() + py / height_px * rect.height(),
        )

    def _widget_to_world(self, pos: QPointF) -> tuple[float, float] | None:
        if self._draft is None:
            return None
        rect = self._image_rect()
        if rect.isNull() or not rect.contains(pos):
            return None
        width_px = max(1, int(self._draft.grid_shape[1]))
        height_px = max(1, int(self._draft.grid_shape[0]))
        px = (pos.x() - rect.x()) / rect.width() * width_px
        py = (pos.y() - rect.y()) / rect.height() * height_px
        x = self._draft.x_min + px * self._draft.resolution_m
        y = self._draft.y_min + (height_px - 1 - py) * self._draft.resolution_m
        return float(x), float(y)

    def _hit_test(self, pos: QPointF) -> tuple[int | None, str | None]:
        rect = self._image_rect()
        best_endpoint_index: int | None = None
        best_endpoint_part: str | None = None
        best_endpoint_distance = self.endpoint_hit_radius_px
        for index, segment in enumerate(self._segments):
            p1 = self._world_to_widget(segment.x1, segment.y1, rect)
            p2 = self._world_to_widget(segment.x2, segment.y2, rect)
            if p1 is None or p2 is None:
                continue
            start_distance = _distance_point_to_point(pos, p1)
            end_distance = _distance_point_to_point(pos, p2)
            if start_distance <= best_endpoint_distance:
                best_endpoint_distance = start_distance
                best_endpoint_index = index
                best_endpoint_part = "start"
            if end_distance <= best_endpoint_distance:
                best_endpoint_distance = end_distance
                best_endpoint_index = index
                best_endpoint_part = "end"
        if best_endpoint_index is not None:
            return best_endpoint_index, best_endpoint_part

        best_index: int | None = None
        best_distance = self.line_hit_radius_px
        for index, segment in enumerate(self._segments):
            p1 = self._world_to_widget(segment.x1, segment.y1, rect)
            p2 = self._world_to_widget(segment.x2, segment.y2, rect)
            if p1 is None or p2 is None:
                continue
            line_distance = _distance_point_to_segment(pos, p1, p2)
            if line_distance < best_distance:
                best_distance = line_distance
                best_index = index
        if best_index is None:
            return None, None
        return best_index, "body"

    def _move_selected(self, world: tuple[float, float]):
        if self._selected_index is None or self._drag_last_world is None:
            return
        segment = self._segments[self._selected_index]
        x, y = world
        last_x, last_y = self._drag_last_world
        if self._drag_part == "start":
            updated = self._move_endpoint(segment, "start", world)
        elif self._drag_part == "end":
            updated = self._move_endpoint(segment, "end", world)
        else:
            self._snap_target_world = None
            dx = x - last_x
            dy = y - last_y
            updated = self._copy_segment_geometry(
                segment,
                segment.x1 + dx,
                segment.y1 + dy,
                segment.x2 + dx,
                segment.y2 + dy,
            )
        if updated.length_m > 0.05:
            self._segments[self._selected_index] = updated

    def _move_endpoint(self, segment: WallSegment, part: str, world: tuple[float, float]) -> WallSegment:
        snapped = self._snap_endpoint((float(world[0]), float(world[1])))
        if part == "start":
            return self._copy_segment_geometry(segment, snapped[0], snapped[1], segment.x2, segment.y2)
        return self._copy_segment_geometry(segment, segment.x1, segment.y1, snapped[0], snapped[1])

    def _snap_endpoint(self, moving: tuple[float, float]) -> tuple[float, float]:
        target = self._nearest_other_endpoint(moving)
        self._snap_target_world = target
        if target is None:
            return moving
        return float(target[0]), float(target[1])

    def _nearest_other_endpoint(self, moving: tuple[float, float]) -> tuple[float, float] | None:
        rect = self._image_rect()
        moving_pos = self._world_to_widget(moving[0], moving[1], rect)
        if moving_pos is None:
            return None
        best_endpoint: tuple[float, float] | None = None
        best_distance = self.endpoint_snap_radius_px
        for index, segment in enumerate(self._segments):
            if index == self._selected_index:
                continue
            for endpoint in ((segment.x1, segment.y1), (segment.x2, segment.y2)):
                endpoint_pos = self._world_to_widget(endpoint[0], endpoint[1], rect)
                if endpoint_pos is None:
                    continue
                distance = _distance_point_to_point(moving_pos, endpoint_pos)
                if distance <= best_distance:
                    best_distance = distance
                    best_endpoint = (float(endpoint[0]), float(endpoint[1]))
        return best_endpoint

    def _copy_segment_geometry(
        self,
        source: WallSegment,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> WallSegment:
        return self._make_wall_segment(
            (x1, y1),
            (x2, y2),
            z_min=source.z_min,
            z_max=source.z_max,
            point_count=source.point_count,
            height_span_m=source.height_span_m,
        ) or source

    def _make_wall_segment(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        z_min: float = 0.0,
        z_max: float = 2.6,
        point_count: int = 0,
        height_span_m: float | None = None,
    ) -> WallSegment | None:
        x1, y1 = start
        x2, y2 = end
        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)
        length = float((dx * dx + dy * dy) ** 0.5)
        if length < 0.05:
            return None
        orientation = _segment_orientation_from_delta(dx, dy)
        if orientation == "vertical":
            x = (float(x1) + float(x2)) / 2.0
            x1 = x2 = x
            length = abs(float(y2) - float(y1))
        elif orientation == "horizontal":
            y = (float(y1) + float(y2)) / 2.0
            y1 = y2 = y
            length = abs(float(x2) - float(x1))
        span = float(height_span_m if height_span_m is not None else z_max - z_min)
        return WallSegment(
            orientation,
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            float(z_min),
            float(z_max),
            float(length),
            int(point_count),
            span,
        )

    def _make_axis_aligned_segment(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        z_min: float = 0.0,
        z_max: float = 2.6,
        point_count: int = 0,
        height_span_m: float | None = None,
        prefer_orientation: str | None = None,
    ) -> WallSegment | None:
        x1, y1 = start
        x2, y2 = end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if max(dx, dy) < 0.05:
            return None
        orientation = prefer_orientation if prefer_orientation in {"horizontal", "vertical"} else None
        if orientation is None:
            orientation = "horizontal" if dx >= dy else "vertical"
        if orientation == "vertical":
            x = (x1 + x2) / 2.0
            length = dy
            x1 = x2 = x
        else:
            y = (y1 + y2) / 2.0
            length = dx
            y1 = y2 = y
        span = float(height_span_m if height_span_m is not None else z_max - z_min)
        return WallSegment(
            orientation,
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            float(z_min),
            float(z_max),
            float(length),
            int(point_count),
            span,
        )

    def _default_z_range(self) -> tuple[float, float]:
        if self._segments:
            z_min = min(segment.z_min for segment in self._segments)
            z_max = max(segment.z_max for segment in self._segments)
            return float(z_min), float(z_max)
        return 0.0, 2.6


def _segment_orientation_from_delta(dx: float, dy: float, *, tolerance: float = 1e-6) -> str:
    if abs(dx) <= tolerance:
        return "vertical"
    if abs(dy) <= tolerance:
        return "horizontal"
    return "free"


def _distance_point_to_point(a: QPointF, b: QPointF) -> float:
    return float(((a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2) ** 0.5)


def _distance_point_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
    ax, ay = a.x(), a.y()
    bx, by = b.x(), b.y()
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return _distance_point_to_point(p, a)
    t = max(0.0, min(1.0, ((p.x() - ax) * dx + (p.y() - ay) * dy) / denom))
    x = ax + t * dx
    y = ay + t * dy
    return float(((p.x() - x) ** 2 + (p.y() - y) ** 2) ** 0.5)


class WallModelWorkbench(QWidget):
    status_changed = Signal(str)
    back_requested = Signal()

    def __init__(self, workspace: Path, *, max_points: int, parent=None):
        super().__init__(parent)
        self.setObjectName("wallModelWorkbench")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.workspace = Path(workspace)
        self.dataset: Dataset | None = None
        self.wall_line_draft: WallLineDraft | None = None
        self.wall_openings: list[WallOpening] = []
        self.result: WallModelResult | None = None

        self.workspace_label = QLabel(self.workspace.name)
        self.workspace_label.setToolTip(str(self.workspace))
        self.workspace_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.max_points = QSpinBox()
        self.max_points.setRange(10_000, 5_000_000)
        self.max_points.setSingleStep(50_000)
        self.max_points.setValue(int(max_points))

        self.resolution = QDoubleSpinBox()
        self.resolution.setRange(0.02, 0.30)
        self.resolution.setDecimals(3)
        self.resolution.setSingleStep(0.01)
        self.resolution.setValue(0.05)

        self.load_button = QPushButton("加载默认点云")
        self.load_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.load_button.clicked.connect(self.load_default_dataset)
        self.generate_lines_button = QPushButton("生成墙体线")
        self.generate_lines_button.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.generate_lines_button.clicked.connect(self.generate_wall_lines_action)
        self.export_obj_button = QPushButton("从墙体线生成 OBJ")
        self.export_obj_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.export_obj_button.setEnabled(False)
        self.export_obj_button.clicked.connect(self.export_wall_model_action)
        self.save_lines_button = QPushButton("保存墙体线")
        self.save_lines_button.setEnabled(False)
        self.save_lines_button.clicked.connect(self.save_current_wall_lines)
        self.add_line_button = QPushButton("添加墙线")
        self.add_line_button.setEnabled(False)
        self.add_line_button.clicked.connect(self.start_add_wall_line)
        self.delete_line_button = QPushButton("删除选中墙线")
        self.delete_line_button.setEnabled(False)
        self.delete_line_button.clicked.connect(self.delete_selected_wall_line)
        self.zoom_in_button = QPushButton("放大")
        self.zoom_in_button.setEnabled(False)
        self.zoom_in_button.clicked.connect(lambda: self.line_editor.zoom_by(1.2))
        self.zoom_out_button = QPushButton("缩小")
        self.zoom_out_button.setEnabled(False)
        self.zoom_out_button.clicked.connect(lambda: self.line_editor.zoom_by(1.0 / 1.2))
        self.reset_zoom_button = QPushButton("重置视图")
        self.reset_zoom_button.setEnabled(False)
        self.reset_zoom_button.clicked.connect(lambda: self.line_editor.reset_zoom())
        self.back_button = QPushButton("返回主视图")
        self.back_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.back_button.clicked.connect(self.back_requested)
        self.back_button.hide()

        self.info = QLabel("尚未加载点云")
        self.info.setObjectName("wallModelInfo")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.line_editor = WallLineEditor()
        self.line_editor.lines_changed.connect(self._on_wall_lines_changed)
        self.line_editor.edit_finished.connect(self._on_wall_line_edit_finished)
        self.model_view = ObjModelView()
        self.model_view.setMinimumSize(520, 420)

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        toolbar = QWidget()
        toolbar.setObjectName("wallModelToolbar")
        controls = QHBoxLayout(toolbar)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setSpacing(8)
        controls.addWidget(QLabel("workspace"))
        controls.addWidget(self.workspace_label, 1)
        controls.addWidget(QLabel("max points"))
        controls.addWidget(self.max_points)
        controls.addWidget(QLabel("resolution"))
        controls.addWidget(self.resolution)
        controls.addWidget(self.load_button)
        controls.addWidget(self.generate_lines_button)
        controls.addWidget(self.export_obj_button)
        controls.addWidget(self.back_button)
        layout.addWidget(toolbar)
        layout.addWidget(self.info)

        edit_toolbar = QWidget()
        edit_toolbar.setObjectName("wallLineEditToolbar")
        edit_controls = QHBoxLayout(edit_toolbar)
        edit_controls.setContentsMargins(12, 8, 12, 8)
        edit_controls.setSpacing(8)
        edit_controls.addWidget(QLabel("墙线编辑"))
        edit_controls.addWidget(self.add_line_button)
        edit_controls.addWidget(self.delete_line_button)
        edit_controls.addWidget(self.save_lines_button)
        edit_controls.addWidget(QLabel("视图"))
        edit_controls.addWidget(self.zoom_in_button)
        edit_controls.addWidget(self.zoom_out_button)
        edit_controls.addWidget(self.reset_zoom_button)
        edit_controls.addStretch(1)
        layout.addWidget(edit_toolbar)

        previews = QHBoxLayout()
        previews.setSpacing(10)
        previews.addWidget(self.line_editor, 1)
        previews.addWidget(self.model_view, 1)
        layout.addLayout(previews, 1)

    def set_dataset(self, dataset: Dataset | None):
        if self.dataset is dataset:
            return
        self.dataset = dataset
        self.wall_line_draft = None
        self.wall_openings = []
        self.result = None
        self.line_editor.set_opening_overlays([], [])
        self._set_wall_line_actions_enabled(False)
        if dataset is None:
            self.info.setText("尚未加载点云")
            return
        self.info.setText(
            f"已加载: {dataset.data_root.name}\n"
            f"点云: {dataset.points.shape[0]:,}/{dataset.total_points:,} "
            f"(step {dataset.sample_step})\n"
            f"LAS: {dataset.pointcloud_file}"
        )

    def load_default_dataset(self):
        self._show_status("正在加载默认点云...")
        QApplication.processEvents()
        try:
            cfg = find_default_dataset(self.workspace)
            if cfg is None:
                raise RuntimeError(f"no dataset found under {self.workspace}")
            self.set_dataset(load_dataset(cfg, max_points=int(self.max_points.value())))
        except Exception as exc:
            self.set_dataset(None)
            QMessageBox.warning(self, "点云加载失败", str(exc))
            self._show_status("点云加载失败")
            return
        self._show_status("默认点云已加载")

    def generate_wall_lines_action(self) -> bool:
        if self.dataset is None:
            self.load_default_dataset()
            if self.dataset is None:
                return False
        self.generate_lines_button.setEnabled(False)
        self._show_status("正在生成墙体线...")
        self._log_operation(
            "wall_lines_generation_started",
            data_root=self.dataset.data_root,
            point_count=int(self.dataset.points.shape[0]),
            resolution_m=float(self.resolution.value()),
        )
        QApplication.processEvents()
        try:
            self.wall_line_draft = build_wall_lines(
                self.workspace,
                self.dataset.data_root,
                self.dataset.points,
                resolution_m=float(self.resolution.value()),
            )
        except Exception as exc:
            self._log_operation("wall_lines_generation_failed", level="error", error=exc)
            QMessageBox.warning(self, "墙体线生成失败", str(exc))
            self._show_status("墙体线生成失败")
            return False
        finally:
            self.generate_lines_button.setEnabled(True)
        self._show_wall_lines(self.wall_line_draft)
        self._log_operation(
            "wall_lines_generation_finished",
            wall_lines_path=self.wall_line_draft.wall_lines_path,
            topdown_preview_path=self.wall_line_draft.topdown_preview_path,
            segment_count=self.wall_line_draft.segment_count,
        )
        return True

    def export_wall_model_action(self) -> bool:
        if self.dataset is None:
            QMessageBox.warning(self, "缺少点云", "请先加载点云并生成墙体线。")
            return False
        if self.wall_line_draft is None:
            QMessageBox.warning(self, "缺少墙体线", "请先生成或加载墙体线。")
            return False
        self.export_obj_button.setEnabled(False)
        self._show_status("正在从已确认墙体线生成 OBJ...")
        self._log_operation(
            "wall_model_export_started",
            data_root=self.dataset.data_root,
            wall_line_count=len(self.line_editor.wall_segments()),
        )
        QApplication.processEvents()
        try:
            draft = self._current_wall_line_draft(source="manual_confirmed_wall_lines")
            render_wall_line_draft_preview(draft).save(draft.topdown_preview_path)
            save_wall_line_draft(
                self.workspace,
                self.dataset.data_root,
                draft.segments,
                x_min=draft.x_min,
                y_min=draft.y_min,
                x_max=draft.x_max,
                y_max=draft.y_max,
                resolution_m=draft.resolution_m,
                source=draft.source,
                grid_shape=draft.grid_shape,
                topdown_preview_path=draft.topdown_preview_path,
            )
            openings = load_wall_openings(self.workspace, self.dataset.data_root)
            self.result = export_wall_model_from_wall_lines(
                self.workspace,
                self.dataset.data_root,
                draft,
                openings=openings,
                resolution_m=float(self.resolution.value()),
            )
        except Exception as exc:
            self._log_operation("wall_model_export_failed", level="error", error=exc)
            QMessageBox.warning(self, "墙体模型生成失败", str(exc))
            self._show_status("墙体模型生成失败")
            self.export_obj_button.setEnabled(True)
            return False
        self.wall_line_draft = draft
        self._show_result(self.result)
        self._log_operation(
            "wall_model_export_finished",
            obj_path=self.result.obj_path,
            metadata_path=self.result.metadata_path,
            segment_count=self.result.segment_count,
            vertex_count=self.result.vertex_count,
            face_count=self.result.face_count,
        )
        self.export_obj_button.setEnabled(True)
        return True

    def generate_wall_artifacts(self):
        if self.generate_wall_lines_action():
            self.export_wall_model_action()

    def save_current_wall_lines(self):
        if self.dataset is None or self.wall_line_draft is None:
            return
        draft = self._current_wall_line_draft(source="manual_saved_wall_lines")
        render_wall_line_draft_preview(draft).save(draft.topdown_preview_path)
        path = save_wall_line_draft(
            self.workspace,
            self.dataset.data_root,
            draft.segments,
            x_min=draft.x_min,
            y_min=draft.y_min,
            x_max=draft.x_max,
            y_max=draft.y_max,
            resolution_m=draft.resolution_m,
            source=draft.source,
            grid_shape=draft.grid_shape,
            topdown_preview_path=draft.topdown_preview_path,
        )
        self.wall_line_draft = draft
        self._show_status(f"墙体线已保存: {path}")
        self._log_operation("wall_lines_saved", wall_lines_path=path, segment_count=draft.segment_count)

    def start_add_wall_line(self):
        self.line_editor.set_add_mode(True)
        self._show_status("在墙体线图上拖拽新增一条水平或垂直墙线")

    def delete_selected_wall_line(self):
        self.line_editor.delete_selected_segment()

    def _show_result(self, result: WallModelResult):
        self.model_view.load_obj(result.obj_path)
        self.info.setText(
            f"OBJ: {result.obj_path}\n"
            f"墙体线图: {result.topdown_preview_path}\n"
            f"模型预览: {result.preview_path}\n"
            f"元数据: {result.metadata_path}\n"
            f"门窗标记: {result.matched_opening_count} matched · "
            f"{result.projected_opening_count} projected · "
            f"{result.unmatched_opening_count} unmatched\n"
            f"墙体段数: {result.segment_count} · 顶点: {result.vertex_count} · 面: {result.face_count}"
        )
        self._show_status(f"墙体模型已生成: {result.segment_count} 段")

    def _show_wall_lines(self, draft: WallLineDraft):
        self.line_editor.set_draft(draft)
        self._load_wall_opening_overlays()
        self.add_line_button.setEnabled(True)
        self.save_lines_button.setEnabled(True)
        self.zoom_in_button.setEnabled(True)
        self.zoom_out_button.setEnabled(True)
        self.reset_zoom_button.setEnabled(True)
        self.export_obj_button.setEnabled(draft.segment_count > 0)
        self.delete_line_button.setEnabled(draft.segment_count > 0)
        matched_count, projected_count = self.line_editor.opening_overlay_counts()
        self.info.setText(
            f"墙体线: {draft.wall_lines_path}\n"
            f"墙体线图: {draft.topdown_preview_path}\n"
            f"墙体段数: {draft.segment_count}\n"
            f"门窗标记: {matched_count} matched · {projected_count} projected\n"
            "确认或编辑墙体线后，再点击“从墙体线生成 OBJ”。"
        )
        self._show_status(f"墙体线已生成: {draft.segment_count} 段")

    def _current_wall_line_draft(self, *, source: str) -> WallLineDraft:
        if self.wall_line_draft is None:
            raise RuntimeError("wall line draft is not available")
        return WallLineDraft(
            wall_lines_path=self.wall_line_draft.wall_lines_path,
            topdown_preview_path=self.wall_line_draft.topdown_preview_path,
            segments=self.line_editor.wall_segments(),
            x_min=self.wall_line_draft.x_min,
            y_min=self.wall_line_draft.y_min,
            x_max=self.wall_line_draft.x_max,
            y_max=self.wall_line_draft.y_max,
            resolution_m=self.wall_line_draft.resolution_m,
            grid_shape=self.wall_line_draft.grid_shape,
            grid=self.wall_line_draft.grid,
            wall_mask=self.wall_line_draft.wall_mask,
            source=source,
        )

    def _on_wall_lines_changed(self):
        has_lines = len(self.line_editor.wall_segments()) > 0
        self.export_obj_button.setEnabled(has_lines)
        self.save_lines_button.setEnabled(self.wall_line_draft is not None)
        self.delete_line_button.setEnabled(has_lines)
        self._refresh_wall_opening_overlays()

    def _on_wall_line_edit_finished(self, action: str):
        self._log_operation(
            "wall_lines_edited",
            action=action,
            wall_line_count=len(self.line_editor.wall_segments()),
        )

    def _set_wall_line_actions_enabled(self, enabled: bool):
        self.export_obj_button.setEnabled(bool(enabled))
        self.save_lines_button.setEnabled(bool(enabled))
        self.add_line_button.setEnabled(bool(enabled))
        self.delete_line_button.setEnabled(bool(enabled))
        self.zoom_in_button.setEnabled(bool(enabled))
        self.zoom_out_button.setEnabled(bool(enabled))
        self.reset_zoom_button.setEnabled(bool(enabled))

    def _load_wall_opening_overlays(self):
        if self.dataset is None:
            self.wall_openings = []
        else:
            self.wall_openings = load_wall_openings(self.workspace, self.dataset.data_root)
        self._refresh_wall_opening_overlays()

    def _refresh_wall_opening_overlays(self):
        if not self.wall_openings:
            self.line_editor.set_opening_overlays([], [])
            return
        markers, projected, _unmatched = opening_overlays_for_wall_segments(
            self.wall_openings,
            self.line_editor.wall_segments(),
        )
        self.line_editor.set_opening_overlays(markers, projected)

    def _show_status(self, message: str):
        self.status_changed.emit(message)

    def set_back_button_visible(self, visible: bool):
        self.back_button.setVisible(bool(visible))

    def _log_operation(self, event: str, *, level: str = "info", **fields):
        try:
            log_operation(self.workspace, event, component="wall_model", level=level, **fields)
        except Exception:
            return

    def _apply_style(self):
        self.setStyleSheet(
            """
            #wallModelWorkbench {
                background: #0f1214;
                color: #e8ecef;
            }
            #wallModelToolbar {
                background: #151b1e;
                border: 1px solid #2f3b41;
                border-radius: 6px;
            }
            #wallLineEditToolbar {
                background: #111619;
                border: 1px solid #273237;
                border-radius: 6px;
            }
            #wallModelToolbar QLabel, #wallLineEditToolbar QLabel {
                color: #f0f5f7;
                font-weight: 600;
            }
            #wallModelInfo {
                background: #13181b;
                color: #e8eef1;
                border: 1px solid #2b3439;
                border-radius: 6px;
                padding: 10px 12px;
            }
            QWidget#wallLineEditor {
                background: #0b0d0f;
                color: #aeb8be;
                border: 1px solid #2d353a;
            }
            QPushButton {
                background: #263238;
                color: #f4f7f8;
                border: 1px solid #3c4a51;
                border-radius: 6px;
                padding: 7px 10px;
            }
            QPushButton:hover {
                background: #314148;
                border-color: #5d747e;
            }
            QPushButton:pressed {
                background: #1b252a;
            }
            QSpinBox, QDoubleSpinBox {
                background: #171b1e;
                color: #f4f7f8;
                border: 1px solid #394349;
                border-radius: 4px;
                padding: 4px 6px;
            }
            """
        )


class WallModelToolWindow(QMainWindow):
    def __init__(self, workspace: Path, *, max_points: int):
        super().__init__()
        self.setWindowTitle("Wall Model Tool")
        self.resize(1180, 820)
        self.workbench = WallModelWorkbench(workspace, max_points=max_points, parent=self)
        self.workbench.status_changed.connect(self._show_status)
        self.setCentralWidget(self.workbench)
        self.setStatusBar(QStatusBar())

    def generate_wall_artifacts(self):
        self.workbench.generate_wall_artifacts()

    def _show_status(self, message: str):
        self.statusBar().showMessage(message)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    app = QApplication([sys.argv[0]])
    win = WallModelToolWindow(args.workspace, max_points=args.max_points)
    win.show()
    if args.auto_generate:
        QTimer.singleShot(0, win.generate_wall_artifacts)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
