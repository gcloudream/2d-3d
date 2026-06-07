"""Standalone UI for generating and previewing wall model artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import moderngl
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QSurfaceFormat
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

from core.dataset import Dataset, find_default_dataset, load_dataset
from core.wall_model import WallModelResult, generate_wall_model
from core.wall_openings import load_wall_openings
from render.orbit_camera import OrbitCamera


WORKSPACE = HERE.parent

_VERT_SRC = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

_FRAG_SRC = """
#version 330
uniform vec4 color;
out vec4 frag;
void main() {
    frag = color;
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


def load_obj_triangles(path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[float, float, float]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            continue
        if parts[0] != "f" or len(parts) < 4:
            continue
        indices = [_parse_face_index(token, len(vertices)) for token in parts[1:]]
        for i in range(1, len(indices) - 1):
            for index in (indices[0], indices[i], indices[i + 1]):
                triangles.append(vertices[index])
    return np.asarray(triangles, dtype=np.float32).reshape((-1, 3))


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
        self._pending_triangles: np.ndarray | None = None
        self.show_mesh_edges = True
        self._dragging = False
        self._last_pos = QPoint()

    def load_obj(self, path: Path):
        triangles = load_obj_triangles(path)
        if self._ctx is None:
            self._pending_triangles = triangles
            return
        self._upload(triangles)
        self.update()

    def initializeGL(self):
        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._prog = self._ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)
        if self._pending_triangles is not None:
            self._upload(self._pending_triangles)
            self._pending_triangles = None

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
        self._prog["color"].value = (0.68, 0.78, 0.82, 1.0)
        self._vao.render(mode=moderngl.TRIANGLES)
        if self.show_mesh_edges and self._edge_vao is not None:
            self._prog["color"].value = (0.05, 0.08, 0.09, 1.0)
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

    def _upload(self, triangles: np.ndarray):
        self._release_buffers()
        data = np.ascontiguousarray(np.asarray(triangles, dtype=np.float32).reshape((-1, 3)))
        self._triangles = data
        if len(data) == 0 or self._ctx is None or self._prog is None:
            return
        self.camera.fit_to_points(data)
        self._vbo = self._ctx.buffer(data.tobytes())
        self._vao = self._ctx.simple_vertex_array(self._prog, self._vbo, "in_pos")
        edges = _triangle_edges(data)
        self._edge_vbo = self._ctx.buffer(edges.tobytes())
        self._edge_vao = self._ctx.simple_vertex_array(self._prog, self._edge_vbo, "in_pos")

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


class WallModelWorkbench(QWidget):
    status_changed = Signal(str)
    back_requested = Signal()

    def __init__(self, workspace: Path, *, max_points: int, parent=None):
        super().__init__(parent)
        self.setObjectName("wallModelWorkbench")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.workspace = Path(workspace)
        self.dataset: Dataset | None = None
        self.result: WallModelResult | None = None
        self._topdown_path: Path | None = None

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
        self.generate_button = QPushButton("生成墙体线图和 OBJ")
        self.generate_button.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.generate_button.clicked.connect(self.generate_wall_artifacts)
        self.back_button = QPushButton("返回主视图")
        self.back_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.back_button.clicked.connect(self.back_requested)
        self.back_button.hide()

        self.info = QLabel("尚未加载点云")
        self.info.setObjectName("wallModelInfo")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.topdown_image = QLabel("墙体线图")
        self.topdown_image.setAlignment(Qt.AlignCenter)
        self.topdown_image.setMinimumSize(520, 420)
        self.topdown_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
        controls.addWidget(self.generate_button)
        controls.addWidget(self.back_button)
        layout.addWidget(toolbar)
        layout.addWidget(self.info)

        previews = QHBoxLayout()
        previews.setSpacing(10)
        previews.addWidget(self.topdown_image, 1)
        previews.addWidget(self.model_view, 1)
        layout.addLayout(previews, 1)

    def load_default_dataset(self):
        self._show_status("正在加载默认点云...")
        QApplication.processEvents()
        try:
            cfg = find_default_dataset(self.workspace)
            if cfg is None:
                raise RuntimeError(f"no dataset found under {self.workspace}")
            self.dataset = load_dataset(cfg, max_points=int(self.max_points.value()))
        except Exception as exc:
            self.dataset = None
            QMessageBox.warning(self, "点云加载失败", str(exc))
            self._show_status("点云加载失败")
            return
        self.info.setText(
            f"已加载: {self.dataset.data_root.name}\n"
            f"点云: {self.dataset.points.shape[0]:,}/{self.dataset.total_points:,} "
            f"(step {self.dataset.sample_step})\n"
            f"LAS: {self.dataset.pointcloud_file}"
        )
        self._show_status("默认点云已加载")

    def generate_wall_artifacts(self):
        if self.dataset is None:
            self.load_default_dataset()
            if self.dataset is None:
                return
        self.generate_button.setEnabled(False)
        self._show_status("正在生成墙体线图和 OBJ...")
        QApplication.processEvents()
        try:
            openings = load_wall_openings(self.workspace, self.dataset.data_root)
            self.result = generate_wall_model(
                self.workspace,
                self.dataset.data_root,
                self.dataset.points,
                openings=openings,
                resolution_m=float(self.resolution.value()),
            )
        except Exception as exc:
            QMessageBox.warning(self, "墙体模型生成失败", str(exc))
            self._show_status("墙体模型生成失败")
            return
        finally:
            self.generate_button.setEnabled(True)
        self._show_result(self.result)

    def _show_result(self, result: WallModelResult):
        self._topdown_path = result.topdown_preview_path
        self._set_image(self.topdown_image, result.topdown_preview_path)
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

    def _set_image(self, label: QLabel, path: Path):
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            label.setText(str(path))
            return
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._topdown_path is not None:
            self._set_image(self.topdown_image, self._topdown_path)

    def _show_status(self, message: str):
        self.status_changed.emit(message)

    def set_back_button_visible(self, visible: bool):
        self.back_button.setVisible(bool(visible))

    def _apply_style(self):
        self.topdown_image.setObjectName("topdownImage")
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
            #wallModelToolbar QLabel {
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
            QLabel#topdownImage {
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
