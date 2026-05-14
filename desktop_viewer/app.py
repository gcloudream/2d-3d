from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from data import (
    DatasetConfig,
    CameraPose,
    default_dataset_config,
    load_las_sample,
    parse_camera_file,
    unique_camera_poses,
    validate_images,
)
from unified_scene import UnifiedSceneView


WORKSPACE = Path(__file__).resolve().parents[1]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Point Cloud + Panorama Unified Viewer")
        self.resize(1520, 940)

        self.config = default_dataset_config(WORKSPACE)
        self.poses: list[CameraPose] = []
        self.points = np.empty((0, 3), dtype=np.float32)
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self.current_index = 0

        self.scene_view = UnifiedSceneView()
        self.camera_list = QListWidget()
        self.camera_list.currentRowChanged.connect(self.select_camera)

        self.panorama_checkbox = QCheckBox("显示全景影像")
        self.panorama_checkbox.setChecked(True)
        self.panorama_checkbox.stateChanged.connect(
            lambda _: self.scene_view.set_show_panorama(self.panorama_checkbox.isChecked())
        )

        self.points_checkbox = QCheckBox("显示点云")
        self.points_checkbox.setChecked(True)
        self.points_checkbox.stateChanged.connect(
            lambda _: self.scene_view.set_show_points(self.points_checkbox.isChecked())
        )

        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(8)
        self.point_size_slider.setValue(2)
        self.point_size_slider.valueChanged.connect(self.scene_view.set_point_size)

        self.info_label = QLabel("未加载")
        self.info_label.setWordWrap(True)

        self._build_layout()
        self._build_toolbar()
        self.setStatusBar(QStatusBar())
        self.load_dataset()

    def _build_layout(self) -> None:
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(8)

        side_layout.addWidget(QLabel("全景图片"))
        side_layout.addWidget(self.camera_list, 1)

        nav_row = QHBoxLayout()
        prev_btn = QPushButton("上一张")
        next_btn = QPushButton("下一张")
        reset_btn = QPushButton("重置视角")
        prev_btn.clicked.connect(lambda: self.step_camera(-1))
        next_btn.clicked.connect(lambda: self.step_camera(1))
        reset_btn.clicked.connect(self.scene_view.reset_view)
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        side_layout.addLayout(nav_row)
        side_layout.addWidget(reset_btn)

        side_layout.addWidget(self.panorama_checkbox)
        side_layout.addWidget(self.points_checkbox)

        side_layout.addWidget(QLabel("点云点大小"))
        side_layout.addWidget(self.point_size_slider)
        side_layout.addWidget(QLabel("当前位姿"))
        side_layout.addWidget(self.info_label)

        hint = QLabel("鼠标左键拖动旋转空间，滚轮调整视场。点云和全景影像共用当前视角。")
        hint.setWordWrap(True)
        side_layout.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.scene_view)
        splitter.addWidget(side_panel)
        splitter.setSizes([1240, 280])
        self.setCentralWidget(splitter)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Dataset")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_root = QAction("选择数据目录", self)
        open_root.triggered.connect(self.choose_dataset_root)
        toolbar.addAction(open_root)

        reload_action = QAction("重新加载", self)
        reload_action.triggered.connect(self.load_dataset)
        toolbar.addAction(reload_action)

    def choose_dataset_root(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择数据目录", str(self.config.data_root))
        if not directory:
            return
        root = Path(directory)
        self.config = DatasetConfig(
            data_root=root,
            camera_file=root / "CAM" / "camera_pos.cam",
            image_dir=root / "CAM",
            pointcloud_file=root / "LAS_Rgb" / f"{root.name}_rgb_0.las",
        )
        self.load_dataset()

    def load_dataset(self) -> None:
        try:
            if not self.config.camera_file.exists():
                raise FileNotFoundError(self.config.camera_file)
            if not self.config.image_dir.exists():
                raise FileNotFoundError(self.config.image_dir)
            if not self.config.pointcloud_file.exists():
                fallback = self.config.data_root / "LAS" / f"{self.config.data_root.name}.las"
                if fallback.exists():
                    self.config.pointcloud_file = fallback
                else:
                    raise FileNotFoundError(self.config.pointcloud_file)

            raw_poses = parse_camera_file(self.config.camera_file)
            self.poses = unique_camera_poses(raw_poses)
            missing = validate_images(self.poses, self.config.image_dir)
            if missing:
                raise FileNotFoundError(f"{len(missing)} images are missing; first: {missing[0]}")

            self.statusBar().showMessage("正在加载抽样点云...")
            sample = load_las_sample(self.config.pointcloud_file, max_points=260_000)
            self.points = sample.points
            self.colors = sample.colors
            self.scene_view.set_world_points(self.points, self.colors)

            self.camera_list.blockSignals(True)
            self.camera_list.clear()
            for pose in self.poses:
                suffix = f"  {pose.timestamp:.3f}" if pose.timestamp is not None else ""
                self.camera_list.addItem(f"{pose.image_name}{suffix}")
            self.camera_list.blockSignals(False)

            self.statusBar().showMessage(
                f"已加载 {len(self.poses)} 张全景图；点云抽样 {len(self.points):,}/{sample.total_points:,}，步长 {sample.sample_step}"
            )
            self.camera_list.setCurrentRow(0)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            self.statusBar().showMessage("加载失败")

    def step_camera(self, delta: int) -> None:
        if not self.poses:
            return
        next_index = max(0, min(len(self.poses) - 1, self.current_index + delta))
        self.camera_list.setCurrentRow(next_index)

    def select_camera(self, index: int) -> None:
        if index < 0 or index >= len(self.poses):
            return
        self.current_index = index
        pose = self.poses[index]
        self.scene_view.set_pose(pose, self.config.image_dir / pose.image_name)
        self.info_label.setText(
            f"{pose.image_name}\n"
            f"位置: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})\n"
            f"姿态: roll={np.degrees(pose.roll):.1f}, "
            f"pitch={np.degrees(pose.pitch):.1f}, yaw={np.degrees(pose.yaw):.1f}"
        )


def main() -> int:
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(0)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
