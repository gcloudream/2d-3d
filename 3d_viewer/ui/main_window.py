"""主窗口：左侧 OpenGL 视图，右侧 keyframe 列表 + 控制面板 + 信息。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QListWidget, QMainWindow,
    QMessageBox, QPushButton, QSlider, QSplitter, QStatusBar, QVBoxLayout, QWidget,
    QComboBox,
)

from core.dataset import CameraPose, Dataset, find_default_dataset, load_dataset
from render.scene_view import SceneView


class MainWindow(QMainWindow):
    def __init__(self, workspace: Path):
        super().__init__()
        self.setWindowTitle("3D Viewer — Point Cloud + Panorama")
        self.resize(1500, 940)
        self.workspace = workspace
        self.dataset: Dataset | None = None
        self.current_idx = -1

        self.scene = SceneView()
        self.scene.hover_changed.connect(self._on_hover)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_select)

        self.cb_pano = QCheckBox("显示全景")
        self.cb_pano.setChecked(True)
        self.cb_pano.toggled.connect(self._set_pano_visible)

        self.btn_toggle_pano = QPushButton("隐藏全景影像")
        self.btn_toggle_pano.clicked.connect(self._toggle_pano)

        self.cb_pc = QCheckBox("显示点云")
        self.cb_pc.setChecked(True)
        self.cb_pc.toggled.connect(self.scene.set_show_pc)

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(8)
        self.size_slider.setValue(2)
        self.size_slider.valueChanged.connect(lambda v: self.scene.set_point_size(float(v)))

        self.yaw_offset = QComboBox()
        for label, value in [
            ("0°", 0.0),
            ("+90°", 90.0),
            ("-90°", -90.0),
            ("180°", 180.0),
        ]:
            self.yaw_offset.addItem(label, value)
        self.yaw_offset.currentIndexChanged.connect(self._on_yaw_offset)

        self.lbl_pose = QLabel("—")
        self.lbl_pose.setWordWrap(True)
        self.lbl_hover = QLabel("hover: —")
        self.lbl_hover.setWordWrap(True)
        self.lbl_hover.setStyleSheet("font-family: ui-monospace, Menlo, monospace;")

        self._build_ui()
        self.setStatusBar(QStatusBar())
        self._load()

    def _build_ui(self):
        side = QWidget(); v = QVBoxLayout(side)
        v.setContentsMargins(8, 8, 8, 8); v.setSpacing(8)

        v.addWidget(QLabel("Keyframes"))
        v.addWidget(self.list, 1)

        nav = QHBoxLayout()
        for txt, slot in [("◀", lambda: self._step(-1)),
                          ("▶", lambda: self._step(1)),
                          ("⟳ 视角", self.scene.reset_view)]:
            b = QPushButton(txt); b.clicked.connect(slot); nav.addWidget(b)
        v.addLayout(nav)

        v.addWidget(self.cb_pano)
        v.addWidget(self.btn_toggle_pano)
        v.addWidget(self.cb_pc)

        v.addWidget(QLabel("点大小"))
        v.addWidget(self.size_slider)

        v.addWidget(QLabel("全景水平校准"))
        v.addWidget(self.yaw_offset)

        v.addWidget(QLabel("当前位姿")); v.addWidget(self.lbl_pose)
        v.addWidget(QLabel("吸附信息")); v.addWidget(self.lbl_hover)

        v.addStretch(1)
        v.addWidget(QLabel("拖动: 转视角 · 滚轮: 缩放 · ←/→: 切 keyframe"
                          " · 1/2: 显隐全景/点云 · R: 重置"))

        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(self.scene); sp.addWidget(side)
        sp.setSizes([1180, 320])
        self.setCentralWidget(sp)

    def _load(self):
        self.statusBar().showMessage("loading dataset…")
        try:
            cfg = find_default_dataset(self.workspace)
            if cfg is None:
                raise RuntimeError(f"no dataset found under {self.workspace}")
            self.dataset = load_dataset(cfg, max_points=300_000)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
            self.statusBar().showMessage("加载失败")
            return

        d = self.dataset
        self.scene.set_world_points(d.points, d.colors)
        self.list.blockSignals(True)
        self.list.clear()
        for p in d.poses:
            ts = f"  t={p.timestamp:.2f}" if p.timestamp is not None else ""
            self.list.addItem(f"{p.image_name}{ts}")
        self.list.blockSignals(False)

        self.statusBar().showMessage(
            f"poses={len(d.poses)} · points={d.points.shape[0]:,}/{d.total_points:,} (step {d.sample_step})"
        )
        if d.poses:
            self.list.setCurrentRow(0)

    def _step(self, delta: int):
        if not self.dataset or not self.dataset.poses:
            return
        n = len(self.dataset.poses)
        nxt = max(0, min(n - 1, self.current_idx + delta))
        self.list.setCurrentRow(nxt)

    def _on_select(self, row: int):
        if not self.dataset or row < 0 or row >= len(self.dataset.poses):
            return
        self.current_idx = row
        pose = self.dataset.poses[row]
        img = self.dataset.image_dir / pose.image_name
        if not img.exists():
            QMessageBox.warning(self, "缺图", str(img))
            return
        self.scene.set_keyframe(pose, img)
        self.lbl_pose.setText(
            f"{pose.image_name}\n"
            f"pos: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})\n"
            f"rpy: ({np.degrees(pose.roll):.1f}, {np.degrees(pose.pitch):.1f}, "
            f"{np.degrees(pose.yaw):.1f})°"
        )

    def _on_hover(self, info):
        if info is None:
            self.lbl_hover.setText("hover: —")
            return
        self.lbl_hover.setText(
            f"#{info['index']}\n"
            f"xyz: {info['x']:+.3f}, {info['y']:+.3f}, {info['z']:+.3f}\n"
            f"rgb: ({info['r']}, {info['g']}, {info['b']})"
        )

    def _toggle_pano(self):
        self.cb_pano.setChecked(not self.cb_pano.isChecked())

    def _set_pano_visible(self, on: bool):
        self.scene.set_show_pano(on)
        self.btn_toggle_pano.setText("隐藏全景影像" if on else "显示全景影像")

    def _on_yaw_offset(self):
        self.scene.set_pano_yaw_offset(float(self.yaw_offset.currentData()))

    # 全屏快捷键转给 scene
    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Left,):
            self._step(-1)
        elif e.key() in (Qt.Key_Right,):
            self._step(1)
        else:
            super().keyPressEvent(e)
