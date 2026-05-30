"""主窗口：左侧 OpenGL 视图，右侧 keyframe 列表 + 控制面板 + 信息。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QListWidget, QMainWindow,
    QMessageBox, QPushButton, QSlider, QSplitter, QStatusBar, QVBoxLayout, QWidget,
    QComboBox, QStackedWidget,
)

from core.dataset import CameraPose, Dataset, find_default_dataset, load_dataset
from core.detection_cache import find_detection_json
from core.detection_runner import DETECTION_MODE_LABELS, run_detection_for_image
from core.door_window_refine import refine_detection_selection
from core.projection import project_points_to_panorama, rotation_from_angle
from render.scene_view import SceneView
from ui.pano_annotation_editor import PanoAnnotationEditor


class DetectionWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, workspace: Path, image_path: Path, mode: str):
        super().__init__()
        self.workspace = workspace
        self.image_path = image_path
        self.mode = mode

    def run(self):
        try:
            out = run_detection_for_image(self.workspace, self.image_path, mode=self.mode)
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.finished.emit(str(out))


class MainWindow(QMainWindow):
    def __init__(self, workspace: Path):
        super().__init__()
        self.setWindowTitle("3D Viewer — Point Cloud + Panorama")
        self.resize(1500, 940)
        self.workspace = workspace
        self.dataset: Dataset | None = None
        self.current_idx = -1
        self.current_detections: list[dict] = []
        self.current_image_size: tuple[int, int] | None = None
        self._det_thread: QThread | None = None
        self._det_worker: DetectionWorker | None = None

        self.scene = SceneView()
        self.scene.hover_changed.connect(self._on_hover)
        self.scene.point_clicked.connect(self._on_point_clicked)
        self.editor = PanoAnnotationEditor(self.workspace)
        self.editor.saved.connect(self._on_annotation_saved)
        self.editor.canceled.connect(self._exit_annotation_editor)
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.scene)
        self.view_stack.addWidget(self.editor)

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

        self.cb_pick_dw = QCheckBox("门窗选择")
        self.cb_pick_dw.setChecked(False)
        self.cb_pick_dw.toggled.connect(self._set_door_window_pick_mode)

        self.cb_show_bboxes = QCheckBox("显示检测框")
        self.cb_show_bboxes.setChecked(True)
        self.cb_show_bboxes.toggled.connect(self.scene.set_show_bboxes)

        self.btn_detect_current = QPushButton("检测当前帧")
        self.btn_detect_current.clicked.connect(self._detect_current_frame)

        self.btn_edit_current = QPushButton("编辑当前全景框")
        self.btn_edit_current.clicked.connect(self._edit_current_frame)

        self.det_mode = QComboBox()
        self.det_mode.addItem("精准模式", "precise")
        self.det_mode.addItem("召回模式", "recall")
        self.det_mode.currentIndexChanged.connect(self._on_detection_mode_changed)

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
        self.yaw_offset.setCurrentIndex(2)
        self.yaw_offset.currentIndexChanged.connect(self._on_yaw_offset)

        self.lbl_pose = QLabel("—")
        self.lbl_pose.setWordWrap(True)
        self.lbl_hover = QLabel("hover: —")
        self.lbl_hover.setWordWrap(True)
        self.lbl_hover.setStyleSheet("font-family: ui-monospace, Menlo, monospace;")
        self.lbl_detection = QLabel("检测: —")
        self.lbl_detection.setWordWrap(True)
        self.lbl_detection.setStyleSheet("font-family: ui-monospace, Menlo, monospace;")

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
        v.addWidget(self.cb_pick_dw)
        v.addWidget(self.cb_show_bboxes)
        v.addWidget(QLabel("检测模式"))
        v.addWidget(self.det_mode)
        v.addWidget(self.btn_detect_current)
        v.addWidget(self.btn_edit_current)

        v.addWidget(QLabel("点大小"))
        v.addWidget(self.size_slider)

        v.addWidget(QLabel("全景水平校准"))
        v.addWidget(self.yaw_offset)

        v.addWidget(QLabel("当前位姿")); v.addWidget(self.lbl_pose)
        v.addWidget(QLabel("吸附信息")); v.addWidget(self.lbl_hover)
        v.addWidget(QLabel("门窗命中")); v.addWidget(self.lbl_detection)

        v.addStretch(1)
        v.addWidget(QLabel("拖动: 转视角 · 滚轮: 缩放 · ←/→: 切 keyframe"
                          " · 1/2: 显隐全景/点云 · R: 重置"))

        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(self.view_stack); sp.addWidget(side)
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
        self.scene.set_pano_yaw_offset(float(self.yaw_offset.currentData()))
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
        self.scene.set_highlight_mask(None)
        self.scene.set_selected_detection(-1)
        self._load_detections(pose)
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

    def _set_door_window_pick_mode(self, on: bool):
        self.scene.set_pick_mode(on)
        self.statusBar().showMessage("门窗选择模式：点击点云点" if on else "导航模式")

    def _load_detections(self, pose: CameraPose):
        self.current_detections = []
        self.current_image_size = None
        self.scene.set_detections([], 0, 0)
        mode = self._detection_mode()
        path = find_detection_json(self.workspace, pose.image_name, mode=mode)
        if path is None:
            self.lbl_detection.setText(f"检测: 未找到当前 keyframe JSON\n{DETECTION_MODE_LABELS[mode]}")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.lbl_detection.setText(f"检测: 读取失败\n{path.name}\n{e}")
            return
        self.current_detections = list(payload.get("detections", []))
        self.current_image_size = (int(payload.get("width", 0)), int(payload.get("height", 0)))
        self.scene.set_detections(
            self.current_detections,
            self.current_image_size[0],
            self.current_image_size[1],
        )
        self.lbl_detection.setText(
            f"检测: {len(self.current_detections)} boxes · {DETECTION_MODE_LABELS[mode]}\n"
            f"{path.relative_to(self.workspace)}"
        )

    def _detection_mode(self) -> str:
        return str(self.det_mode.currentData())

    def _on_detection_mode_changed(self):
        if not self.dataset or self.current_idx < 0:
            return
        self.scene.set_highlight_mask(None)
        self.scene.set_selected_detection(-1)
        self._load_detections(self.dataset.poses[self.current_idx])

    def _detect_current_frame(self):
        if not self.dataset or self.current_idx < 0:
            return
        if self._det_thread is not None:
            self.lbl_detection.setText("检测: 当前已有检测任务运行中")
            return
        pose = self.dataset.poses[self.current_idx]
        mode = self._detection_mode()
        cached = find_detection_json(self.workspace, pose.image_name, mode=mode)
        if cached is not None:
            self._load_detections(pose)
            self.statusBar().showMessage(f"已加载缓存: {cached.name}")
            return
        image_path = self.dataset.image_dir / pose.image_name
        self.btn_detect_current.setEnabled(False)
        self.lbl_detection.setText(f"检测中... {DETECTION_MODE_LABELS[mode]}\n{pose.image_name}")
        self.statusBar().showMessage(f"正在检测当前帧（{DETECTION_MODE_LABELS[mode]}），完成后会自动加载结果")

        thread = QThread(self)
        worker = DetectionWorker(self.workspace, image_path, mode)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_detection_finished)
        worker.failed.connect(self._on_detection_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear_detection_worker)
        self._det_thread = thread
        self._det_worker = worker
        thread.start()

    def _edit_current_frame(self):
        if not self.dataset or self.current_idx < 0:
            return
        pose = self.dataset.poses[self.current_idx]
        image_path = self.dataset.image_dir / pose.image_name
        if not image_path.exists():
            QMessageBox.warning(self, "缺图", str(image_path))
            return
        self.editor.load_image(image_path, self.current_detections)
        self.view_stack.setCurrentWidget(self.editor)
        self.statusBar().showMessage(
            "正在编辑当前全景框：拖拽画框，点击框选中，Delete 删除，编辑结束保存"
        )

    def _on_annotation_saved(self, path_text: str):
        self._exit_annotation_editor()
        if self.dataset and self.current_idx >= 0:
            pose = self.dataset.poses[self.current_idx]
            self._load_detections(pose)
        self.statusBar().showMessage(f"手工框已保存: {Path(path_text).name}")

    def _exit_annotation_editor(self):
        self.view_stack.setCurrentWidget(self.scene)

    def _on_detection_finished(self, path_text: str):
        if not self.dataset or self.current_idx < 0:
            return
        pose = self.dataset.poses[self.current_idx]
        out_path = Path(path_text)
        if out_path.stem == Path(pose.image_name).stem:
            self._load_detections(pose)
            self.statusBar().showMessage(f"检测完成: {out_path.name}")
        else:
            self.statusBar().showMessage(f"检测完成并已缓存: {out_path.name}")

    def _on_detection_failed(self, message: str):
        self.lbl_detection.setText(f"检测失败\n{message}")
        self.statusBar().showMessage("检测失败")

    def _clear_detection_worker(self):
        self.btn_detect_current.setEnabled(True)
        self._det_thread = None
        self._det_worker = None

    def _on_point_clicked(self, idx: int):
        if not self.dataset or self.current_idx < 0:
            return
        if idx < 0:
            self.lbl_detection.setText("点击: 未吸附到点云点")
            self.scene.set_highlight_mask(None)
            self.scene.set_selected_detection(-1)
            return
        if not self.current_detections:
            self.lbl_detection.setText("点击: 当前 keyframe 没有检测 JSON")
            self.scene.set_highlight_mask(None)
            self.scene.set_selected_detection(-1)
            return

        pose = self.dataset.poses[self.current_idx]
        img_w, img_h = self.current_image_size or (0, 0)
        if img_w <= 0 or img_h <= 0:
            image_path = self.dataset.image_dir / pose.image_name
            from PIL import Image
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        R = rotation_from_angle(pose.roll, pose.pitch, pose.yaw)
        yaw_offset = float(self.yaw_offset.currentData())
        uv = project_points_to_panorama(
            self.dataset.points,
            pose.position,
            R,
            img_w,
            img_h,
            yaw_offset_deg=yaw_offset,
        )
        selection = refine_detection_selection(
            points=self.dataset.points,
            uv=uv,
            clicked_idx=idx,
            detections=self.current_detections,
            pano_w=float(img_w),
            cam_pos=pose.position,
        )
        if selection.detection_index < 0:
            self.scene.set_highlight_mask(None)
            self.scene.set_selected_detection(-1)
            self.lbl_detection.setText(
                f"点击点 #{idx}\n未落入 door/window bbox\n"
                f"reason: {selection.reason}\n"
                f"uv: ({uv[idx,0]:.1f}, {uv[idx,1]:.1f})"
            )
            return
        highlight = selection.refined_mask if selection.point_count > 0 else selection.coarse_mask
        self.scene.set_highlight_mask(highlight)
        self.scene.set_selected_detection(selection.detection_index)
        score = f"{selection.score:.3f}" if selection.score is not None else "—"
        self.lbl_detection.setText(
            f"命中 #{selection.detection_index} {selection.label} score={score}\n"
            f"confidence: {selection.confidence}\n"
            f"coarse: {selection.coarse_count:,} refined: {selection.point_count:,}\n"
            f"reason: {selection.reason}\n"
            f"uv: ({uv[idx,0]:.1f}, {uv[idx,1]:.1f})"
        )

    # 全屏快捷键转给 scene
    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Left,):
            self._step(-1)
        elif e.key() in (Qt.Key_Right,):
            self._step(1)
        else:
            super().keyPressEvent(e)
