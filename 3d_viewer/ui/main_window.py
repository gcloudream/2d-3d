"""主窗口：左侧 OpenGL 视图，右侧 keyframe 列表 + 控制面板 + 信息。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QListWidget, QMainWindow,
    QMessageBox, QPushButton, QSlider, QSplitter, QStatusBar, QVBoxLayout, QWidget,
    QComboBox, QStackedWidget, QStyle,
)

from core.dataset import CameraPose, Dataset, find_default_dataset, load_dataset
from core.app_logging import configure_operation_logging, log_operation
from core.detection_cache import find_detection_json
from core.detection_schema import normalize_detections
from core.door_window_fusion import (
    FrustumProjectionCache,
    extract_detection_region_from_bbox,
    fuse_detection_and_pointcloud,
    should_highlight_fused,
)
from core.pointcloud_extract import (
    extract_planar_region_from_seed,
    should_highlight_planar_region,
)
from core.projection import rotation_from_angle
from core.wall_openings import (
    append_wall_opening,
    append_wall_opening_event,
    clear_wall_opening_session_files,
    is_recordable_opening_metadata,
    opening_from_selection,
)
from render.scene_view import SceneView
from ui.pano_annotation_editor import PanoAnnotationEditor
from ui.scene_sync import (
    configure_observer_scene,
    set_scene_pair_highlight_mask,
    set_scene_pair_highlight_style,
    set_scene_pair_point_size,
    set_scene_pair_selected_depth_test,
)
from wall_model_tool import WallModelWorkbench


DEFAULT_PANORAMA_YAW_OFFSET_DEG = -90.0
HIGHLIGHT_STYLES = {
    "off": ((1.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    "bbox": ((0.0, 0.85, 1.0), (0.0, 0.2, 1.0)),
    "depth": ((1.0, 0.55, 0.0), (0.85, 0.0, 1.0)),
    "final": ((0.35, 1.0, 0.35), (0.0, 0.65, 0.2)),
}


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
        # Accumulated door/window highlight, so 补充 (supplement) clicks can add
        # disconnected sub-regions a single seed could not reach.
        self._highlight_mask: np.ndarray | None = None
        self._last_opening_candidate: dict | None = None
        self._fusion_cache = FrustumProjectionCache()
        self._debug_masks: dict[str, np.ndarray] = {}
        self._debug_layer_mode = "off"
        self._last_detection_base_text = "检测: —"
        configure_operation_logging(self.workspace)
        self._log_operation("viewer_initialized", window_title="3D Viewer — Point Cloud + Panorama")

        self.scene = SceneView()
        self.scene.set_selected_depth_test(True)
        self.scene.hover_changed.connect(self._on_hover)
        self.scene.point_clicked.connect(self._on_point_clicked)
        self.scene.detection_clicked.connect(self._on_detection_clicked)
        self.cloud_scene = SceneView()
        self.cloud_scene.set_selected_depth_test(True)
        self.cloud_scene.point_clicked.connect(self._on_cloud_point_clicked)
        self.editor = PanoAnnotationEditor(self.workspace)
        self.editor.saved.connect(self._on_annotation_saved)
        self.editor.canceled.connect(self._exit_annotation_editor)
        self.normal_view = QSplitter(Qt.Vertical)
        self.normal_view.addWidget(self.scene)
        self.normal_view.addWidget(self.cloud_scene)
        self.normal_view.setSizes([680, 260])
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.normal_view)
        self.view_stack.addWidget(self.editor)
        self.wall_model_view = WallModelWorkbench(self.workspace, max_points=300_000)
        self.wall_model_view.set_back_button_visible(True)
        self.wall_model_view.status_changed.connect(self._on_wall_model_status)
        self.wall_model_view.back_requested.connect(self._show_main_view)
        self.view_stack.addWidget(self.wall_model_view)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_select)

        self.btn_main_view = QPushButton("主视图")
        self.btn_main_view.setCheckable(True)
        self.btn_main_view.setChecked(True)
        self.btn_main_view.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.btn_main_view.clicked.connect(self._show_main_view)

        self.btn_wall_model = QPushButton("墙体建模")
        self.btn_wall_model.setCheckable(True)
        self.btn_wall_model.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.btn_wall_model.setToolTip("打开墙体线图和 OBJ 生成工作台")
        self.btn_wall_model.clicked.connect(self._show_wall_model)

        self.cb_pano = QCheckBox("显示全景")
        self.cb_pano.setChecked(True)
        self.cb_pano.toggled.connect(self._set_pano_visible)

        self.cb_pc = QCheckBox("显示点云")
        self.cb_pc.setChecked(True)
        self.cb_pc.toggled.connect(self.scene.set_show_pc)

        self.interaction_mode = QComboBox()
        self.interaction_mode.addItem("导航模式", "navigate")
        self.interaction_mode.addItem("点云 seed 提取", "point")
        self.interaction_mode.addItem("检测框提取", "detection")
        self.interaction_mode.currentIndexChanged.connect(self._on_interaction_mode_changed)

        self.cb_supplement = QCheckBox("补充模式 (累加高亮)")
        self.cb_supplement.setChecked(False)
        self.cb_supplement.setToolTip(
            "开启后，点击新点会把提取结果并入已有高亮，用于补全单次点击漏掉的门窗子区域"
        )

        self.debug_layer = QComboBox()
        self.debug_layer.addItem("关闭", "off")
        self.debug_layer.addItem("bbox 候选点", "bbox")
        self.debug_layer.addItem("深度候选点", "depth")
        self.debug_layer.addItem("最终结果点", "final")
        self.debug_layer.currentIndexChanged.connect(self._on_debug_layer_changed)

        self.cb_xray_highlight = QCheckBox("高亮透视 (忽略遮挡)")
        self.cb_xray_highlight.setChecked(False)
        self.cb_xray_highlight.setToolTip(
            "开启后，高亮点会穿透显示，用于判断上方全景视角是否被前景点云遮挡"
        )
        self.cb_xray_highlight.toggled.connect(self._set_highlight_xray)

        self.btn_clear_highlight = QPushButton("清除高亮")
        self.btn_clear_highlight.clicked.connect(self._clear_highlight)
        self.btn_record_opening = QPushButton("记录当前门窗")
        self.btn_record_opening.clicked.connect(self._record_current_opening)

        self.cb_show_bboxes = QCheckBox("显示检测框")
        self.cb_show_bboxes.setChecked(True)
        self.cb_show_bboxes.toggled.connect(self.scene.set_show_bboxes)

        self.btn_edit_current = QPushButton("编辑当前全景框")
        self.btn_edit_current.clicked.connect(self._edit_current_frame)

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(8)
        self.size_slider.setValue(2)
        self.size_slider.valueChanged.connect(self._set_point_size)

        self.yaw_offset = QComboBox()
        for label, value in [
            ("0°", 0.0),
            ("+90°", 90.0),
            ("-90°", DEFAULT_PANORAMA_YAW_OFFSET_DEG),
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
        self.side_panel = QWidget()
        v = QVBoxLayout(self.side_panel)
        v.setContentsMargins(8, 8, 8, 8); v.setSpacing(8)

        workspace_nav = QHBoxLayout()
        workspace_nav.setSpacing(6)
        workspace_nav.addWidget(self.btn_main_view)
        workspace_nav.addWidget(self.btn_wall_model)
        v.addLayout(workspace_nav)

        v.addWidget(QLabel("Keyframes"))
        v.addWidget(self.list, 1)

        nav = QHBoxLayout()
        for txt, slot in [("◀", lambda: self._step(-1)),
                          ("▶", lambda: self._step(1)),
                          ("⟳ 视角", self.scene.reset_view)]:
            b = QPushButton(txt); b.clicked.connect(slot); nav.addWidget(b)
        v.addLayout(nav)

        v.addWidget(self.cb_pano)
        v.addWidget(self.cb_pc)
        v.addWidget(QLabel("交互模式"))
        v.addWidget(self.interaction_mode)
        v.addWidget(self.cb_supplement)
        v.addWidget(QLabel("调试高亮层"))
        v.addWidget(self.debug_layer)
        v.addWidget(self.cb_xray_highlight)
        v.addWidget(self.btn_clear_highlight)
        v.addWidget(self.btn_record_opening)
        v.addWidget(self.cb_show_bboxes)
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
        sp.addWidget(self.view_stack); sp.addWidget(self.side_panel)
        sp.setSizes([1180, 320])
        self.setCentralWidget(sp)

    def _load(self):
        self.statusBar().showMessage("loading dataset…")
        self._log_operation("dataset_load_started", workspace=str(self.workspace))
        try:
            cfg = find_default_dataset(self.workspace)
            if cfg is None:
                raise RuntimeError(f"no dataset found under {self.workspace}")
            self.dataset = load_dataset(cfg, max_points=300_000)
        except Exception as e:
            self._log_operation(
                "dataset_load_failed",
                level="error",
                error=e,
                workspace=str(self.workspace),
            )
            QMessageBox.critical(self, "加载失败", str(e))
            self.statusBar().showMessage("加载失败")
            return

        d = self.dataset
        removed_session_files = clear_wall_opening_session_files(self.workspace, d.data_root)
        self._ensure_yaw_offset_value(d.pano_yaw_offset_deg)
        self._set_yaw_offset_value(DEFAULT_PANORAMA_YAW_OFFSET_DEG)
        self.scene.set_world_points(d.points, d.colors)
        self.cloud_scene.set_world_points(d.points, d.colors)
        self.scene.set_pano_yaw_offset(float(self.yaw_offset.currentData()))
        self.cloud_scene.set_pano_yaw_offset(float(self.yaw_offset.currentData()))
        self._configure_observer_scene()
        self.list.blockSignals(True)
        self.list.clear()
        for p in d.poses:
            ts = f"  t={p.timestamp:.2f}" if p.timestamp is not None else ""
            self.list.addItem(f"{p.image_name}{ts}")
        self.list.blockSignals(False)

        self.statusBar().showMessage(
            f"poses={len(d.poses)} · points={d.points.shape[0]:,}/{d.total_points:,} (step {d.sample_step})"
        )
        self._log_operation(
            "dataset_loaded",
            data_root=d.data_root,
            pose_count=len(d.poses),
            point_count=int(d.points.shape[0]),
            total_points=int(d.total_points),
            sample_step=int(d.sample_step),
            removed_session_files=removed_session_files,
            yaw_offset_deg=float(self.yaw_offset.currentData()),
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
        self._fusion_cache.clear()
        pose = self.dataset.poses[row]
        self._log_operation(
            "keyframe_selected",
            keyframe_index=int(row),
            image_name=pose.image_name,
            position=[pose.x, pose.y, pose.z],
            rpy=[pose.roll, pose.pitch, pose.yaw],
        )
        img = self.dataset.image_dir / pose.image_name
        if not img.exists():
            self._log_operation(
                "keyframe_image_missing",
                level="warning",
                keyframe_index=int(row),
                image_name=pose.image_name,
                image_path=img,
            )
            QMessageBox.warning(self, "缺图", str(img))
            return
        self.scene.set_keyframe(pose, img)
        self.cloud_scene.set_keyframe(pose, img, load_pano=False)
        self._configure_observer_scene()
        self._set_highlight_mask(None)
        self._last_opening_candidate = None
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

    def _set_pano_visible(self, on: bool):
        self.scene.set_show_pano(on)

    def _on_yaw_offset(self):
        self._fusion_cache.clear()
        self._clear_debug_masks()
        self.scene.set_pano_yaw_offset(float(self.yaw_offset.currentData()))
        self.cloud_scene.set_pano_yaw_offset(float(self.yaw_offset.currentData()))
        self._configure_observer_scene()
        self._log_operation(
            "yaw_offset_changed",
            yaw_offset_deg=float(self.yaw_offset.currentData()),
            keyframe_index=int(self.current_idx),
        )

    def _configure_observer_scene(self):
        configure_observer_scene(self.cloud_scene)
        self._apply_highlight_depth_test()
        self._apply_interaction_mode()

    def _set_point_size(self, value: int):
        set_scene_pair_point_size(self.scene, self.cloud_scene, float(value))

    def _set_highlight_mask(self, mask):
        self._highlight_mask = None if mask is None else np.asarray(mask, dtype=bool).copy()
        self._refresh_display_highlight()

    def _display_highlight_mask(self):
        mode = self._debug_layer_mode
        if mode != "off" and mode in self._debug_masks:
            return self._debug_masks[mode]
        return self._highlight_mask

    def _refresh_display_highlight(self):
        set_scene_pair_highlight_mask(self.scene, self.cloud_scene, self._display_highlight_mask())

    def _refresh_highlight_style(self):
        ring_color, fill_color = HIGHLIGHT_STYLES.get(self._debug_layer_mode, HIGHLIGHT_STYLES["off"])
        set_scene_pair_highlight_style(self.scene, self.cloud_scene, ring_color, fill_color)

    def _apply_highlight_depth_test(self):
        xray_enabled = bool(getattr(self, "cb_xray_highlight", None) and self.cb_xray_highlight.isChecked())
        set_scene_pair_selected_depth_test(self.scene, self.cloud_scene, not xray_enabled)

    def _set_highlight_xray(self, on: bool):
        enabled = bool(on)
        self._apply_highlight_depth_test()
        self._log_operation(
            "highlight_xray_changed",
            xray_enabled=enabled,
            selected_depth_test=not enabled,
            keyframe_index=int(self.current_idx),
            source_image=self._current_source_image(),
        )

    def _clear_debug_masks(self):
        self._debug_masks = {}
        self._refresh_display_highlight()
        self._refresh_detection_text()

    def _set_debug_masks_from_selection(self, selection, final_mask):
        masks: dict[str, np.ndarray] = {}
        raw = getattr(selection, "debug_masks", None)
        if isinstance(raw, dict):
            for key in ("bbox", "depth", "final"):
                value = raw.get(key)
                if value is None:
                    continue
                masks[key] = np.asarray(value, dtype=bool).copy()
        if final_mask is not None and "final" not in masks:
            masks["final"] = np.asarray(final_mask, dtype=bool).copy()
        self._debug_masks = masks
        self._refresh_display_highlight()
        self._refresh_detection_text()

    def _on_debug_layer_changed(self):
        self._set_debug_layer(str(self.debug_layer.currentData() or "off"))

    def _set_debug_layer(self, mode: str):
        normalized = str(mode)
        if normalized not in {"off", "bbox", "depth", "final"}:
            normalized = "off"
        self._debug_layer_mode = normalized
        for i in range(self.debug_layer.count()):
            if self.debug_layer.itemData(i) == normalized:
                if self.debug_layer.currentIndex() != i:
                    self.debug_layer.blockSignals(True)
                    self.debug_layer.setCurrentIndex(i)
                    self.debug_layer.blockSignals(False)
                break
        self._refresh_highlight_style()
        self._refresh_display_highlight()
        self._refresh_detection_text()
        mask = self._debug_masks.get(normalized) if normalized != "off" else self._highlight_mask
        self._log_operation(
            "debug_layer_changed",
            debug_layer=normalized,
            point_count=self._mask_count(mask),
            available_layers=sorted(self._debug_masks.keys()),
            keyframe_index=int(self.current_idx),
            source_image=self._current_source_image(),
        )

    def _debug_layer_suffix(self) -> str:
        mode = self._debug_layer_mode
        if mode == "off":
            return ""
        mask = self._debug_masks.get(mode)
        count = self._mask_count(mask)
        return f"\n调试层: {mode} · points={count:,}"

    def _set_detection_text(self, text: str):
        self._last_detection_base_text = str(text)
        self._refresh_detection_text()

    def _refresh_detection_text(self):
        if hasattr(self, "lbl_detection"):
            self.lbl_detection.setText(self._last_detection_base_text + self._debug_layer_suffix())

    def _apply_extraction_highlight(self, new_mask):
        """Set or accumulate the extraction highlight depending on 补充 mode.

        In supplement mode a new extraction is unioned into the existing
        highlight so disconnected door/window sub-regions a single seed could
        not reach can be added click by click. Otherwise it replaces.
        """
        if new_mask is None:
            if not self.cb_supplement.isChecked():
                self._set_highlight_mask(None)
                return None
            return self._highlight_mask
        new_mask = np.asarray(new_mask, dtype=bool)
        if self.cb_supplement.isChecked() and self._highlight_mask is not None:
            if len(self._highlight_mask) == len(new_mask):
                new_mask = self._highlight_mask | new_mask
        self._set_highlight_mask(new_mask)
        return self._highlight_mask

    def _clear_highlight(self):
        self._set_highlight_mask(None)
        self._clear_debug_masks()
        self._last_opening_candidate = None
        self.scene.set_selected_detection(-1)
        self.statusBar().showMessage("已清除高亮")
        self._log_operation("highlight_cleared", keyframe_index=int(self.current_idx))

    def _set_yaw_offset_value(self, degrees: float):
        degrees = float(degrees)
        for i in range(self.yaw_offset.count()):
            if abs(float(self.yaw_offset.itemData(i)) - degrees) < 1e-6:
                self.yaw_offset.setCurrentIndex(i)
                return
        self.yaw_offset.insertItem(0, f"标定 {degrees:+.1f}°", degrees)
        self.yaw_offset.setCurrentIndex(0)

    def _ensure_yaw_offset_value(self, degrees: float):
        degrees = float(degrees)
        for i in range(self.yaw_offset.count()):
            if abs(float(self.yaw_offset.itemData(i)) - degrees) < 1e-6:
                return
        self.yaw_offset.insertItem(0, f"标定 {degrees:+.1f}°", degrees)

    def _current_interaction_mode(self) -> str:
        return str(self.interaction_mode.currentData() or "navigate")

    def _apply_interaction_mode(self):
        mode = self._current_interaction_mode()
        self.scene.set_interaction_mode(mode)
        self.cloud_scene.set_interaction_mode("point" if mode == "point" else "navigate")

    def _on_interaction_mode_changed(self):
        self._apply_interaction_mode()
        messages = {
            "navigate": "导航模式（高亮保留）",
            "point": "点云 seed 提取：在上方或下方点云中点击门/窗上的点",
            "detection": "检测框提取：在上方全景检测框内点击",
        }
        mode = self._current_interaction_mode()
        self.statusBar().showMessage(messages.get(mode, messages["navigate"]))
        self._log_operation("interaction_mode_changed", mode=mode, keyframe_index=int(self.current_idx))

    def _on_cloud_point_clicked(self, idx: int):
        self._run_pointcloud_extraction(idx)

    def _normalize_detection_hit(self, hit) -> tuple[int, float | None, float | None] | None:
        if not hit:
            return None
        try:
            det_idx = int(hit[0])
        except (TypeError, ValueError, IndexError):
            return None
        if det_idx < 0 or det_idx >= len(self.current_detections):
            return None

        click_u = None
        click_v = None
        try:
            raw_u = float(hit[1])
            raw_v = float(hit[2])
        except (TypeError, ValueError, IndexError):
            raw_u = raw_v = float("nan")
        if np.isfinite(raw_u) and np.isfinite(raw_v):
            click_u = raw_u
            click_v = raw_v
        return det_idx, click_u, click_v

    def _consume_point_detection_hit(self) -> tuple[int, float | None, float | None] | None:
        consumer = getattr(self.scene, "consume_point_detection_hit", None)
        if not callable(consumer):
            return None
        return self._normalize_detection_hit(consumer())

    def _run_point_detection_fallback(self, seed_idx: int, detection_hit, fallback_reason: str) -> bool:
        normalized = self._normalize_detection_hit(detection_hit)
        if normalized is None:
            return False
        det_idx, click_u, click_v = normalized
        return self._run_detection_bbox_extraction(
            det_idx,
            click_u,
            click_v,
            text_prefix="检测框兜底提取",
            event_name="pointcloud_detection_fallback_extract",
            seed_index=seed_idx,
            extra_log_fields={
                "fallback_detection_index": int(det_idx),
                "fallback_reason": str(fallback_reason),
            },
        )

    def _run_pointcloud_extraction(self, idx: int, fallback_detection_hit=None):
        """Fusion-first door/window extraction from a clicked seed.

        Shared by the top (keyframe) and bottom (global) views: a click in
        either pane addresses the same shared world points, so the highlight
        is identical regardless of which view was clicked. In 补充 (supplement)
        mode the new region is unioned into the existing highlight.
        """
        if not self.dataset:
            return
        fallback_detection_hit = self._normalize_detection_hit(fallback_detection_hit)
        if idx < 0:
            if self._run_point_detection_fallback(idx, fallback_detection_hit, "point_pick_missed"):
                return
            self._set_detection_text("点云提取: 未吸附到点云点")
            self._apply_extraction_highlight(None)
            self._clear_debug_masks()
            self._last_opening_candidate = None
            self._log_operation(
                "pointcloud_extract_failed",
                level="warning",
                reason="point_pick_missed",
                keyframe_index=int(self.current_idx),
            )
            return

        supplement = self.cb_supplement.isChecked()
        prefix = "补充" if (supplement and self._highlight_mask is not None) else "融合提取"

        # Strategy A: if the current keyframe has detections and the clicked
        # seed falls inside one of their frustums, fuse the 2D box constraint
        # with the pure point-cloud geometry. Otherwise fall back to the
        # unconstrained pure-point-cloud extraction.
        fused = self._try_fused_extraction(idx)
        if fused is not None and fused.source == "fused":
            highlight = fused.mask if should_highlight_fused(fused) else None
            if highlight is None and self._run_point_detection_fallback(
                idx,
                fallback_detection_hit,
                str(getattr(fused, "reason", "fused_highlight_rejected")),
            ):
                return
            effective_highlight = self._set_opening_candidate_from_extraction(idx, fused, highlight)
            self._log_opening_selection_event("extract_opening_candidate", idx, fused, effective_highlight)
            self.scene.set_selected_detection(fused.detection_index)
            width = f"{fused.width_m:.2f}" if fused.width_m is not None else "—"
            height = f"{fused.height_m:.2f}" if fused.height_m is not None else "—"
            score = f"{fused.score:.3f}" if fused.score is not None else "—"
            total = int(self._highlight_mask.sum()) if self._highlight_mask is not None else fused.point_count
            diagnostics = self._format_extraction_diagnostics(fused)
            self._set_detection_text(
                f"{prefix} #{idx} (框#{fused.detection_index} score={score})\n"
                f"label: {fused.label} · confidence: {fused.confidence}\n"
                f"this: {fused.point_count:,} · total: {total:,}\n"
                f"size: {width} × {height} m\n"
                f"reason: {fused.reason}{diagnostics}"
            )
            self._log_operation(
                "pointcloud_fused_extract",
                **self._selection_log_fields(fused, seed_index=idx),
            )
            return

        fallback_reason = "no_fused_result" if fused is None else str(getattr(fused, "reason", "no_fused_result"))
        if self._run_point_detection_fallback(idx, fallback_detection_hit, fallback_reason):
            return

        selection = extract_planar_region_from_seed(self.dataset.points, idx)
        highlight = selection.mask if should_highlight_planar_region(selection) else None
        effective_highlight = self._set_opening_candidate_from_extraction(idx, selection, highlight)
        self._log_opening_selection_event("extract_opening_candidate", idx, selection, effective_highlight)
        if not supplement:
            self.scene.set_selected_detection(-1)
        width = f"{selection.width_m:.2f}" if selection.width_m is not None else "—"
        height = f"{selection.height_m:.2f}" if selection.height_m is not None else "—"
        total = int(self._highlight_mask.sum()) if self._highlight_mask is not None else selection.point_count
        pure_prefix = "补充" if (supplement and self._highlight_mask is not None) else "点云提取"
        diagnostics = self._format_extraction_diagnostics(selection)
        self._set_detection_text(
            f"{pure_prefix} #{idx} (纯点云)\n"
            f"label: {selection.label} · confidence: {selection.confidence}\n"
            f"this: {selection.point_count:,} · total: {total:,}\n"
            f"size: {width} × {height} m\n"
            f"reason: {selection.reason}{diagnostics}"
        )
        self._log_operation(
            "pointcloud_pure_extract",
            **self._selection_log_fields(selection, seed_index=idx),
        )

    def _opening_candidate_from_selection(self, seed_idx: int, selection, highlight):
        if highlight is None:
            return None
        return {
            "mask": np.asarray(highlight, dtype=bool),
            "label": selection.label,
            "seed_index": int(seed_idx),
            "confidence": selection.confidence,
            "reason": selection.reason,
            "plane_point": selection.plane_point,
            "plane_normal": selection.plane_normal,
            "width_m": selection.width_m,
            "height_m": selection.height_m,
            "detection_index": getattr(selection, "detection_index", -1),
            "score": getattr(selection, "score", None),
        }

    def _set_opening_candidate_from_extraction(self, seed_idx: int, selection, latest_highlight):
        effective_highlight = self._apply_extraction_highlight(latest_highlight)
        self._set_debug_masks_from_selection(selection, latest_highlight)
        if latest_highlight is None and self.cb_supplement.isChecked():
            self._last_opening_candidate = None
            return effective_highlight
        self._last_opening_candidate = self._opening_candidate_from_selection(
            seed_idx,
            selection,
            effective_highlight,
        )
        return effective_highlight

    def _log_opening_selection_event(self, event: str, seed_idx: int, selection, highlight):
        if self.dataset is None:
            return
        payload = self._opening_selection_event_payload(event, seed_idx, selection, highlight)
        append_wall_opening_event(self.workspace, self.dataset.data_root, payload)

    def _opening_selection_event_payload(self, event: str, seed_idx: int, selection, highlight) -> dict:
        highlight_mask = None if highlight is None else np.asarray(highlight, dtype=bool)
        payload = {
            "event": event,
            "source_image": self._current_source_image(),
            "keyframe_index": int(self.current_idx),
            "seed_index": int(seed_idx),
            "label": str(getattr(selection, "label", "")),
            "confidence": str(getattr(selection, "confidence", "")),
            "reason": str(getattr(selection, "reason", "")),
            "selection_source": str(getattr(selection, "source", "")),
            "detection_index": int(getattr(selection, "detection_index", -1)),
            "score": self._json_float_or_none(getattr(selection, "score", None)),
            "width_m": self._json_float_or_none(getattr(selection, "width_m", None)),
            "height_m": self._json_float_or_none(getattr(selection, "height_m", None)),
            "candidate_point_count": int(getattr(selection, "point_count", 0)),
            "highlight_point_count": self._mask_count(self._highlight_mask),
            "supplement_mode": bool(self.cb_supplement.isChecked()),
            "event_mask_source": "effective_highlight" if self.cb_supplement.isChecked() else "latest_highlight",
            "yaw_offset_deg": self._json_float_or_none(self.yaw_offset.currentData()),
        }
        diagnostics = self._json_extraction_diagnostics(getattr(selection, "diagnostics", None))
        if diagnostics:
            payload["extraction_diagnostics"] = diagnostics
        if self.dataset is not None and highlight_mask is not None and len(highlight_mask) == len(self.dataset.points):
            selected = np.asarray(self.dataset.points, dtype=np.float64)[highlight_mask]
            if len(selected) > 0:
                payload["candidate_bbox_min"] = self._rounded_list(selected.min(axis=0))
                payload["candidate_bbox_max"] = self._rounded_list(selected.max(axis=0))
                payload["candidate_center"] = self._rounded_list(selected.mean(axis=0))
        return payload

    def _record_current_opening(self):
        if self.dataset is None or self._last_opening_candidate is None:
            self.statusBar().showMessage("请先用点云门窗提取选中一个门窗区域")
            self._log_operation(
                "record_opening_rejected",
                level="warning",
                reason="no_current_candidate",
                keyframe_index=int(self.current_idx),
            )
            return
        reason = str(self._last_opening_candidate.get("reason", ""))
        label = str(self._last_opening_candidate.get("label", "")).lower()
        confidence = str(self._last_opening_candidate.get("confidence", ""))
        if not is_recordable_opening_metadata(label, reason, confidence):
            self.statusBar().showMessage(f"当前候选未通过门窗几何校验，未记录: {reason or label}")
            self._log_operation(
                "record_opening_rejected",
                level="warning",
                reason="unrecordable_candidate",
                candidate_reason=reason,
                label=label,
                confidence=confidence,
                keyframe_index=int(self.current_idx),
            )
            return
        source_image = self._current_source_image()
        try:
            opening = opening_from_selection(
                self.dataset.points,
                self._last_opening_candidate["mask"],
                label=str(self._last_opening_candidate.get("label", "object")),
                source_image=source_image,
                seed_index=int(self._last_opening_candidate.get("seed_index", -1)),
                confidence=str(self._last_opening_candidate.get("confidence", "")),
                reason=str(self._last_opening_candidate.get("reason", "")),
                plane_point=self._last_opening_candidate.get("plane_point"),
                plane_normal=self._last_opening_candidate.get("plane_normal"),
                width_m=self._last_opening_candidate.get("width_m"),
                height_m=self._last_opening_candidate.get("height_m"),
                detection_index=int(self._last_opening_candidate.get("detection_index", -1)),
                score=self._last_opening_candidate.get("score"),
            )
            saved = append_wall_opening(self.workspace, self.dataset.data_root, opening)
            self._log_recorded_opening(saved)
        except Exception as exc:
            self._log_operation(
                "record_opening_failed",
                level="error",
                error=exc,
                reason=reason,
                label=label,
                keyframe_index=int(self.current_idx),
            )
            QMessageBox.warning(self, "门窗记录失败", str(exc))
            return
        self.statusBar().showMessage(f"已记录门窗: {saved.id} ({saved.label})")
        self._log_operation(
            "record_opening_saved",
            opening_id=saved.id,
            label=saved.label,
            point_count=saved.point_count,
            reason=saved.reason,
            confidence=saved.confidence,
            source_image=saved.source_image,
            keyframe_index=int(self.current_idx),
            detection_index=saved.detection_index,
        )

    def _log_recorded_opening(self, opening):
        if self.dataset is None:
            return
        candidate_mask = np.asarray(self._last_opening_candidate.get("mask", []), dtype=bool)
        append_wall_opening_event(
            self.workspace,
            self.dataset.data_root,
            {
                "event": "record_opening_saved",
                "opening_id": opening.id,
                "source_image": opening.source_image,
                "keyframe_index": int(self.current_idx),
                "seed_index": opening.seed_index,
                "label": opening.label,
                "confidence": opening.confidence,
                "reason": opening.reason,
                "detection_index": opening.detection_index,
                "score": self._json_float_or_none(opening.score),
                "width_m": self._json_float_or_none(opening.width_m),
                "height_m": self._json_float_or_none(opening.height_m),
                "point_count": opening.point_count,
                "candidate_point_count": self._mask_count(candidate_mask),
                "highlight_point_count": self._mask_count(self._highlight_mask),
                "supplement_mode": bool(self.cb_supplement.isChecked()),
                "record_mask_source": "accumulated_highlight" if self.cb_supplement.isChecked() else "latest_highlight",
                "yaw_offset_deg": self._json_float_or_none(self.yaw_offset.currentData()),
                "center": list(opening.center),
                "normal": list(opening.normal),
                "bbox_min": list(opening.bbox_min),
                "bbox_max": list(opening.bbox_max),
                "z_min": opening.z_min,
                "z_max": opening.z_max,
            },
        )

    def _current_source_image(self) -> str:
        if self.dataset is not None and 0 <= self.current_idx < len(self.dataset.poses):
            return self.dataset.poses[self.current_idx].image_name
        return ""

    @staticmethod
    def _mask_count(mask) -> int:
        if mask is None:
            return 0
        return int(np.asarray(mask, dtype=bool).sum())

    @staticmethod
    def _json_float_or_none(value):
        if value is None:
            return None
        return round(float(value), 5)

    @staticmethod
    def _rounded_list(values) -> list[float]:
        return [round(float(v), 5) for v in np.asarray(values, dtype=np.float64).reshape(3)]

    def _log_operation(self, event: str, *, level: str = "info", **fields):
        try:
            return log_operation(
                self.workspace,
                event,
                component="main_window",
                level=level,
                **fields,
            )
        except Exception:
            return None

    @staticmethod
    def _safe_positive_int(value) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _json_extraction_diagnostics(diagnostics) -> dict:
        if not isinstance(diagnostics, dict):
            return {}
        out = {}
        for key, value in diagnostics.items():
            if isinstance(value, (np.integer,)):
                out[str(key)] = int(value)
            elif isinstance(value, (np.floating,)):
                out[str(key)] = round(float(value), 5)
            elif isinstance(value, float):
                out[str(key)] = round(value, 5)
            elif isinstance(value, (str, int, bool)) or value is None:
                out[str(key)] = value
        return out

    def _format_extraction_diagnostics(self, selection) -> str:
        diagnostics = getattr(selection, "diagnostics", None)
        if not isinstance(diagnostics, dict):
            return ""
        parts: list[str] = []
        bbox_count = diagnostics.get("bbox_candidate_count")
        if bbox_count is not None:
            parts.append(f"bbox={int(bbox_count):,}")
        depth_count = diagnostics.get("selected_depth_candidate_count")
        if depth_count is not None:
            parts.append(f"depth={int(depth_count):,}")
        seed_count = diagnostics.get("seed_attempt_count")
        if seed_count is not None:
            parts.append(f"seeds={int(seed_count)}")
        total_ms = diagnostics.get("total_ms")
        if total_ms is not None:
            parts.append(f"time={float(total_ms):.1f}ms")
        if diagnostics.get("projection_cache_hit") or diagnostics.get("candidate_cache_hit"):
            parts.append("cache=hit")
        return "" if not parts else "\n" + "diag: " + " · ".join(parts)

    def _selection_log_fields(self, selection, *, seed_index: int | None = None) -> dict:
        fields = {
            "keyframe_index": int(self.current_idx),
            "source_image": self._current_source_image(),
            "label": str(getattr(selection, "label", "")),
            "confidence": str(getattr(selection, "confidence", "")),
            "reason": str(getattr(selection, "reason", "")),
            "selection_source": str(getattr(selection, "source", "")),
            "detection_index": int(getattr(selection, "detection_index", -1)),
            "point_count": int(getattr(selection, "point_count", 0)),
            "highlight_point_count": self._mask_count(self._highlight_mask),
            "width_m": self._json_float_or_none(getattr(selection, "width_m", None)),
            "height_m": self._json_float_or_none(getattr(selection, "height_m", None)),
            "score": self._json_float_or_none(getattr(selection, "score", None)),
            "supplement_mode": bool(self.cb_supplement.isChecked()),
            "yaw_offset_deg": self._json_float_or_none(self.yaw_offset.currentData()),
        }
        if seed_index is not None:
            fields["seed_index"] = int(seed_index)
        diagnostics = self._json_extraction_diagnostics(getattr(selection, "diagnostics", None))
        if diagnostics:
            fields["diagnostics"] = diagnostics
        return fields

    def _try_fused_extraction(self, idx: int):
        """Run frustum-fused extraction for the current keyframe, or None."""
        if self.current_idx < 0 or not self.current_detections:
            return None
        pose = self.dataset.poses[self.current_idx]
        img_w, img_h = self.current_image_size or (0, 0)
        if img_w <= 0 or img_h <= 0:
            return None
        R = rotation_from_angle(pose.roll, pose.pitch, pose.yaw)
        yaw_offset = float(self.yaw_offset.currentData())
        return fuse_detection_and_pointcloud(
            self.dataset.points,
            idx,
            self.current_detections,
            pose.position,
            R,
            img_w,
            img_h,
            yaw_offset_deg=yaw_offset,
            cache=self._fusion_cache,
        )

    def _run_detection_bbox_extraction(
        self,
        detection_index: int,
        click_u: float | None = None,
        click_v: float | None = None,
        *,
        text_prefix: str = "检测框提取",
        event_name: str = "detection_bbox_extract",
        seed_index: int = -1,
        extra_log_fields: dict | None = None,
    ) -> bool:
        if not self.dataset or self.current_idx < 0:
            return False
        if detection_index < 0 or detection_index >= len(self.current_detections):
            return False
        pose = self.dataset.poses[self.current_idx]
        img_w, img_h = self.current_image_size or (0, 0)
        if img_w <= 0 or img_h <= 0:
            return False
        R = rotation_from_angle(pose.roll, pose.pitch, pose.yaw)
        yaw_offset = float(self.yaw_offset.currentData())
        click_uv = None
        if click_u is not None and click_v is not None:
            click_uv = (float(click_u), float(click_v))
        selection = extract_detection_region_from_bbox(
            self.dataset.points,
            int(detection_index),
            self.current_detections,
            pose.position,
            R,
            img_w,
            img_h,
            yaw_offset_deg=yaw_offset,
            click_uv=click_uv,
            cache=self._fusion_cache,
        )
        highlight = selection.mask if should_highlight_fused(selection) else None
        effective_highlight = self._set_opening_candidate_from_extraction(seed_index, selection, highlight)
        self._log_opening_selection_event("extract_opening_candidate", seed_index, selection, effective_highlight)
        self.scene.set_selected_detection(selection.detection_index)
        width = f"{selection.width_m:.2f}" if selection.width_m is not None else "—"
        height = f"{selection.height_m:.2f}" if selection.height_m is not None else "—"
        score = f"{selection.score:.3f}" if selection.score is not None else "—"
        total = int(self._highlight_mask.sum()) if self._highlight_mask is not None else selection.point_count
        diagnostics = self._format_extraction_diagnostics(selection)
        self._set_detection_text(
            f"{text_prefix} (框#{selection.detection_index} score={score})\n"
            f"label: {selection.label} · confidence: {selection.confidence}\n"
            f"this: {selection.point_count:,} · total: {total:,}\n"
            f"size: {width} × {height} m\n"
            f"reason: {selection.reason}{diagnostics}"
        )
        log_fields = {
            "click_uv": click_uv,
            **self._selection_log_fields(selection, seed_index=seed_index),
        }
        if extra_log_fields:
            log_fields.update(extra_log_fields)
        self._log_operation(
            event_name,
            **log_fields,
        )
        return True

    def _on_detection_clicked(
        self,
        detection_index: int,
        click_u: float | None = None,
        click_v: float | None = None,
    ):
        self._run_detection_bbox_extraction(detection_index, click_u, click_v)

    def _load_detections(self, pose: CameraPose):
        self._fusion_cache.clear()
        self._clear_debug_masks()
        self.current_detections = []
        self.current_image_size = None
        self.scene.set_detections([], 0, 0)
        path = find_detection_json(self.workspace, pose.image_name)
        if path is None:
            self._set_detection_text("检测: 未找到当前 keyframe JSON\n请使用“编辑当前全景框”绘制门窗框")
            self._log_operation(
                "detection_load_not_found",
                level="warning",
                image_name=pose.image_name,
                keyframe_index=int(self.current_idx),
            )
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self._set_detection_text(f"检测: 读取失败\n{path.name}\n{e}")
            self._log_operation(
                "detection_load_failed",
                level="error",
                reason="read_json_failed",
                image_name=pose.image_name,
                path=path,
                error=e,
                keyframe_index=int(self.current_idx),
            )
            return
        img_w = self._safe_positive_int(payload.get("width"))
        img_h = self._safe_positive_int(payload.get("height"))
        if img_w is None or img_h is None:
            self._set_detection_text(
                f"检测: 尺寸无效\n{path.name}\nwidth={payload.get('width')!r} height={payload.get('height')!r}"
            )
            self._log_operation(
                "detection_load_failed",
                level="error",
                reason="invalid_image_size",
                image_name=pose.image_name,
                path=path,
                width=payload.get("width"),
                height=payload.get("height"),
                keyframe_index=int(self.current_idx),
            )
            return
        self.current_image_size = (img_w, img_h)
        self.current_detections = normalize_detections(
            payload.get("detections", []),
            img_w,
            img_h,
        )
        self.scene.set_detections(
            self.current_detections,
            img_w,
            img_h,
        )
        self._set_detection_text(
            f"检测: {len(self.current_detections)} boxes\n"
            f"{path.relative_to(self.workspace)}"
        )
        self._log_operation(
            "detection_loaded",
            image_name=pose.image_name,
            path=path,
            keyframe_index=int(self.current_idx),
            image_size=[img_w, img_h],
            raw_detection_count=len(payload.get("detections", [])) if isinstance(payload.get("detections", []), list) else None,
            normalized_detection_count=len(self.current_detections),
        )

    def _edit_current_frame(self):
        if not self.dataset or self.current_idx < 0:
            return
        pose = self.dataset.poses[self.current_idx]
        image_path = self.dataset.image_dir / pose.image_name
        if not image_path.exists():
            self._log_operation(
                "annotation_edit_failed",
                level="warning",
                reason="image_missing",
                image_name=pose.image_name,
                image_path=image_path,
                keyframe_index=int(self.current_idx),
            )
            QMessageBox.warning(self, "缺图", str(image_path))
            return
        self.editor.load_image(image_path, self.current_detections)
        self.view_stack.setCurrentWidget(self.editor)
        self._sync_workspace_buttons()
        self.statusBar().showMessage(
            "正在编辑当前全景框：拖拽画框，点击框选中，Delete 删除，编辑结束保存"
        )
        self._log_operation(
            "annotation_edit_started",
            image_name=pose.image_name,
            image_path=image_path,
            detection_count=len(self.current_detections),
            keyframe_index=int(self.current_idx),
        )

    def _on_annotation_saved(self, path_text: str):
        self._exit_annotation_editor()
        if self.dataset and self.current_idx >= 0:
            pose = self.dataset.poses[self.current_idx]
            self._load_detections(pose)
        self.statusBar().showMessage(f"手工框已保存: {Path(path_text).name}")
        self._log_operation(
            "annotation_saved",
            path=path_text,
            keyframe_index=int(self.current_idx),
        )

    def _exit_annotation_editor(self):
        self._show_main_view()

    def _show_main_view(self):
        self.side_panel.show()
        self.view_stack.setCurrentWidget(self.normal_view)
        self._sync_workspace_buttons()
        self.statusBar().showMessage("已切回点云主视图")
        self._log_operation("view_changed", view="main", keyframe_index=int(self.current_idx))

    def _show_wall_model(self):
        self.view_stack.setCurrentWidget(self.wall_model_view)
        self.side_panel.hide()
        self._sync_workspace_buttons()
        self.statusBar().showMessage("已打开墙体建模工作台")
        self._log_operation("view_changed", view="wall_model", keyframe_index=int(self.current_idx))

    def _sync_workspace_buttons(self):
        current = self.view_stack.currentWidget()
        self.btn_main_view.setChecked(current is self.normal_view)
        self.btn_wall_model.setChecked(current is self.wall_model_view)

    def _on_wall_model_status(self, message: str):
        self.statusBar().showMessage(message)
        self._log_operation("wall_model_status", message=message, keyframe_index=int(self.current_idx))

    def _on_point_clicked(self, idx: int):
        if not self.dataset or self.current_idx < 0:
            return
        # The top keyframe view is only clickable in point-cloud extraction mode;
        # use the same fusion-first seed extraction as the bottom view, with the
        # clicked 2D detection box as a constrained fallback if seed picking hits
        # an unrelated projected point.
        self._run_pointcloud_extraction(idx, fallback_detection_hit=self._consume_point_detection_hit())

    # 全屏快捷键转给 scene
    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Left,):
            self._step(-1)
        elif e.key() in (Qt.Key_Right,):
            self._step(1)
        else:
            super().keyPressEvent(e)
