"""Qt in-window editor for flat panorama door/window boxes."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from core.annotations import save_manual_annotations


class PanoAnnotationEditor(QWidget):
    saved = Signal(str)
    canceled = Signal()

    def __init__(self, workspace: Path, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self.image_path: Path | None = None
        self._canvas = _PanoCanvas()

        self.lbl_title = QLabel("全景框编辑")
        self.lbl_title.setWordWrap(True)

        self.label_combo = QComboBox()
        self.label_combo.addItem("window")
        self.label_combo.addItem("door")
        self.label_combo.currentTextChanged.connect(self._canvas.set_current_label)

        self.btn_delete = QPushButton("删除选中框")
        self.btn_delete.clicked.connect(self._canvas.delete_selected)

        self.btn_save = QPushButton("编辑结束")
        self.btn_save.clicked.connect(self._save)

        self.btn_cancel = QPushButton("取消编辑")
        self.btn_cancel.clicked.connect(self.canceled.emit)

        tools = QHBoxLayout()
        tools.addWidget(QLabel("类别"))
        tools.addWidget(self.label_combo)
        tools.addWidget(self.btn_delete)
        tools.addStretch(1)
        tools.addWidget(self.btn_cancel)
        tools.addWidget(self.btn_save)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setWidget(self._canvas)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.lbl_title)
        layout.addLayout(tools)
        layout.addWidget(scroll, 1)

    def load_image(self, image_path: Path, detections: list[dict]):
        self.image_path = image_path
        self.lbl_title.setText(f"全景框编辑: {image_path.name}")
        self._canvas.load_image(image_path, detections)

    def _save(self):
        if self.image_path is None:
            return
        out = save_manual_annotations(
            workspace=self.workspace,
            image_path=self.image_path,
            width=self._canvas.image_width,
            height=self._canvas.image_height,
            detections=self._canvas.detections(),
        )
        self.saved.emit(str(out))


class _PanoCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(800, 400)
        self._pixmap = QPixmap()
        self._img_w = 0
        self._img_h = 0
        self._detections: list[dict] = []
        self._selected = -1
        self._drawing = False
        self._start = QPoint()
        self._draft = QRect()
        self._current_label = "window"

    @property
    def image_width(self) -> int:
        return self._img_w

    @property
    def image_height(self) -> int:
        return self._img_h

    def load_image(self, image_path: Path, detections: list[dict]):
        image = QImage(str(image_path))
        if image.isNull():
            self._pixmap = QPixmap()
            self._img_w = 0
            self._img_h = 0
            self._detections = []
            self.update()
            return
        self._pixmap = QPixmap.fromImage(image)
        self._img_w = image.width()
        self._img_h = image.height()
        self._detections = [self._normalized_detection(det) for det in detections if self._valid_bbox(det)]
        self._selected = -1
        self.updateGeometry()
        self.update()

    def detections(self) -> list[dict]:
        return [dict(det) for det in self._detections]

    def set_current_label(self, label: str):
        self._current_label = label
        if 0 <= self._selected < len(self._detections):
            self._detections[self._selected]["label"] = label
            self.update()

    def delete_selected(self):
        if 0 <= self._selected < len(self._detections):
            self._detections.pop(self._selected)
            self._selected = -1
            self.update()

    def sizeHint(self):
        if self._pixmap.isNull():
            return super().sizeHint()
        return self._pixmap.size().scaled(1200, 600, Qt.KeepAspectRatio)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 22))
        target = self._image_rect()
        if self._pixmap.isNull() or target.isEmpty():
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignCenter, "无法加载当前全景图")
            return
        painter.drawPixmap(target, self._pixmap)
        for i, det in enumerate(self._detections):
            self._draw_detection(painter, det, i == self._selected)
        if self._drawing and not self._draft.isNull():
            painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.DashLine))
            painter.drawRect(self._draft.normalized())

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return
        self.setFocus()
        pos = event.position().toPoint()
        hit = self._hit_test(pos)
        if hit >= 0:
            self._selected = hit
            self.update()
            return
        if not self._image_rect().contains(pos):
            self._selected = -1
            self.update()
            return
        self._drawing = True
        self._start = pos
        self._draft = QRect(pos, pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._drawing:
            return
        pos = self._clamp_to_image(event.position().toPoint())
        self._draft = QRect(self._start, pos)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton or not self._drawing:
            return
        self._drawing = False
        rect = self._draft.normalized()
        self._draft = QRect()
        if rect.width() < 8 or rect.height() < 8:
            self.update()
            return
        bbox = self._widget_rect_to_bbox(rect)
        self._detections.append({
            "label": self._current_label,
            "score": 1.0,
            "source": "manual",
            "bbox": bbox,
        })
        self._selected = len(self._detections) - 1
        self.update()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
            return
        super().keyPressEvent(event)

    def _image_rect(self) -> QRect:
        if self._pixmap.isNull() or self._img_w <= 0 or self._img_h <= 0:
            return QRect()
        scaled = self._pixmap.size().scaled(self.size(), Qt.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    def _draw_detection(self, painter: QPainter, det: dict, selected: bool):
        rect = self._bbox_to_widget_rect(det["bbox"])
        color = QColor(255, 157, 46) if det.get("label") == "door" else QColor(0, 216, 255)
        if selected:
            color = QColor(255, 42, 42)
        painter.setPen(QPen(color, 3))
        painter.fillRect(rect, QColor(color.red(), color.green(), color.blue(), 28))
        painter.drawRect(rect)
        painter.fillRect(QRect(rect.left(), rect.top() - 22, 96, 22), QColor(0, 0, 0, 180))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect.left() + 5, rect.top() - 6, str(det.get("label", "window")))

    def _bbox_to_widget_rect(self, bbox: list[float]) -> QRect:
        target = self._image_rect()
        sx = target.width() / max(1, self._img_w)
        sy = target.height() / max(1, self._img_h)
        x1, y1, x2, y2 = bbox
        return QRect(
            round(target.left() + x1 * sx),
            round(target.top() + y1 * sy),
            round((x2 - x1) * sx),
            round((y2 - y1) * sy),
        )

    def _widget_rect_to_bbox(self, rect: QRect) -> list[float]:
        target = self._image_rect()
        sx = self._img_w / max(1, target.width())
        sy = self._img_h / max(1, target.height())
        x1 = (rect.left() - target.left()) * sx
        y1 = (rect.top() - target.top()) * sy
        x2 = (rect.right() - target.left()) * sx
        y2 = (rect.bottom() - target.top()) * sy
        return [
            max(0.0, min(float(self._img_w), x1)),
            max(0.0, min(float(self._img_h), y1)),
            max(0.0, min(float(self._img_w), x2)),
            max(0.0, min(float(self._img_h), y2)),
        ]

    def _hit_test(self, pos: QPoint) -> int:
        for i in range(len(self._detections) - 1, -1, -1):
            if self._bbox_to_widget_rect(self._detections[i]["bbox"]).contains(pos):
                return i
        return -1

    def _clamp_to_image(self, pos: QPoint) -> QPoint:
        rect = self._image_rect()
        return QPoint(
            max(rect.left(), min(rect.right(), pos.x())),
            max(rect.top(), min(rect.bottom(), pos.y())),
        )

    def _valid_bbox(self, det: dict) -> bool:
        bbox = det.get("bbox")
        return isinstance(bbox, list) and len(bbox) == 4

    def _normalized_detection(self, det: dict) -> dict:
        label = det.get("label") if det.get("label") in ("door", "window") else "window"
        return {
            "label": label,
            "score": float(det.get("score", 1.0)),
            "source": det.get("source", "manual"),
            "bbox": [float(v) for v in det["bbox"]],
        }
