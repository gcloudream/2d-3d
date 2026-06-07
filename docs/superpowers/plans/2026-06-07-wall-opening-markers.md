# Wall Opening Markers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record extracted door/window point-cloud regions and show them as rectangular markers in generated wall top-down images and OBJ wall models.

**Architecture:** Add a focused `core.wall_openings` module for persistence and geometry records. Extend `core.wall_model` with opening-to-wall matching and marker-frame mesh generation while keeping the existing wall mesh intact. Wire the main viewer to save the current highlighted extraction and the wall-model workbench to load those records automatically.

**Tech Stack:** Python 3.12, PySide6, NumPy, Pillow, unittest, existing OBJ writer and wall-model pipeline.

---

## File Structure

- Create `3d_viewer/core/wall_openings.py`: dataclasses, JSON persistence, and extraction-result-to-opening conversion.
- Create `3d_viewer/tests/test_wall_openings.py`: persistence and conversion tests.
- Modify `3d_viewer/ui/main_window.py`: keep the last extraction candidate and add `记录当前门窗`.
- Create `3d_viewer/tests/test_main_window_wall_openings.py`: main-window record button tests.
- Modify `3d_viewer/core/wall_model.py`: opening matching, top-down drawing, OBJ marker-frame mesh, metadata counts.
- Extend `3d_viewer/tests/test_wall_model.py`: marker matching and OBJ frame tests.
- Modify `3d_viewer/wall_model_tool.py`: load saved openings before generating and show matched/unmatched counts.
- Modify `3d_viewer/generate_wall_model.py`: load saved openings for CLI generation.
- Extend `3d_viewer/tests/test_wall_model_tool.py` and `3d_viewer/tests/test_generate_wall_model_cli.py` only if behavior changes require new assertions.

---

### Task 1: Add Opening Persistence

**Files:**
- Create: `3d_viewer/core/wall_openings.py`
- Create: `3d_viewer/tests/test_wall_openings.py`

- [ ] **Step 1: Write failing tests for persistence and conversion**

Create `3d_viewer/tests/test_wall_openings.py`:

```python
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.wall_openings import (
    WallOpening,
    append_wall_opening,
    load_wall_openings,
    opening_from_selection,
    save_wall_openings,
    wall_openings_path,
)


class WallOpeningsTest(unittest.TestCase):
    def test_saves_and_loads_openings_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            opening = WallOpening(
                id="window-0001",
                label="window",
                source_image="608.jpg",
                seed_index=12,
                point_count=4,
                confidence="high",
                reason="accepted_vertical_planar_region",
                center=(1.0, 2.0, 1.2),
                normal=(1.0, 0.0, 0.0),
                bbox_min=(0.8, 1.6, 0.9),
                bbox_max=(1.2, 2.4, 1.5),
                width_m=0.8,
                height_m=0.6,
                z_min=0.9,
                z_max=1.5,
                detection_index=3,
                score=0.91,
            )

            save_wall_openings(workspace, data_root, [opening])
            loaded = load_wall_openings(workspace, data_root)

        self.assertEqual(loaded, [opening])

    def test_append_assigns_next_stable_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            opening = WallOpening(
                id="",
                label="door",
                source_image="frame.jpg",
                seed_index=2,
                point_count=3,
                confidence="high",
                reason="accepted_vertical_planar_region",
                center=(0.0, 0.0, 1.0),
                normal=(0.0, 1.0, 0.0),
                bbox_min=(-0.4, -0.1, 0.0),
                bbox_max=(0.4, 0.1, 2.0),
                width_m=0.8,
                height_m=2.0,
                z_min=0.0,
                z_max=2.0,
                detection_index=-1,
                score=None,
            )

            saved = append_wall_opening(workspace, data_root, opening)

        self.assertEqual(saved.id, "door-0001")
        self.assertEqual(load_wall_openings(workspace, data_root)[0].id, "door-0001")

    def test_builds_opening_from_selection_mask(self):
        points = np.asarray([
            [1.0, 0.0, 0.8],
            [1.0, 0.4, 1.0],
            [1.0, 0.8, 1.6],
            [3.0, 3.0, 0.0],
        ])
        mask = np.asarray([True, True, True, False])

        opening = opening_from_selection(
            points,
            mask,
            label="window",
            source_image="608.jpg",
            seed_index=1,
            confidence="high",
            reason="accepted_vertical_planar_region",
            plane_point=np.asarray([1.0, 0.4, 1.0]),
            plane_normal=np.asarray([1.0, 0.0, 0.0]),
            width_m=0.8,
            height_m=0.8,
            detection_index=2,
            score=0.8,
        )

        self.assertEqual(opening.label, "window")
        self.assertEqual(opening.point_count, 3)
        self.assertEqual(opening.center, (1.0, 0.4, 1.13333))
        self.assertEqual(opening.bbox_min, (1.0, 0.0, 0.8))
        self.assertEqual(opening.bbox_max, (1.0, 0.8, 1.6))
        self.assertEqual(opening.z_min, 0.8)
        self.assertEqual(opening.z_max, 1.6)

    def test_wall_openings_path_uses_dataset_name(self):
        path = wall_openings_path(Path("/workspace"), Path("/workspace/scan-a"))

        self.assertEqual(path, Path("/workspace/out/wall_openings/scan-a_openings.json"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_openings.py
```

Expected: fails because `core.wall_openings` does not exist.

- [ ] **Step 3: Implement `core.wall_openings`**

Create `3d_viewer/core/wall_openings.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WallOpening:
    id: str
    label: str
    source_image: str
    seed_index: int
    point_count: int
    confidence: str
    reason: str
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    width_m: float | None
    height_m: float | None
    z_min: float
    z_max: float
    detection_index: int
    score: float | None


def wall_openings_path(workspace: Path, data_root: Path) -> Path:
    return Path(workspace) / "out" / "wall_openings" / f"{Path(data_root).name}_openings.json"


def load_wall_openings(workspace: Path, data_root: Path) -> list[WallOpening]:
    path = wall_openings_path(workspace, data_root)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [WallOpening(**item) for item in payload.get("openings", [])]


def save_wall_openings(workspace: Path, data_root: Path, openings: list[WallOpening]) -> Path:
    path = wall_openings_path(workspace, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"openings": [asdict(item) for item in openings]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def append_wall_opening(workspace: Path, data_root: Path, opening: WallOpening) -> WallOpening:
    existing = load_wall_openings(workspace, data_root)
    label = opening.label or "opening"
    saved = WallOpening(
        id=opening.id or f"{label}-{len(existing) + 1:04d}",
        label=opening.label,
        source_image=opening.source_image,
        seed_index=opening.seed_index,
        point_count=opening.point_count,
        confidence=opening.confidence,
        reason=opening.reason,
        center=opening.center,
        normal=opening.normal,
        bbox_min=opening.bbox_min,
        bbox_max=opening.bbox_max,
        width_m=opening.width_m,
        height_m=opening.height_m,
        z_min=opening.z_min,
        z_max=opening.z_max,
        detection_index=opening.detection_index,
        score=opening.score,
    )
    save_wall_openings(workspace, data_root, [*existing, saved])
    return saved


def opening_from_selection(
    points: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
    source_image: str,
    seed_index: int,
    confidence: str,
    reason: str,
    plane_point: np.ndarray | None,
    plane_normal: np.ndarray | None,
    width_m: float | None,
    height_m: float | None,
    detection_index: int = -1,
    score: float | None = None,
) -> WallOpening:
    selected = np.asarray(points, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if len(selected) == 0:
        raise ValueError("cannot record an opening without selected points")
    bbox_min = selected.min(axis=0)
    bbox_max = selected.max(axis=0)
    center = selected.mean(axis=0)
    normal = np.asarray(plane_normal if plane_normal is not None else [0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm > 0.0:
        normal = normal / norm
    return WallOpening(
        id="",
        label=label or "object",
        source_image=source_image,
        seed_index=int(seed_index),
        point_count=int(len(selected)),
        confidence=confidence,
        reason=reason,
        center=_rounded_xyz(center),
        normal=_rounded_xyz(normal),
        bbox_min=_rounded_xyz(bbox_min),
        bbox_max=_rounded_xyz(bbox_max),
        width_m=None if width_m is None else round(float(width_m), 5),
        height_m=None if height_m is None else round(float(height_m), 5),
        z_min=round(float(bbox_min[2]), 5),
        z_max=round(float(bbox_max[2]), 5),
        detection_index=int(detection_index),
        score=None if score is None else round(float(score), 5),
    )


def _rounded_xyz(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(round(float(v), 5) for v in values.reshape(3))
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_openings.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add 3d_viewer/core/wall_openings.py 3d_viewer/tests/test_wall_openings.py
git commit -m "add wall opening persistence"
```

---

### Task 2: Add Main Viewer Recording

**Files:**
- Modify: `3d_viewer/ui/main_window.py`
- Create: `3d_viewer/tests/test_main_window_wall_openings.py`

- [ ] **Step 1: Write failing tests for the record button**

Create `3d_viewer/tests/test_main_window_wall_openings.py`:

```python
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from core.dataset import Dataset
from core.wall_openings import load_wall_openings
from ui.main_window import MainWindow


class MainWindowWallOpeningsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_record_current_opening_saves_last_highlight(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            dataset = Dataset(
                data_root=data_root,
                camera_file=data_root / "camera_pos.cam",
                image_dir=data_root,
                pointcloud_file=data_root / "cloud.las",
                poses=[],
                points=np.asarray([
                    [1.0, 0.0, 0.8],
                    [1.0, 0.4, 1.0],
                    [1.0, 0.8, 1.6],
                ], dtype=np.float64),
                colors=np.zeros((3, 3), dtype=np.uint8),
                total_points=3,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=0.0,
            )
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win.current_idx = -1
            win._last_opening_candidate = {
                "mask": np.asarray([True, True, True]),
                "label": "window",
                "seed_index": 1,
                "confidence": "high",
                "reason": "accepted_vertical_planar_region",
                "plane_point": np.asarray([1.0, 0.4, 1.0]),
                "plane_normal": np.asarray([1.0, 0.0, 0.0]),
                "width_m": 0.8,
                "height_m": 0.8,
                "detection_index": 2,
                "score": 0.8,
            }

            win._record_current_opening()

            openings = load_wall_openings(workspace, data_root)
            self.assertEqual(len(openings), 1)
            self.assertEqual(openings[0].label, "window")
            self.assertEqual(openings[0].source_image, "")

    def test_clear_highlight_clears_opening_candidate(self):
        with patch.object(MainWindow, "_load", lambda self: None):
            win = MainWindow(ROOT.parent)
        self.addCleanup(win.close)
        win._last_opening_candidate = {"label": "window"}

        win._clear_highlight()

        self.assertIsNone(win._last_opening_candidate)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_main_window_wall_openings.py
```

Expected: fails because `_last_opening_candidate` and `_record_current_opening` are not implemented.

- [ ] **Step 3: Modify imports and state in `MainWindow`**

In `3d_viewer/ui/main_window.py`, add imports:

```python
from core.wall_openings import append_wall_opening, opening_from_selection
```

In `MainWindow.__init__`, after `_highlight_mask`:

```python
self._last_opening_candidate: dict | None = None
```

Add button setup near `btn_clear_highlight`:

```python
self.btn_record_opening = QPushButton("记录当前门窗")
self.btn_record_opening.clicked.connect(self._record_current_opening)
```

Add it to `_build_ui` after `self.btn_clear_highlight`:

```python
v.addWidget(self.btn_record_opening)
```

- [ ] **Step 4: Track the last extraction candidate**

In `_clear_highlight`, add:

```python
self._last_opening_candidate = None
```

In `_on_select`, after `_set_highlight_mask(None)`, add:

```python
self._last_opening_candidate = None
```

In `_run_pointcloud_extraction`, after a fused selection has been evaluated and before `return`, add:

```python
self._last_opening_candidate = self._opening_candidate_from_selection(idx, fused, highlight)
```

In the pure point-cloud branch after `highlight` is computed, add:

```python
self._last_opening_candidate = self._opening_candidate_from_selection(idx, selection, highlight)
```

Add helper:

```python
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
```

- [ ] **Step 5: Implement `_record_current_opening`**

Add to `MainWindow`:

```python
def _record_current_opening(self):
    if self.dataset is None or self._last_opening_candidate is None:
        self.statusBar().showMessage("请先用点云门窗提取选中一个门窗区域")
        return
    source_image = ""
    if self.current_idx >= 0 and self.current_idx < len(self.dataset.poses):
        source_image = self.dataset.poses[self.current_idx].image_name
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
    except Exception as exc:
        QMessageBox.warning(self, "门窗记录失败", str(exc))
        return
    self.statusBar().showMessage(f"已记录门窗: {saved.id} ({saved.label})")
```

- [ ] **Step 6: Run Task 2 tests**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_main_window_wall_openings.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add 3d_viewer/ui/main_window.py 3d_viewer/tests/test_main_window_wall_openings.py
git commit -m "record extracted wall openings"
```

---

### Task 3: Match Openings To Wall Segments

**Files:**
- Modify: `3d_viewer/core/wall_model.py`
- Modify: `3d_viewer/tests/test_wall_model.py`

- [ ] **Step 1: Write failing matching tests**

Append to `WallModelTest` in `3d_viewer/tests/test_wall_model.py`:

```python
    def test_matches_wall_opening_to_vertical_segment(self):
        from core.wall_openings import WallOpening
        from core.wall_model import WallSegment, match_wall_openings_to_segments

        wall = WallSegment("vertical", 1.0, 0.0, 1.0, 3.0, 0.0, 2.6, 3.0, 100, 2.6)
        opening = WallOpening(
            id="window-0001",
            label="window",
            source_image="608.jpg",
            seed_index=1,
            point_count=20,
            confidence="high",
            reason="accepted_vertical_planar_region",
            center=(1.05, 1.2, 1.2),
            normal=(1.0, 0.0, 0.0),
            bbox_min=(1.02, 0.8, 0.9),
            bbox_max=(1.08, 1.6, 1.5),
            width_m=0.8,
            height_m=0.6,
            z_min=0.9,
            z_max=1.5,
            detection_index=-1,
            score=None,
        )

        matched, unmatched = match_wall_openings_to_segments([opening], [wall])

        self.assertEqual(len(matched), 1)
        self.assertEqual(unmatched, [])
        self.assertEqual(matched[0].opening_id, "window-0001")
        self.assertEqual(matched[0].segment_index, 0)
        self.assertEqual(matched[0].axis_min, 0.8)
        self.assertEqual(matched[0].axis_max, 1.6)

    def test_rejects_opening_far_from_wall_segment(self):
        from core.wall_openings import WallOpening
        from core.wall_model import WallSegment, match_wall_openings_to_segments

        wall = WallSegment("horizontal", 0.0, 0.0, 3.0, 0.0, 0.0, 2.6, 3.0, 100, 2.6)
        opening = WallOpening(
            id="door-0001",
            label="door",
            source_image="608.jpg",
            seed_index=1,
            point_count=20,
            confidence="high",
            reason="accepted_vertical_planar_region",
            center=(1.0, 2.0, 1.0),
            normal=(0.0, 1.0, 0.0),
            bbox_min=(0.6, 1.9, 0.0),
            bbox_max=(1.4, 2.1, 2.0),
            width_m=0.8,
            height_m=2.0,
            z_min=0.0,
            z_max=2.0,
            detection_index=-1,
            score=None,
        )

        matched, unmatched = match_wall_openings_to_segments([opening], [wall], max_plane_distance_m=0.25)

        self.assertEqual(matched, [])
        self.assertEqual(unmatched[0]["id"], "door-0001")
        self.assertEqual(unmatched[0]["reason"], "no_matching_wall_segment")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_model.py
```

Expected: fails because `match_wall_openings_to_segments` does not exist.

- [ ] **Step 3: Add matching dataclass and function**

In `3d_viewer/core/wall_model.py`, import:

```python
from core.wall_openings import WallOpening
```

Add near `WallModelResult`:

```python
@dataclass(frozen=True)
class WallOpeningMarker:
    opening_id: str
    label: str
    segment_index: int
    orientation: str
    wall_coord: float
    axis_min: float
    axis_max: float
    z_min: float
    z_max: float
    side: float
```

Add:

```python
def match_wall_openings_to_segments(
    openings: Iterable[WallOpening],
    segments: list[WallSegment],
    *,
    max_plane_distance_m: float = 0.35,
    span_tolerance_m: float = 0.25,
) -> tuple[list[WallOpeningMarker], list[dict]]:
    matched: list[WallOpeningMarker] = []
    unmatched: list[dict] = []
    for opening in openings:
        best = None
        best_distance = float("inf")
        center = np.asarray(opening.center, dtype=np.float64)
        bbox_min = np.asarray(opening.bbox_min, dtype=np.float64)
        bbox_max = np.asarray(opening.bbox_max, dtype=np.float64)
        for index, seg in enumerate(segments):
            if seg.orientation == "vertical":
                axis_min, axis_max = sorted((float(bbox_min[1]), float(bbox_max[1])))
                seg_min, seg_max = sorted((seg.y1, seg.y2))
                distance = abs(float(center[0]) - seg.x1)
                side = 1.0 if float(center[0]) >= seg.x1 else -1.0
                wall_coord = seg.x1
            else:
                axis_min, axis_max = sorted((float(bbox_min[0]), float(bbox_max[0])))
                seg_min, seg_max = sorted((seg.x1, seg.x2))
                distance = abs(float(center[1]) - seg.y1)
                side = 1.0 if float(center[1]) >= seg.y1 else -1.0
                wall_coord = seg.y1
            if distance > max_plane_distance_m:
                continue
            if axis_max < seg_min - span_tolerance_m or axis_min > seg_max + span_tolerance_m:
                continue
            if distance < best_distance:
                best_distance = distance
                best = (index, seg, wall_coord, max(seg_min, axis_min), min(seg_max, axis_max), side)
        if best is None:
            unmatched.append({"id": opening.id, "label": opening.label, "reason": "no_matching_wall_segment"})
            continue
        index, seg, wall_coord, axis_min, axis_max, side = best
        if axis_max <= axis_min:
            unmatched.append({"id": opening.id, "label": opening.label, "reason": "empty_projected_opening_span"})
            continue
        matched.append(
            WallOpeningMarker(
                opening_id=opening.id,
                label=opening.label,
                segment_index=index,
                orientation=seg.orientation,
                wall_coord=float(wall_coord),
                axis_min=round(float(axis_min), 5),
                axis_max=round(float(axis_max), 5),
                z_min=round(max(float(opening.z_min), float(seg.z_min)), 5),
                z_max=round(min(float(opening.z_max), float(seg.z_max)), 5),
                side=float(side),
            )
        )
    return matched, unmatched
```

- [ ] **Step 4: Run Task 3 tests**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_model.py
```

Expected: all wall model tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add 3d_viewer/core/wall_model.py 3d_viewer/tests/test_wall_model.py
git commit -m "match openings to wall segments"
```

---

### Task 4: Draw Top-Down Opening Markers

**Files:**
- Modify: `3d_viewer/core/wall_model.py`
- Modify: `3d_viewer/tests/test_wall_model.py`

- [ ] **Step 1: Write failing top-down marker test**

Append to `WallModelTest`:

```python
    def test_topdown_preview_draws_opening_marker(self):
        from core.wall_model import WallOpeningMarker, WallSegment, render_wall_model_topdown_preview

        grid = np.zeros((80, 80), dtype=np.uint8)
        wall_mask = np.zeros_like(grid, dtype=bool)
        segment = WallSegment("vertical", 1.0, 0.0, 1.0, 3.0, 0.0, 2.6, 3.0, 100, 2.6)
        marker = WallOpeningMarker(
            opening_id="window-0001",
            label="window",
            segment_index=0,
            orientation="vertical",
            wall_coord=1.0,
            axis_min=1.0,
            axis_max=1.5,
            z_min=0.9,
            z_max=1.5,
            side=1.0,
        )

        image = render_wall_model_topdown_preview(
            grid,
            wall_mask,
            [segment],
            x_min=0.0,
            y_min=0.0,
            resolution_m=0.05,
            opening_markers=[marker],
        )

        pixels = np.asarray(image)
        green_pixels = (pixels[:, :, 1] > 180) & (pixels[:, :, 0] < 80)
        self.assertTrue(green_pixels.any())
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_model.py
```

Expected: fails because `render_wall_model_topdown_preview` has no `opening_markers` parameter.

- [ ] **Step 3: Extend top-down renderer**

Change signature:

```python
def render_wall_model_topdown_preview(
    grid: np.ndarray,
    wall_mask: np.ndarray,
    segments: list[WallSegment],
    x_min: float,
    y_min: float,
    resolution_m: float,
    *,
    opening_markers: list[WallOpeningMarker] | None = None,
) -> Image.Image:
```

After drawing red wall lines, add:

```python
    for marker in opening_markers or []:
        color = (40, 230, 110) if marker.label == "window" else (255, 205, 40)
        if marker.orientation == "vertical":
            x = (marker.wall_coord - x_min) / resolution_m
            y1 = height - 1 - (marker.axis_min - y_min) / resolution_m
            y2 = height - 1 - (marker.axis_max - y_min) / resolution_m
            draw.line((x, y1, x, y2), fill=color, width=7)
            draw.line((x, y1, x, y2), fill=(10, 20, 15), width=2)
        else:
            x1 = (marker.axis_min - x_min) / resolution_m
            x2 = (marker.axis_max - x_min) / resolution_m
            y = height - 1 - (marker.wall_coord - y_min) / resolution_m
            draw.line((x1, y, x2, y), fill=color, width=7)
            draw.line((x1, y, x2, y), fill=(10, 20, 15), width=2)
```

- [ ] **Step 4: Run Task 4 tests**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_model.py
```

Expected: all wall model tests pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add 3d_viewer/core/wall_model.py 3d_viewer/tests/test_wall_model.py
git commit -m "draw opening markers in topdown preview"
```

---

### Task 5: Add OBJ Marker-Frame Geometry

**Files:**
- Modify: `3d_viewer/core/wall_model.py`
- Modify: `3d_viewer/tests/test_wall_model.py`

- [ ] **Step 1: Write failing OBJ marker mesh test**

Append to `WallModelTest`:

```python
    def test_opening_marker_frames_create_extra_mesh(self):
        from core.wall_model import WallOpeningMarker, opening_markers_to_mesh

        marker = WallOpeningMarker(
            opening_id="door-0001",
            label="door",
            segment_index=0,
            orientation="horizontal",
            wall_coord=2.0,
            axis_min=0.5,
            axis_max=1.3,
            z_min=0.0,
            z_max=2.1,
            side=1.0,
        )

        vertices, faces = opening_markers_to_mesh([marker], wall_thickness_m=0.08)

        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)
        self.assertEqual(len(vertices) % 8, 0)
        self.assertEqual(len(faces) % 6, 0)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_model.py
```

Expected: fails because `opening_markers_to_mesh` does not exist.

- [ ] **Step 3: Implement marker-frame mesh helpers**

Add to `3d_viewer/core/wall_model.py`:

```python
def opening_markers_to_mesh(
    markers: Iterable[WallOpeningMarker],
    *,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
    frame_width_m: float = 0.04,
    marker_depth_m: float = 0.025,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for marker in markers:
        axis_min, axis_max = sorted((marker.axis_min, marker.axis_max))
        z_min, z_max = sorted((marker.z_min, marker.z_max))
        if axis_max <= axis_min or z_max <= z_min:
            continue
        fw = min(frame_width_m, (axis_max - axis_min) / 2.0, (z_max - z_min) / 2.0)
        strips = [
            (axis_min, axis_max, z_min, z_min + fw),
            (axis_min, axis_max, z_max - fw, z_max),
            (axis_min, axis_min + fw, z_min, z_max),
            (axis_max - fw, axis_max, z_min, z_max),
        ]
        for a1, a2, z1, z2 in strips:
            _append_marker_strip_box(
                vertices,
                faces,
                marker,
                a1,
                a2,
                z1,
                z2,
                wall_thickness_m=wall_thickness_m,
                marker_depth_m=marker_depth_m,
            )
    return vertices, faces


def _append_marker_strip_box(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int, int]],
    marker: WallOpeningMarker,
    axis_min: float,
    axis_max: float,
    z_min: float,
    z_max: float,
    *,
    wall_thickness_m: float,
    marker_depth_m: float,
) -> None:
    face_offset = marker.side * (wall_thickness_m / 2.0 + 0.01)
    depth = marker.side * marker_depth_m
    if marker.orientation == "vertical":
        x1 = marker.wall_coord + face_offset
        x2 = marker.wall_coord + face_offset + depth
        y1, y2 = axis_min, axis_max
    else:
        x1, x2 = axis_min, axis_max
        y1 = marker.wall_coord + face_offset
        y2 = marker.wall_coord + face_offset + depth
    base = len(vertices) + 1
    vertices.extend([
        (x1, y1, z_min), (x2, y1, z_min), (x2, y2, z_min), (x1, y2, z_min),
        (x1, y1, z_max), (x2, y1, z_max), (x2, y2, z_max), (x1, y2, z_max),
    ])
    faces.extend([
        (base + 0, base + 1, base + 2, base + 3),
        (base + 4, base + 7, base + 6, base + 5),
        (base + 0, base + 4, base + 5, base + 1),
        (base + 1, base + 5, base + 6, base + 2),
        (base + 2, base + 6, base + 7, base + 3),
        (base + 3, base + 7, base + 4, base + 0),
    ])
```

- [ ] **Step 4: Run Task 5 tests**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_model.py
```

Expected: all wall model tests pass.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add 3d_viewer/core/wall_model.py 3d_viewer/tests/test_wall_model.py
git commit -m "add opening marker frames to obj mesh"
```

---

### Task 6: Wire Openings Into Wall Generation

**Files:**
- Modify: `3d_viewer/core/wall_model.py`
- Modify: `3d_viewer/wall_model_tool.py`
- Modify: `3d_viewer/generate_wall_model.py`
- Modify: `3d_viewer/tests/test_wall_model.py`

- [ ] **Step 1: Write failing integration test**

Append to `WallModelTest`:

```python
    def test_generate_wall_model_includes_opening_metadata(self):
        from core.wall_openings import WallOpening
        from core.wall_model import generate_wall_model

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            xs = np.linspace(0.0, 4.0, 45)
            ys = np.linspace(0.0, 3.0, 35)
            zs = np.linspace(0.0, 2.4, 12)
            points = []
            for z in zs:
                for x in xs:
                    points.append([x, 0.0, z])
                    points.append([x, 3.0, z])
                for y in ys:
                    points.append([0.0, y, z])
                    points.append([4.0, y, z])
            points = np.asarray(points, dtype=np.float64)
            opening = WallOpening(
                id="window-0001",
                label="window",
                source_image="608.jpg",
                seed_index=1,
                point_count=20,
                confidence="high",
                reason="accepted_vertical_planar_region",
                center=(0.02, 2.0, 1.2),
                normal=(1.0, 0.0, 0.0),
                bbox_min=(0.0, 1.6, 0.9),
                bbox_max=(0.08, 2.4, 1.5),
                width_m=0.8,
                height_m=0.6,
                z_min=0.9,
                z_max=1.5,
                detection_index=-1,
                score=None,
            )

            result = generate_wall_model(
                workspace,
                data_root,
                points,
                openings=[opening],
                resolution_m=0.1,
            )

            payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["matched_opening_count"], 1)
            self.assertEqual(payload["unmatched_opening_count"], 0)
            self.assertEqual(payload["opening_markers"][0]["opening_id"], "window-0001")
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_model.py
```

Expected: fails because `generate_wall_model` has no `openings` parameter.

- [ ] **Step 3: Extend `WallModelResult` and `generate_wall_model`**

Change `WallModelResult`:

```python
@dataclass(frozen=True)
class WallModelResult:
    obj_path: Path
    metadata_path: Path
    preview_path: Path
    topdown_preview_path: Path
    segment_count: int
    vertex_count: int
    face_count: int
    matched_opening_count: int = 0
    unmatched_opening_count: int = 0
```

Change `generate_wall_model` signature:

```python
def generate_wall_model(
    workspace: Path,
    data_root: Path,
    points: np.ndarray,
    *,
    openings: Iterable[WallOpening] | None = None,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
    min_wall_length_m: float = DEFAULT_MIN_MODEL_WALL_LENGTH_M,
    strip_width_m: float = DEFAULT_STRIP_WIDTH_M,
) -> WallModelResult:
```

After final `segments = snap_wall_segment_endpoints(...)`, add:

```python
    opening_markers, unmatched_openings = match_wall_openings_to_segments(
        list(openings or []),
        segments,
    )
```

Change mesh generation:

```python
    vertices, faces = wall_segments_to_mesh(segments, wall_thickness_m=wall_thickness_m)
    marker_vertices, marker_faces = opening_markers_to_mesh(
        opening_markers,
        wall_thickness_m=wall_thickness_m,
    )
    marker_base = len(vertices)
    vertices.extend(marker_vertices)
    faces.extend(tuple(index + marker_base for index in face) for face in marker_faces)
```

Pass markers into `_write_metadata` and `render_wall_model_topdown_preview`.

- [ ] **Step 4: Extend metadata writer**

Change `_write_metadata` signature to include:

```python
    opening_markers: list[WallOpeningMarker] | None = None,
    unmatched_openings: list[dict] | None = None,
```

Add fields:

```python
"matched_opening_count": len(opening_markers or []),
"unmatched_opening_count": len(unmatched_openings or []),
"opening_markers": [marker.__dict__ for marker in (opening_markers or [])],
"unmatched_openings": unmatched_openings or [],
```

- [ ] **Step 5: Load openings in workbench and CLI**

In `3d_viewer/wall_model_tool.py`, import:

```python
from core.wall_openings import load_wall_openings
```

In `generate_wall_artifacts`, before `generate_wall_model`:

```python
openings = load_wall_openings(self.workspace, self.dataset.data_root)
```

Pass:

```python
openings=openings,
```

In `_show_result`, include counts:

```python
f"门窗标记: {result.matched_opening_count} matched · {result.unmatched_opening_count} unmatched\n"
```

In `3d_viewer/generate_wall_model.py`, load openings after loading the dataset and pass them into `generate_wall_model`.

- [ ] **Step 6: Run Task 6 tests**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_model.py /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_wall_model_tool.py /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests/test_generate_wall_model_cli.py
```

Expected: all listed tests pass.

- [ ] **Step 7: Commit Task 6**

Run:

```bash
git add 3d_viewer/core/wall_model.py 3d_viewer/wall_model_tool.py 3d_viewer/generate_wall_model.py 3d_viewer/tests/test_wall_model.py
git commit -m "include recorded openings in wall models"
```

---

### Task 7: Full Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run complete test suite**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python -m unittest discover -s /Users/gengchen/Desktop/3dtiqu/3d_viewer/tests
```

Expected: all tests pass.

- [ ] **Step 2: Run generator manually if dataset is available**

Run:

```bash
/Users/gengchen/Desktop/3dtiqu/.venv/bin/python /Users/gengchen/Desktop/3dtiqu/3d_viewer/generate_wall_model.py --workspace /Users/gengchen/Desktop/3dtiqu
```

Expected: generator completes and reports wall segments plus matched/unmatched opening counts.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status -sb
```

Expected: only `.codegraph/` remains untracked, or no unexpected files are present.

- [ ] **Step 4: Push branch**

Run:

```bash
git push
```

Expected: current branch updates the existing GitHub PR.
