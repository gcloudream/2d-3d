# Door/Window Point Cloud Fusion Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace bbox-only door/window point selection with seed-constrained 3D refinement using the clicked point, connected components, and depth filtering.

**Architecture:** Keep the existing 2D bbox matching as the coarse ROI stage, then add a focused `core/door_window_refine.py` module that refines that ROI using only sampled viewer points. The UI will call this module from the existing click path and highlight `refined_mask` instead of the raw bbox mask, while reporting coarse/refined counts and a confidence/reason string.

**Tech Stack:** Python 3.12, NumPy, SciPy `cKDTree`, existing PySide6/ModernGL viewer, `unittest`.

---

## Scope

This plan implements Phase 1 from `docs/superpowers/specs/2026-05-30-door-window-pointcloud-fusion-design.md`.

Included:

- New `RefinedDoorWindowSelection` result object.
- Existing bbox matching reused as the coarse ROI.
- Seed connected component inside the coarse ROI.
- Seed-relative depth filtering.
- UI integration for refined highlight and status text.
- Unit tests for success and failure behavior.

Excluded from Phase 1:

- Plane RANSAC.
- Door/window dimension scoring.
- Rectangularity scoring.
- Full LAS export.

## File Structure

- Create `3d_viewer/core/door_window_refine.py`
  - Owns refinement dataclass, connected-component helper, depth helper, and high-level refinement function.
  - Depends only on NumPy, SciPy, and `core.door_window`.

- Create `3d_viewer/tests/test_door_window_refine.py`
  - Synthetic tests for coarse hit matching, connected component selection, depth filtering, and failure reasons.

- Modify `3d_viewer/ui/main_window.py`
  - Replace direct `select_detection_region` use in `_on_point_clicked` with `refine_detection_selection`.
  - Keep current projection and detection loading behavior.

- Modify `3d_viewer/README.md`
  - Update door/window selection description to say bbox results are refined by 3D point connectivity and depth filtering.

## Important Worktree Note

At plan creation time, the worktree already contains uncommitted alignment-fix changes in:

- `3d_viewer/README.md`
- `3d_viewer/core/dataset.py`
- `3d_viewer/render/camera.py`
- `3d_viewer/render/scene_view.py`
- `3d_viewer/ui/main_window.py`
- `3d_viewer/validate_door_window_match.py`
- `3d_viewer/validate_projection.py`
- `3d_viewer/tests/test_camera.py`
- `3d_viewer/tests/test_dataset_calibration.py`

Do not revert those changes. Implementation should work on top of the current tree. Commits in this plan should stage only files touched for this Phase 1 feature unless the user explicitly asks to include the alignment fix.

---

### Task 1: Add Refinement Unit Tests

**Files:**
- Create: `3d_viewer/tests/test_door_window_refine.py`
- Later implementation target: `3d_viewer/core/door_window_refine.py`

- [ ] **Step 1: Write the failing test file**

Create `3d_viewer/tests/test_door_window_refine.py` with:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.door_window_refine import (
    connected_component_from_seed,
    filter_by_seed_depth,
    refine_detection_selection,
)


class DoorWindowRefineTest(unittest.TestCase):
    def test_connected_component_keeps_seed_component_inside_candidate_mask(self):
        points = np.array([
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [2.00, 0.00, 0.00],
            [2.05, 0.00, 0.00],
            [5.00, 0.00, 0.00],
        ], dtype=np.float64)
        candidate = np.array([True, True, True, True, True, False])

        mask = connected_component_from_seed(
            points=points,
            candidate_mask=candidate,
            seed_idx=1,
            radius=0.11,
        )

        self.assertEqual(mask.tolist(), [True, True, True, False, False, False])

    def test_depth_filter_keeps_points_near_seed_range(self):
        points = np.array([
            [1.00, 0.00, 0.00],
            [1.05, 0.00, 0.00],
            [1.20, 0.00, 0.00],
            [1.80, 0.00, 0.00],
        ], dtype=np.float64)
        component = np.array([True, True, True, True])
        cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        mask = filter_by_seed_depth(
            points=points,
            component_mask=component,
            seed_idx=0,
            cam_pos=cam_pos,
            max_delta=0.25,
        )

        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_refine_selection_returns_refined_component_for_clicked_bbox(self):
        points = np.array([
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [2.00, 0.00, 0.00],
            [2.05, 0.00, 0.00],
            [4.00, 0.00, 0.00],
        ], dtype=np.float64)
        uv = np.array([
            [12.0, 12.0],
            [13.0, 12.0],
            [14.0, 12.0],
            [15.0, 12.0],
            [16.0, 12.0],
            [90.0, 90.0],
        ], dtype=np.float64)
        detections = [{"label": "window", "score": 0.8, "bbox": [10.0, 10.0, 20.0, 20.0]}]

        selection = refine_detection_selection(
            points=points,
            uv=uv,
            clicked_idx=1,
            detections=detections,
            pano_w=100.0,
            cam_pos=np.zeros(3, dtype=np.float64),
            component_radius=0.11,
            depth_delta=0.30,
            min_refined_points=3,
        )

        self.assertEqual(selection.detection_index, 0)
        self.assertEqual(selection.label, "window")
        self.assertEqual(selection.confidence, "medium")
        self.assertEqual(selection.reason, "seed_component_depth_filtered")
        self.assertEqual(selection.coarse_count, 5)
        self.assertEqual(selection.point_count, 3)
        self.assertEqual(selection.coarse_mask.tolist(), [True, True, True, True, True, False])
        self.assertEqual(selection.refined_mask.tolist(), [True, True, True, False, False, False])

    def test_refine_selection_reports_no_bbox_hit(self):
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        uv = np.array([[80.0, 80.0]], dtype=np.float64)
        detections = [{"label": "door", "score": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]}]

        selection = refine_detection_selection(
            points=points,
            uv=uv,
            clicked_idx=0,
            detections=detections,
            pano_w=100.0,
            cam_pos=np.zeros(3, dtype=np.float64),
        )

        self.assertEqual(selection.detection_index, -1)
        self.assertEqual(selection.confidence, "none")
        self.assertEqual(selection.reason, "no_bbox_hit")
        self.assertEqual(selection.point_count, 0)
        self.assertEqual(selection.refined_mask.tolist(), [False])

    def test_refine_selection_reports_too_few_refined_points(self):
        points = np.array([
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [2.00, 0.00, 0.00],
        ], dtype=np.float64)
        uv = np.array([
            [12.0, 12.0],
            [13.0, 12.0],
            [14.0, 12.0],
        ], dtype=np.float64)
        detections = [{"label": "door", "score": 0.7, "bbox": [10.0, 10.0, 20.0, 20.0]}]

        selection = refine_detection_selection(
            points=points,
            uv=uv,
            clicked_idx=0,
            detections=detections,
            pano_w=100.0,
            cam_pos=np.zeros(3, dtype=np.float64),
            component_radius=0.11,
            depth_delta=0.20,
            min_refined_points=3,
        )

        self.assertEqual(selection.detection_index, 0)
        self.assertEqual(selection.confidence, "low")
        self.assertEqual(selection.reason, "too_few_refined_points")
        self.assertEqual(selection.point_count, 2)
        self.assertEqual(selection.refined_mask.tolist(), [True, True, False])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_door_window_refine
```

Expected: FAIL or ERROR because `core.door_window_refine` does not exist.

- [ ] **Step 3: Commit the failing tests only if following strict red-green with local commits disabled**

Do not commit this red state. Proceed to Task 2.

---

### Task 2: Implement Core Refinement Module

**Files:**
- Create: `3d_viewer/core/door_window_refine.py`
- Test: `3d_viewer/tests/test_door_window_refine.py`

- [ ] **Step 1: Create the module with dataclass and helpers**

Create `3d_viewer/core/door_window_refine.py`:

```python
"""3D point-cloud refinement for door/window bbox selections."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from core.door_window import match_points_to_detections


DEFAULT_COMPONENT_RADIUS = 0.22
DEFAULT_DEPTH_DELTA = 0.35
DEFAULT_MIN_REFINED_POINTS = 80


@dataclass(frozen=True)
class RefinedDoorWindowSelection:
    detection_index: int
    detection: dict | None
    label: str
    score: float | None
    coarse_mask: np.ndarray
    refined_mask: np.ndarray
    confidence: str
    point_count: int
    coarse_count: int
    reason: str


def _empty_result(n: int, reason: str) -> RefinedDoorWindowSelection:
    empty = np.zeros(n, dtype=bool)
    return RefinedDoorWindowSelection(
        detection_index=-1,
        detection=None,
        label="",
        score=None,
        coarse_mask=empty.copy(),
        refined_mask=empty,
        confidence="none",
        point_count=0,
        coarse_count=0,
        reason=reason,
    )


def connected_component_from_seed(
    points: np.ndarray,
    candidate_mask: np.ndarray,
    seed_idx: int,
    radius: float = DEFAULT_COMPONENT_RADIUS,
) -> np.ndarray:
    points64 = np.asarray(points, dtype=np.float64)
    candidate = np.asarray(candidate_mask, dtype=bool)
    if len(points64) != len(candidate):
        raise ValueError(f"points length {len(points64)} != candidate mask length {len(candidate)}")
    if seed_idx < 0 or seed_idx >= len(points64) or not candidate[seed_idx]:
        return np.zeros(len(points64), dtype=bool)

    candidate_indices = np.flatnonzero(candidate)
    local_index = {int(global_idx): i for i, global_idx in enumerate(candidate_indices)}
    tree = cKDTree(points64[candidate_indices])
    visited_local = np.zeros(len(candidate_indices), dtype=bool)
    start_local = local_index[int(seed_idx)]
    visited_local[start_local] = True
    queue: deque[int] = deque([start_local])

    while queue:
        cur = queue.popleft()
        for nxt in tree.query_ball_point(points64[candidate_indices[cur]], r=float(radius)):
            if visited_local[nxt]:
                continue
            visited_local[nxt] = True
            queue.append(int(nxt))

    result = np.zeros(len(points64), dtype=bool)
    result[candidate_indices[visited_local]] = True
    return result


def filter_by_seed_depth(
    points: np.ndarray,
    component_mask: np.ndarray,
    seed_idx: int,
    cam_pos: np.ndarray,
    max_delta: float = DEFAULT_DEPTH_DELTA,
) -> np.ndarray:
    points64 = np.asarray(points, dtype=np.float64)
    component = np.asarray(component_mask, dtype=bool)
    if len(points64) != len(component):
        raise ValueError(f"points length {len(points64)} != component mask length {len(component)}")
    if seed_idx < 0 or seed_idx >= len(points64) or not component[seed_idx]:
        return np.zeros(len(points64), dtype=bool)

    cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    distances = np.linalg.norm(points64 - cam.reshape(1, 3), axis=1)
    seed_depth = float(distances[seed_idx])
    depth_ok = np.abs(distances - seed_depth) <= float(max_delta)
    return component & depth_ok


def refine_detection_selection(
    points: np.ndarray,
    uv: np.ndarray,
    clicked_idx: int,
    detections: Sequence[dict],
    pano_w: float,
    cam_pos: np.ndarray,
    component_radius: float = DEFAULT_COMPONENT_RADIUS,
    depth_delta: float = DEFAULT_DEPTH_DELTA,
    min_refined_points: int = DEFAULT_MIN_REFINED_POINTS,
) -> RefinedDoorWindowSelection:
    points_arr = np.asarray(points)
    uv_arr = np.asarray(uv)
    n = len(points_arr)
    if len(uv_arr) != n:
        raise ValueError(f"uv length {len(uv_arr)} != points length {n}")
    if clicked_idx < 0 or clicked_idx >= n:
        return _empty_result(n, "clicked_idx_out_of_range")
    if not detections:
        return _empty_result(n, "no_detections")

    matches = match_points_to_detections(uv_arr, detections, pano_w)
    det_idx = int(matches.match_indices[clicked_idx])
    if det_idx < 0:
        return _empty_result(n, "no_bbox_hit")

    coarse_mask = matches.match_indices == det_idx
    component_mask = connected_component_from_seed(
        points_arr,
        coarse_mask,
        clicked_idx,
        radius=component_radius,
    )
    depth_mask = filter_by_seed_depth(
        points_arr,
        component_mask,
        clicked_idx,
        cam_pos=cam_pos,
        max_delta=depth_delta,
    )

    det = detections[det_idx]
    point_count = int(depth_mask.sum())
    coarse_count = int(coarse_mask.sum())
    confidence = "medium" if point_count >= int(min_refined_points) else "low"
    reason = "seed_component_depth_filtered" if confidence == "medium" else "too_few_refined_points"

    return RefinedDoorWindowSelection(
        detection_index=det_idx,
        detection=det,
        label=str(det.get("label", "")),
        score=float(det["score"]) if "score" in det else None,
        coarse_mask=coarse_mask,
        refined_mask=depth_mask,
        confidence=confidence,
        point_count=point_count,
        coarse_count=coarse_count,
        reason=reason,
    )
```

- [ ] **Step 2: Run the refinement tests**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_door_window_refine
```

Expected: `Ran 5 tests` and `OK`.

- [ ] **Step 3: Run existing door/window tests**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_door_window 3d_viewer.tests.test_projection
```

Expected: all tests pass.

- [ ] **Step 4: Commit core refinement**

Run:

```bash
git add 3d_viewer/core/door_window_refine.py 3d_viewer/tests/test_door_window_refine.py
git commit -m "feat: refine door window point selections"
```

Expected: commit includes only the new module and new tests.

---

### Task 3: Integrate Refined Selection Into the UI Click Flow

**Files:**
- Modify: `3d_viewer/ui/main_window.py`
- Test: `3d_viewer/tests/test_door_window_refine.py` plus full test suite.

- [ ] **Step 1: Update imports**

In `3d_viewer/ui/main_window.py`, change:

```python
from core.door_window import select_detection_region
```

to:

```python
from core.door_window_refine import refine_detection_selection
```

- [ ] **Step 2: Replace `_on_point_clicked` selection block**

Inside `_on_point_clicked`, replace the block:

```python
        selection = select_detection_region(
            clicked_idx=idx,
            uv=uv,
            detections=self.current_detections,
            pano_w=float(img_w),
        )
        if selection.detection_index < 0:
            self.scene.set_highlight_mask(None)
            self.scene.set_selected_detection(-1)
            self.lbl_detection.setText(
                f"点击点 #{idx}\n未落入 door/window bbox\nuv: ({uv[idx,0]:.1f}, {uv[idx,1]:.1f})"
            )
            return
        self.scene.set_highlight_mask(selection.mask)
        self.scene.set_selected_detection(selection.detection_index)
        score = f"{selection.score:.3f}" if selection.score is not None else "—"
        self.lbl_detection.setText(
            f"命中 #{selection.detection_index} {selection.label} score={score}\n"
            f"points: {selection.point_count:,}\n"
            f"uv: ({uv[idx,0]:.1f}, {uv[idx,1]:.1f})"
        )
```

with:

```python
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
```

- [ ] **Step 3: Run targeted tests**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_door_window_refine 3d_viewer.tests.test_door_window
```

Expected: all tests pass.

- [ ] **Step 4: Run the smoke test**

Run:

```bash
.venv/bin/python 3d_viewer/_smoke_test.py
```

Expected: prints `[OK] smoke test passed`.

- [ ] **Step 5: Commit UI integration**

Run:

```bash
git add 3d_viewer/ui/main_window.py
git commit -m "feat: use refined door window point selection"
```

Expected: commit includes only `3d_viewer/ui/main_window.py`.

---

### Task 4: Add a Real-Data Validation Script

**Files:**
- Create: `3d_viewer/validate_door_window_refine.py`
- Test indirectly with one real keyframe and existing JSON.

- [ ] **Step 1: Create validation script**

Create `3d_viewer/validate_door_window_refine.py`:

```python
"""Validate coarse-vs-refined door/window point selection on one keyframe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORKSPACE = HERE.parent

from core.dataset import CameraPose, find_default_dataset, load_dataset
from core.door_window_refine import refine_detection_selection
from core.projection import project_points_to_panorama, rotation_from_angle


def _find_pose_for_image(poses: list[CameraPose], image_name: str) -> CameraPose:
    for pose in poses:
        if pose.image_name == image_name:
            return pose
    raise RuntimeError(f"no pose found for {image_name}")


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate refined door/window point cloud selection")
    ap.add_argument("--detections", required=True, help="path to detection or manual annotation JSON")
    ap.add_argument("--clicked-index", type=int, default=None, help="sampled point index to use as seed")
    ap.add_argument("--max-points", type=int, default=300_000)
    ap.add_argument("--output", default=str(WORKSPACE / "out" / "door_window_refine"))
    args = ap.parse_args()

    det_path = Path(args.detections).expanduser().resolve()
    payload = json.loads(det_path.read_text(encoding="utf-8"))
    image_path = Path(payload["image"])
    detections = list(payload.get("detections", []))
    if not detections:
        raise RuntimeError(f"no detections in {det_path}")

    cfg = find_default_dataset(WORKSPACE)
    if cfg is None:
        raise RuntimeError(f"no dataset found under {WORKSPACE}")
    dataset = load_dataset(cfg, max_points=args.max_points)
    pose = _find_pose_for_image(dataset.poses, image_path.name)

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    R = rotation_from_angle(pose.roll, pose.pitch, pose.yaw)
    uv = project_points_to_panorama(
        dataset.points,
        pose.position,
        R,
        img_w,
        img_h,
        yaw_offset_deg=dataset.pano_yaw_offset_deg,
    )

    clicked_idx = args.clicked_index
    if clicked_idx is None:
        first_bbox = detections[0]["bbox"]
        x1, y1, x2, y2 = [float(v) for v in first_bbox]
        if x2 < x1:
            inside_x = (uv[:, 0] >= x1) | (uv[:, 0] <= x2)
        else:
            inside_x = (uv[:, 0] >= x1) & (uv[:, 0] <= x2)
        inside = inside_x & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        indices = np.flatnonzero(inside)
        if len(indices) == 0:
            raise RuntimeError("first detection contains no sampled points")
        clicked_idx = int(indices[len(indices) // 2])

    selection = refine_detection_selection(
        points=dataset.points,
        uv=uv,
        clicked_idx=clicked_idx,
        detections=detections,
        pano_w=float(img_w),
        cam_pos=pose.position,
    )

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    font = _load_font(28)
    small_font = _load_font(18)

    coarse_idx = np.flatnonzero(selection.coarse_mask)
    refined_idx = np.flatnonzero(selection.refined_mask)
    for idx in coarse_idx[:: max(1, len(coarse_idx) // 6000)]:
        u, v = uv[idx]
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(0, 210, 255))
    for idx in refined_idx[:: max(1, len(refined_idx) // 6000)]:
        u, v = uv[idx]
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 40, 40))

    draw.rectangle((18, 18, 1260, 142), fill=(0, 0, 0))
    draw.text((34, 30), image_path.name, fill=(255, 255, 255), font=font)
    draw.text(
        (34, 78),
        f"cyan=coarse red=refined clicked={clicked_idx} confidence={selection.confidence} "
        f"coarse={selection.coarse_count} refined={selection.point_count} reason={selection.reason}",
        fill=(255, 255, 255),
        font=small_font,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_refine_check.jpg"
    vis.save(out_path, quality=92)

    print(f"[image] {image_path.name}")
    print(f"[clicked] {clicked_idx}")
    print(f"[detection] {selection.detection_index} {selection.label}")
    print(f"[confidence] {selection.confidence}")
    print(f"[points] coarse={selection.coarse_count:,} refined={selection.point_count:,}")
    print(f"[reason] {selection.reason}")
    print(f"[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the script on a known annotation**

Run:

```bash
.venv/bin/python 3d_viewer/validate_door_window_refine.py \
  --detections out/door_window_annotations/608.675997_IMG.json \
  --max-points 120000
```

Expected:

- command exits `0`
- output prints coarse and refined counts
- image is written under `out/door_window_refine/`

- [ ] **Step 3: Open or inspect the output path**

Use the generated path from script output. The image should show cyan coarse points and red refined points. Red should be a subset of cyan near the clicked seed's connected surface.

- [ ] **Step 4: Commit validation script**

Run:

```bash
git add 3d_viewer/validate_door_window_refine.py
git commit -m "test: add door window refinement validation"
```

Expected: commit includes only the validation script.

---

### Task 5: Update Documentation

**Files:**
- Modify: `3d_viewer/README.md`

- [ ] **Step 1: Update the door/window selection bullet**

In `3d_viewer/README.md`, replace:

```markdown
- `门窗选择`：开启后点击点云点，如果该点投影落入 door/window 框，会批量高亮同一框内的点云点。
```

with:

```markdown
- `门窗选择`：开启后点击点云点，如果该点投影落入 door/window 框，先得到 bbox 粗候选点，再按点击点附近的 3D 连通性和深度一致性精修高亮区域。
```

- [ ] **Step 2: Add a short limitations note**

After the manual annotation paragraph, add:

```markdown
门窗点云提取分两层：全景 bbox 只提供粗范围，最终高亮会用点击点所在的 3D 连通区域和深度一致性做精修。玻璃、屏幕、黑色反光物体附近仍可能因为 LiDAR 点稀疏而低置信度，右侧状态栏会显示 coarse/refined 点数和原因。
```

- [ ] **Step 3: Run documentation diff**

Run:

```bash
git diff -- 3d_viewer/README.md
```

Expected: only the door/window selection description and limitations note changed for this task.

- [ ] **Step 4: Commit docs**

Run:

```bash
git add 3d_viewer/README.md
git commit -m "docs: describe refined door window selection"
```

Expected: commit includes only README changes for this feature. If README still contains unrelated alignment-fix edits from prior work, do not stage the whole file blindly; stage only the Phase 1 hunks interactively or postpone this docs task until the alignment changes are committed.

---

### Task 6: Final Verification

**Files:**
- Verify all files touched by Tasks 1-5.

- [ ] **Step 1: Run full 3D viewer unit tests**

Run:

```bash
.venv/bin/python -m unittest discover -s 3d_viewer/tests
```

Expected: all tests pass, including the new refinement tests.

- [ ] **Step 2: Run smoke test**

Run:

```bash
.venv/bin/python 3d_viewer/_smoke_test.py
```

Expected: output ends with `[OK] smoke test passed`.

- [ ] **Step 3: Run validation script**

Run:

```bash
.venv/bin/python 3d_viewer/validate_door_window_refine.py \
  --detections out/door_window_annotations/608.675997_IMG.json \
  --max-points 120000
```

Expected: output includes `[confidence]`, a `[points] coarse=12 refined=5` style count line, `[reason]`, and `[out]`.

- [ ] **Step 4: Check diff hygiene**

Run:

```bash
git status --short
git diff --check
```

Expected:

- no whitespace errors from `git diff --check`
- only intentional files are modified or committed
- pre-existing alignment-fix changes are not accidentally bundled into Phase 1 commits

- [ ] **Step 5: Report results**

Report:

- unit test result
- smoke test result
- validation script output path
- whether refined points are fewer than or equal to coarse points
- any known limitation observed in the validation image

Do not claim the feature is complete unless all commands above were run and read.
