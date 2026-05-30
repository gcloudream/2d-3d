# Dual Point Cloud View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a synchronized pure 3D point cloud pane below the existing panorama-overlap interaction pane.

**Architecture:** Keep `SceneView` as the rendering unit and instantiate it twice in `MainWindow`. Add a small UI sync helper so point-size, highlight-mask, and observer-scene setup can be unit tested without constructing OpenGL windows.

**Tech Stack:** Python 3.12, PySide6, existing ModernGL `SceneView`, `unittest`.

---

## Scope

Included:

- Add a second `SceneView` named `cloud_scene`.
- Place normal viewing mode inside a vertical splitter: top interaction scene, bottom pure 3D scene.
- Keep annotation editor switching through the existing `QStackedWidget`.
- Upload dataset points to both scenes.
- Sync keyframe changes, point size, highlight masks, and selection clearing.
- Configure the bottom scene as point-cloud-only: no panorama, no detection boxes, no pick mode.

Excluded:

- Full LAS export.
- Synchronized cameras between panes.
- Picking or detection editing inside the pure 3D pane.

## Files

- Create `3d_viewer/ui/scene_sync.py`
  - Owns testable scene-pair helper functions.

- Create `3d_viewer/tests/test_scene_sync.py`
  - Uses fake scene objects to verify helper calls.

- Modify `3d_viewer/ui/main_window.py`
  - Add second `SceneView`.
  - Add vertical splitter inside normal-mode stack page.
  - Route highlight clearing and point-size changes through helpers.

## Worktree Note

The worktree still has unrelated alignment-calibration changes in several files. Do not stage or revert those unless explicitly requested.

## Tasks

### Task 1: Add Sync Helper Tests

- [ ] Create `3d_viewer/tests/test_scene_sync.py`.
- [ ] Test that configuring the observer scene disables panorama, bbox, and picking while keeping point cloud visible.
- [ ] Test that setting point size updates both scenes.
- [ ] Test that setting highlight mask updates both scenes.
- [ ] Run `.venv/bin/python -m unittest 3d_viewer.tests.test_scene_sync` and verify it fails because `ui.scene_sync` does not exist.

### Task 2: Implement Sync Helper

- [ ] Create `3d_viewer/ui/scene_sync.py`.
- [ ] Implement `configure_observer_scene(scene)`.
- [ ] Implement `set_scene_pair_point_size(primary, observer, size)`.
- [ ] Implement `set_scene_pair_highlight_mask(primary, observer, mask)`.
- [ ] Run `.venv/bin/python -m unittest 3d_viewer.tests.test_scene_sync` and verify it passes.

### Task 3: Wire Dual Scene UI

- [ ] Modify `MainWindow.__init__` to create `self.cloud_scene = SceneView()`.
- [ ] Do not connect `cloud_scene.point_clicked`; the bottom view is observer-only.
- [ ] Modify `_build_ui` so normal mode uses `QSplitter(Qt.Vertical)` containing `self.scene` and `self.cloud_scene`.
- [ ] Keep `PanoAnnotationEditor` as the second page in `self.view_stack`.
- [ ] In `_load`, upload points/colors to both scenes and configure the observer scene.
- [ ] In `_on_select`, call `set_keyframe` on both scenes and clear both highlight masks.
- [ ] In `_on_point_clicked`, apply accepted highlight masks to both scenes.
- [ ] In detection mode changes, clear both scenes.
- [ ] Change the point-size slider to update both scenes.

### Task 4: Verification and Commit

- [ ] Run `.venv/bin/python -m unittest discover -s 3d_viewer/tests`.
- [ ] Run `.venv/bin/python 3d_viewer/_smoke_test.py`.
- [ ] Run `git diff --check`.
- [ ] Stage only the dual-view plan, sync helper, sync tests, and intended `main_window.py` hunks.
- [ ] Commit with `feat: add dual point cloud view`.
