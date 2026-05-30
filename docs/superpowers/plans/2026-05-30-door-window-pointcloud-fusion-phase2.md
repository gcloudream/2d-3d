# Door/Window Point Cloud Fusion Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic 3D geometry scoring so bbox-refined door/window selections can reject obvious non-door/window objects.

**Architecture:** Extend `core/door_window_refine.py` after the existing Phase 1 component/depth filtering stage. Fit a local plane to the refined points, estimate planar width/height, score against label-aware door/window thresholds, and return plane/size metadata plus a clearer confidence and reason.

**Tech Stack:** Python 3.12, NumPy, existing `unittest` suite.

---

## Scope

Included:

- Least-squares plane fitting for refined sampled points.
- Plane verticality check using the project Z axis.
- PCA-based planar width/height estimation.
- Label-aware dimension scoring for `door` and `window`.
- Unit tests for accepting plausible window geometry and rejecting implausible non-window geometry.
- Existing UI reuse through `confidence` and `reason` fields.

Excluded:

- Full RANSAC iteration tuning.
- Full LAS export.
- Global pure point-cloud door/window search outside a clicked bbox.

## File Structure

- Modify `3d_viewer/core/door_window_refine.py`
  - Add `PlaneFit` and `GeometryScore` dataclasses.
  - Add `fit_plane_least_squares(points)`.
  - Add `estimate_planar_extent(points, normal)`.
  - Add `score_door_window_geometry(points, label)`.
  - Call geometry scoring from `refine_detection_selection`.

- Modify `3d_viewer/tests/test_door_window_refine.py`
  - Add tests that drive the new public helpers and integrated selection behavior.

## Important Worktree Note

The repository still has unrelated uncommitted alignment-calibration changes. Do not stage those as part of Phase 2.

---

### Task 1: Add Geometry Scoring Tests

**Files:**
- Modify: `3d_viewer/tests/test_door_window_refine.py`
- Later implementation target: `3d_viewer/core/door_window_refine.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `fit_plane_least_squares`, `score_door_window_geometry`, and verify:

```python
def test_geometry_score_accepts_plausible_vertical_window_patch(self):
    points = make_patch(width=1.2, height=0.9, z0=1.0)
    score = score_door_window_geometry(points, "window")
    self.assertEqual(score.confidence, "high")
    self.assertEqual(score.reason, "accepted_vertical_window_geometry")
```

```python
def test_geometry_score_rejects_window_patch_that_is_too_small(self):
    points = make_patch(width=0.12, height=0.12, z0=1.0)
    score = score_door_window_geometry(points, "window")
    self.assertEqual(score.confidence, "low")
    self.assertEqual(score.reason, "rejected_implausible_window_size")
```

```python
def test_refine_selection_uses_geometry_reason_for_plausible_window(self):
    selection = refine_detection_selection(...)
    self.assertEqual(selection.confidence, "high")
    self.assertEqual(selection.reason, "accepted_vertical_window_geometry")
    self.assertIsNotNone(selection.plane_normal)
    self.assertGreater(selection.width_m, 1.0)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_door_window_refine
```

Expected: fail because the geometry helpers and new fields do not exist.

### Task 2: Implement Geometry Scoring

**Files:**
- Modify: `3d_viewer/core/door_window_refine.py`
- Test: `3d_viewer/tests/test_door_window_refine.py`

- [ ] **Step 1: Add dataclasses and constants**

Add constants for minimum plane points, verticality threshold, and label-aware dimensions.

- [ ] **Step 2: Implement plane fitting**

Use centroid + SVD to compute the least-squares plane normal and RMS residual.

- [ ] **Step 3: Implement planar extent estimation**

Project points onto two plane basis axes, force one basis axis to align with world Z when possible, and report width/height.

- [ ] **Step 4: Implement scoring**

Reject if too few points, not planar enough, not vertical enough, or outside label-aware size thresholds. Accept plausible patches with `high` confidence.

- [ ] **Step 5: Integrate into selection**

After Phase 1 depth filtering, call geometry scoring. Use the geometry confidence/reason when enough points exist.

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_door_window_refine
```

Expected: all tests pass.

### Task 3: Validate Phase 2

**Files:**
- Existing tests and validation scripts.

- [ ] **Step 1: Run full unit suite**

```bash
.venv/bin/python -m unittest discover -s 3d_viewer/tests
```

- [ ] **Step 2: Run smoke test**

```bash
.venv/bin/python 3d_viewer/_smoke_test.py
```

- [ ] **Step 3: Run real-data refinement validator**

```bash
.venv/bin/python 3d_viewer/validate_door_window_refine.py --detections out/door_window_annotations/608.675997_IMG.json --max-points 120000
```

- [ ] **Step 4: Commit only Phase 2 changes**

Stage the Phase 2 plan, `door_window_refine.py`, and `test_door_window_refine.py`; do not stage unrelated alignment changes.
