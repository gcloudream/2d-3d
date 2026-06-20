# Supplement Opening Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "记录当前门窗" save the full accumulated highlight when 补充模式 is used, so the saved opening spatial bbox matches the door/window area visible on screen.

**Architecture:** The UI already stores the displayed accumulated highlight in `MainWindow._highlight_mask`; the bug is that `_last_opening_candidate["mask"]` is built from only the latest extraction's `highlight`. Refactor the extraction path so `_apply_extraction_highlight()` returns the effective displayed mask, then build the save candidate from that effective mask in supplement mode. Keep existing wall model matching unchanged because it should consume the corrected saved opening bbox.

**Tech Stack:** Python, PySide6 UI, NumPy boolean masks, unittest.

---

## Root Cause

Current code path in `3d_viewer/ui/main_window.py`:

- `_apply_extraction_highlight(new_mask)` unions `new_mask` into `self._highlight_mask` when `self.cb_supplement.isChecked()` is true.
- `_run_pointcloud_extraction()` then calls `_opening_candidate_from_selection(idx, selection, highlight)` with the original single-click `highlight`, not the unioned `self._highlight_mask`.
- `_record_current_opening()` calls `opening_from_selection(self.dataset.points, self._last_opening_candidate["mask"], ...)`, so it saves only the last successful click's mask.

This explains the observed behavior: the screen can show a large accumulated door/window highlight, while `out/wall_openings/*_openings.json` contains a small low-height region from the last click. The wall model then correctly fails to mark the expected window location because it only sees the saved small region.

## File Structure

- Modify `3d_viewer/ui/main_window.py`
  - Return the effective display mask from `_apply_extraction_highlight()`.
  - Build `_last_opening_candidate` from the accumulated mask in supplement mode.
  - Preserve the previous candidate on failed supplement clicks.
  - Add log fields that distinguish latest extraction mask from record mask.
- Modify `3d_viewer/tests/test_main_window_wall_openings.py`
  - Add focused tests for supplement accumulation and failed supplement clicks.
  - Keep existing tests for direct save and logging.
- No wall model algorithm changes are required.

---

### Task 1: Add Regression Test For Supplement Saving The Accumulated Mask

**Files:**
- Modify: `3d_viewer/tests/test_main_window_wall_openings.py`

- [ ] **Step 1: Write the failing test**

Add this test method to `MainWindowWallOpeningsTest`:

```python
    def test_record_current_opening_uses_accumulated_highlight_in_supplement_mode(self):
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
                    [1.0, 1.2, 1.8],
                ], dtype=np.float64),
                colors=np.zeros((4, 3), dtype=np.uint8),
                total_points=4,
                sample_step=1,
                pano_calibration=None,
                pano_yaw_offset_deg=0.0,
            )
            with patch.object(MainWindow, "_load", lambda self: None):
                win = MainWindow(workspace)
            self.addCleanup(win.close)
            win.dataset = dataset
            win.current_idx = -1
            win.cb_supplement.setChecked(True)
            win._set_highlight_mask(np.asarray([True, True, False, False]))
            first_selection = SimpleNamespace(
                label="window",
                confidence="medium",
                reason="seed_component_depth_filtered",
                point_count=2,
                detection_index=4,
                score=1.0,
                width_m=0.4,
                height_m=0.7,
                source="fused",
                plane_point=np.asarray([1.0, 0.0, 1.0]),
                plane_normal=np.asarray([1.0, 0.0, 0.0]),
            )
            win._last_opening_candidate = win._opening_candidate_from_selection(
                10,
                first_selection,
                np.asarray([True, True, False, False]),
            )
            second_selection = SimpleNamespace(
                label="window",
                confidence="medium",
                reason="seed_component_depth_filtered",
                point_count=2,
                detection_index=4,
                score=1.0,
                width_m=0.4,
                height_m=0.7,
                source="fused",
                plane_point=np.asarray([1.0, 0.8, 1.6]),
                plane_normal=np.asarray([1.0, 0.0, 0.0]),
            )

            effective = win._apply_extraction_highlight(np.asarray([False, False, True, True]))
            win._last_opening_candidate = win._opening_candidate_from_selection(11, second_selection, effective)
            win._record_current_opening()

            openings = load_wall_openings(workspace, data_root)
            self.assertEqual(len(openings), 1)
            self.assertEqual(openings[0].point_count, 4)
            self.assertEqual(openings[0].bbox_min, (1.0, 0.0, 0.8))
            self.assertEqual(openings[0].bbox_max, (1.0, 1.2, 1.8))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_main_window_wall_openings.MainWindowWallOpeningsTest.test_record_current_opening_uses_accumulated_highlight_in_supplement_mode
```

Expected: FAIL because `_apply_extraction_highlight()` currently returns `None`, so the candidate cannot use the accumulated mask.

---

### Task 2: Return The Effective Mask From Highlight Application

**Files:**
- Modify: `3d_viewer/ui/main_window.py`

- [ ] **Step 1: Update `_apply_extraction_highlight()`**

Replace `_apply_extraction_highlight()` with:

```python
    def _apply_extraction_highlight(self, new_mask):
        """Set or accumulate the extraction highlight depending on 补充 mode."""
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
```

- [ ] **Step 2: Run the new test**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_main_window_wall_openings.MainWindowWallOpeningsTest.test_record_current_opening_uses_accumulated_highlight_in_supplement_mode
```

Expected: PASS if the direct helper-path test now receives the accumulated mask.

---

### Task 3: Use The Effective Mask In Real Extraction Paths

**Files:**
- Modify: `3d_viewer/ui/main_window.py`
- Test: `3d_viewer/tests/test_main_window_wall_openings.py`

- [ ] **Step 1: Add a helper to update the save candidate**

Add this method near `_opening_candidate_from_selection()`:

```python
    def _set_opening_candidate_from_extraction(self, seed_idx: int, selection, latest_highlight):
        effective_highlight = self._apply_extraction_highlight(latest_highlight)
        if latest_highlight is None and self.cb_supplement.isChecked():
            return effective_highlight
        self._last_opening_candidate = self._opening_candidate_from_selection(
            seed_idx,
            selection,
            effective_highlight,
        )
        return effective_highlight
```

- [ ] **Step 2: Replace fused path update**

In `_run_pointcloud_extraction()`, replace:

```python
            self._apply_extraction_highlight(highlight)
            self._last_opening_candidate = self._opening_candidate_from_selection(idx, fused, highlight)
            self._log_opening_selection_event("extract_opening_candidate", idx, fused, highlight)
```

with:

```python
            effective_highlight = self._set_opening_candidate_from_extraction(idx, fused, highlight)
            self._log_opening_selection_event("extract_opening_candidate", idx, fused, effective_highlight)
```

- [ ] **Step 3: Replace pure path update**

In `_run_pointcloud_extraction()`, replace:

```python
        self._apply_extraction_highlight(highlight)
        self._last_opening_candidate = self._opening_candidate_from_selection(idx, selection, highlight)
        self._log_opening_selection_event("extract_opening_candidate", idx, selection, highlight)
```

with:

```python
        effective_highlight = self._set_opening_candidate_from_extraction(idx, selection, highlight)
        self._log_opening_selection_event("extract_opening_candidate", idx, selection, effective_highlight)
```

- [ ] **Step 4: Run main window tests**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_main_window_wall_openings
```

Expected: all tests pass.

---

### Task 4: Preserve Candidate On Failed Supplement Click

**Files:**
- Modify: `3d_viewer/tests/test_main_window_wall_openings.py`
- Modify: `3d_viewer/ui/main_window.py`

- [ ] **Step 1: Write the failing test**

Add:

```python
    def test_failed_supplement_click_preserves_existing_candidate(self):
        with patch.object(MainWindow, "_load", lambda self: None):
            win = MainWindow(ROOT.parent)
        self.addCleanup(win.close)
        win.cb_supplement.setChecked(True)
        old_mask = np.asarray([True, False, False])
        win._set_highlight_mask(old_mask)
        win._last_opening_candidate = {"mask": old_mask, "label": "window"}
        selection = SimpleNamespace(
            label="window",
            confidence="low",
            reason="rejected_not_vertical_plane",
            point_count=0,
            detection_index=4,
            score=1.0,
            width_m=None,
            height_m=None,
            source="fused",
        )

        effective = win._set_opening_candidate_from_extraction(7, selection, None)

        self.assertTrue(np.array_equal(effective, old_mask))
        self.assertTrue(np.array_equal(win._last_opening_candidate["mask"], old_mask))
```

- [ ] **Step 2: Run the test**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_main_window_wall_openings.MainWindowWallOpeningsTest.test_failed_supplement_click_preserves_existing_candidate
```

Expected: PASS if Task 3 helper was implemented as specified.

---

### Task 5: Make Logs Explicit About Latest vs Saved Mask

**Files:**
- Modify: `3d_viewer/ui/main_window.py`
- Modify: `3d_viewer/tests/test_main_window_wall_openings.py`

- [ ] **Step 1: Extend saved event payload**

In `_log_recorded_opening()`, add:

```python
                "record_mask_source": "accumulated_highlight" if self.cb_supplement.isChecked() else "latest_highlight",
```

- [ ] **Step 2: Extend extraction event payload**

In `_opening_selection_event_payload()`, add:

```python
            "event_mask_source": "effective_highlight" if self.cb_supplement.isChecked() else "latest_highlight",
```

- [ ] **Step 3: Update tests**

In `test_record_current_opening_saves_last_highlight`, assert:

```python
            self.assertEqual(event["record_mask_source"], "latest_highlight")
```

In `test_logs_extraction_candidate_metrics`, assert:

```python
            self.assertEqual(event["event_mask_source"], "latest_highlight")
```

In the new supplement save test, assert the saved event has:

```python
            self.assertEqual(events[-1]["record_mask_source"], "accumulated_highlight")
```

- [ ] **Step 4: Run main window tests**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_main_window_wall_openings
```

Expected: all tests pass.

---

### Task 6: Full Verification

**Files:**
- No edits.

- [ ] **Step 1: Run targeted wall opening tests**

Run:

```bash
.venv/bin/python -m unittest 3d_viewer.tests.test_main_window_wall_openings 3d_viewer.tests.test_wall_model
```

Expected: all tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
.venv/bin/python -m unittest discover -s 3d_viewer/tests
```

Expected: all tests pass.

- [ ] **Step 3: Manual reproduction**

Run:

```bash
.venv/bin/python 3d_viewer/main.py
```

Manual steps:

1. Enable `点云门窗提取`.
2. Enable `补充模式`.
3. Click multiple points on the same window until the displayed highlight covers the complete window region.
4. Click `记录当前门窗`.
5. Inspect `out/wall_openings/20260129135824_events.jsonl`.
6. Confirm the latest `record_opening_saved` event has `record_mask_source: "accumulated_highlight"` and `point_count == highlight_point_count`.
7. Generate wall model and confirm `generate_wall_model_opening_match` reflects the corrected opening position.

Expected: the saved opening bbox matches the screen-highlighted accumulated window region instead of the last small click region.

---

## Self-Review

**Spec coverage:** The plan addresses the code-level root cause, saves accumulated highlight in supplement mode, preserves existing candidates on failed supplement clicks, and improves logs for future unexpected results.

**Placeholder scan:** No placeholders remain; all changed methods and tests have concrete code.

**Type consistency:** The plan uses existing `np.ndarray` boolean masks, existing `MainWindow` fields, and existing wall opening persistence APIs.
