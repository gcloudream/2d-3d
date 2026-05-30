# Dual Point Cloud View Design

## Purpose

The current 3D viewer uses one primary OpenGL scene for the interactive workflow: panorama image, point cloud, detection boxes, and selected door/window points are all rendered in the same camera view. This is useful for selecting points from the panorama-aligned view, but it makes it harder to inspect the selected points as a 3D object inside the whole point cloud.

This design adds a second synchronized 3D point cloud view. The user continues selecting door/window points in the existing panorama-overlap view, while a new pure 3D point cloud view shows the entire scene and highlights the same selected points in real 3D space.

## Goals

- Keep the existing panorama-overlap view as the main interaction surface.
- Add a pure 3D point cloud view in the same window, using the same sampled point cloud.
- Synchronize selected point masks from the main view into the pure 3D view.
- Let the pure 3D view be independently rotated and zoomed for spatial inspection.
- Keep the right-side controls and existing keyframe workflow intact.

## Non-Goals

- Do not replace the current panorama-overlap selection workflow.
- Do not implement full LAS export in this step.
- Do not make the two view cameras move in lockstep; independent orbit is more useful for inspection.
- Do not duplicate detection editing or selection interactions in the pure 3D view.

## Recommended UI Layout

Use a vertical splitter inside the main view area:

1. Top pane: the existing panorama + point cloud + detection boxes scene.
2. Bottom pane: a new pure 3D point cloud scene.
3. Right pane: the existing keyframe list and controls.

Initial splitter sizing should favor the existing workflow, for example 70% top and 30% bottom. The user can resize the splitter manually.

## Behavior

### Main Panorama-Overlap View

The existing `self.scene` remains responsible for:

- panorama display,
- point cloud display,
- detection box overlay,
- hover and click picking,
- door/window selection mode,
- selected bbox outline.

### Pure 3D Point Cloud View

The new view should be another `SceneView` instance configured as an observer:

- show point cloud: enabled,
- show panorama: disabled,
- show detection boxes: disabled,
- pick mode: disabled,
- hover/click signals ignored by `MainWindow`,
- independent camera orbit and zoom,
- receives the same world points and selected highlight mask.

It should still receive the current keyframe pose so `reset_view` has a meaningful camera position. Users can then rotate away from that pose to inspect the full scene.

### Synchronization

When the dataset loads:

- upload the same sampled points and colors to both views,
- apply the current point size to both views,
- set the pure 3D view to point-cloud-only mode.

When the keyframe changes:

- set the keyframe on both views,
- keep panorama disabled on the pure 3D view,
- clear highlight masks in both views.

When a door/window selection succeeds:

- main view shows the selected detection and allowed highlight behavior,
- pure 3D view receives the same accepted highlight mask,
- rejected low-confidence geometry clears the pure 3D highlight as well.

When detection mode changes or annotation edits reload detections:

- clear highlight masks in both views.

When point size changes:

- apply the same point size to both views.

## Component Changes

### `3d_viewer/ui/main_window.py`

Add a second `SceneView`, for example `self.cloud_scene`.

Replace the current `QStackedWidget` setup with a container that preserves the annotation editor switch:

- Normal mode: show a vertical splitter containing `self.scene` and `self.cloud_scene`.
- Annotation mode: show the existing `PanoAnnotationEditor`.

Add small helper methods:

- `_set_highlight_mask(mask)` to update both scenes.
- `_clear_selection()` to clear highlight and selected detection consistently.
- `_set_point_size(value)` to update both scenes.
- `_configure_cloud_scene()` to turn off panorama, bbox overlay, and pick mode.

### `3d_viewer/render/scene_view.py`

No major rendering changes are expected. The existing `SceneView` API already supports:

- `set_world_points`,
- `set_keyframe`,
- `set_show_pano`,
- `set_show_pc`,
- `set_show_bboxes`,
- `set_highlight_mask`,
- `set_point_size`,
- `reset_view`.

If needed, add a small convenience method only when it reduces duplicated setup in `MainWindow`.

## Testing

Use lightweight UI-logic tests where possible:

- verify `MainWindow` creates both scene instances,
- verify point-size updates are forwarded to both scenes,
- verify accepted selection masks are forwarded to both scenes,
- verify rejected selections clear both highlights.

Manual verification:

1. Start the viewer with `run_3d_viewer.command` or `.venv/bin/python 3d_viewer/main.py`.
2. Confirm two OpenGL panes appear in normal mode.
3. Confirm the top pane still shows panorama + point cloud + boxes.
4. Confirm the bottom pane shows only the point cloud.
5. Select a door/window point in the top pane.
6. Confirm the selected points are highlighted in the bottom pure 3D pane.
7. Rotate/zoom the bottom pane independently.
8. Switch keyframes and confirm both panes update.

## Risks

- Two OpenGL views upload the same sampled point cloud, so GPU memory use roughly doubles for viewer points.
- Two native OpenGL windows in a splitter may expose platform-specific resize quirks on macOS.
- If the pure 3D view accidentally remains connected to picking, user clicks in the bottom pane could confuse the door/window workflow.

The implementation should keep the first version small and reversible: two `SceneView` instances, synchronized from `MainWindow`, with no new rendering subsystem.
