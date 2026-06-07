# Door/Window Markers In Wall Models Design

## Goal

Record door/window regions selected by the existing point-cloud extraction flow, then reuse those records when generating wall-model artifacts. The wall top-down image and OBJ preview should show door/window locations as highlighted rectangular frames. The OBJ should not cut holes in the wall mesh.

## Current State

The main viewer can highlight a door/window region from a clicked point. That result is currently transient: `MainWindow` stores a boolean `_highlight_mask`, and the extraction result includes useful geometry such as label, confidence, plane point, plane normal, width, and height. The wall-model generator currently reads only the sampled point cloud and outputs wall segments, a top-down image, metadata JSON, and an OBJ wall mesh.

## Proposed User Flow

1. The user enables point-cloud door/window extraction in the main view.
2. The user clicks a door/window point and verifies the highlighted region.
3. The user clicks a new `记录当前门窗` button.
4. The app saves a door/window marker record for the current dataset.
5. The user opens the wall-model workbench and generates wall artifacts.
6. The generated top-down image and OBJ show the recorded door/window positions as rectangular frames.

## Data Model

Persist records under the workspace output directory, for example:

`out/wall_openings/<dataset_name>_openings.json`

Each opening record should include:

- Stable id
- Label: `door`, `window`, or `object`
- Source keyframe image name
- Seed point index
- Point count
- Confidence and reason
- Center point in world coordinates
- Plane normal
- Estimated width and height
- Vertical range from selected points
- Optional detection score/index when the fused extraction path was used

The saved record should use world-space geometry rather than only point indices, because wall-model generation may use a different point sampling count later.

## Mapping Openings To Walls

During wall-model generation, load saved openings and match each one to the closest compatible wall segment:

- Compare the opening center to each wall segment.
- Require the opening to project within the segment span, with a small tolerance.
- Require distance to the wall plane to be under a configurable threshold.
- Prefer segments whose axis-aligned orientation is compatible with the opening plane normal.
- Clamp the projected rectangle to the wall segment bounds.

Openings that cannot be matched should not be forced into the model. Store them in metadata as `unmatched_openings` with the reason.

## Top-Down Preview

Draw recorded door/window markers over the existing wall-line image:

- Keep existing wall lines red.
- Draw matched openings as short high-contrast colored rectangles or bars on top of the matching wall.
- Use distinct colors for door and window markers.
- Optionally label each marker with a small id if it remains legible.

## OBJ Output

Do not cut holes in wall geometry. Add rectangular marker-frame geometry slightly offset from the wall surface:

- The marker lies on the matching wall face.
- It uses the opening's projected wall-axis interval and vertical range.
- It is offset outward by a tiny amount to avoid z-fighting.
- It is represented as thin rectangular frame strips, so it is visible even with the current uniform-color OBJ preview and mesh edge overlay.
- The generated OBJ may include `g opening_<id>` groups for downstream inspection.

## UI Changes

Main viewer:

- Add `记录当前门窗`.
- Enable it only when a valid highlighted region exists.
- Show a status message after saving, including label and record count.
- Keep existing `清除高亮` behavior.

Wall-model workbench:

- Load saved openings automatically when generating.
- Show counts for matched and unmatched openings in the info panel.

## Error Handling

- If no highlighted region exists, `记录当前门窗` should explain that the user must extract a door/window first.
- If the selected region is too small or lacks geometry, do not save a record.
- If no opening file exists, wall generation proceeds exactly as it does today.
- If some openings cannot match walls, still generate the wall model and report unmatched records in metadata.

## Testing

Add focused tests for:

- Serializing and loading opening records.
- Building an opening record from a highlighted extraction result.
- Matching openings to horizontal and vertical wall segments.
- Rejecting unmatched openings.
- Rendering top-down opening markers.
- Generating OBJ marker-frame geometry without changing wall opening behavior.
- Main window button behavior for saving the current highlighted opening.

## Non-Goals

- No real mesh cut-outs in this iteration.
- No manual rectangle editing UI in this iteration.
- No material/color OBJ viewer support required in this iteration.
