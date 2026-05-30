# Door/Window Point Cloud Fusion Design

## Purpose

The current 3D viewer selects door/window point cloud regions primarily by projecting sampled 3D points into the panorama image and checking whether each projected pixel falls inside a 2D door/window detection box. This is useful for rough selection, but it treats the 2D box as the final geometry. If the panorama detector catches a cabinet, shelf, monitor, reflection, or an oversized area around a real door/window, the extracted point cloud inherits that error.

This design changes the selection model so that the 2D detection box supplies a coarse region of interest, while 3D point cloud geometry confirms and refines the actual selected region.

## Goals

- Use panorama door/window detections as coarse candidate regions, not final extraction masks.
- Refine each selected region using point cloud priors such as connectivity, depth consistency, vertical plane structure, size, and rectangularity.
- Preserve the existing interaction: the user can still enable door/window selection and click a point in the 3D view.
- Return a clear result even when refinement rejects a candidate: show the coarse match, explain why 3D validation failed, and avoid silently highlighting bad geometry.
- Keep the first implementation testable on sampled viewer points, while leaving a path to run the same rules against the full LAS for export-quality output.

## Non-Goals

- This does not replace the existing 2D detector in the first version.
- This does not require training a new model.
- This does not attempt global pure point-cloud door/window search across the whole scene in the first version.
- This does not promise perfect glass extraction; glass often has weak or missing LiDAR returns.

## Current Pipeline

The current click flow in `3d_viewer/ui/main_window.py` is:

1. User clicks a point in door/window selection mode.
2. The viewer projects all sampled points into the current panorama using `project_points_to_panorama`.
3. `select_detection_region` finds which 2D detection bbox contains the clicked point.
4. All points whose projected pixels fall inside that same bbox are highlighted.

This produces a `bbox_mask`, but it does not inspect whether those points form a plausible 3D door/window.

## Recommended Pipeline

The refined flow should be:

1. Project sampled points into the current panorama.
2. Match the clicked point to a 2D door/window bbox, using the existing wrap-aware bbox logic.
3. Build a coarse ROI from points projected inside that bbox.
4. Start from the clicked point as a seed.
5. Inside the coarse ROI, keep the connected 3D component near the seed.
6. Apply depth and outlier filtering to remove points that are in front of or behind the selected surface.
7. Fit one or more local planes to the remaining candidate points.
8. Score the candidate with door/window priors.
9. Highlight the refined mask when confidence is acceptable; otherwise show a rejection reason and optionally display the coarse mask as a diagnostic.

The central principle is:

```text
2D bbox = candidate generator
clicked point = target selector
3D geometry = final validator and mask refiner
```

## 3D Priors

The first implementation should use deterministic geometry rules instead of another model:

- Connectivity: select only the ROI component connected to the clicked seed in 3D space.
- Depth consistency: reject points that are far in depth from the seed-relative surface band.
- Plane fit: door/window candidates should usually lie on or near a dominant vertical plane.
- Verticality: the fitted plane normal should be mostly horizontal, and the candidate's principal axes should include a vertical direction.
- Size range: door and window candidates should fall inside plausible width and height ranges.
- Rectangularity: the projected 3D boundary on the fitted plane should be roughly rectangular.
- Density sanity: candidates with too few points, excessive scatter, or very low inlier ratio should be marked low confidence.

Suggested initial thresholds:

- Neighbor radius: `0.12m-0.25m`, configurable.
- Plane inlier distance: `0.03m-0.08m`.
- Door height: `1.6m-2.6m`; door width: `0.5m-1.5m`.
- Window height: `0.3m-2.2m`; window width: `0.3m-3.5m`.
- Minimum refined sampled points: `80` for sampled viewer data.
- Minimum plane inlier ratio: `0.45` for normal objects; lower confidence rather than hard reject for glass.

These are starting values and should be visible in code as named constants so they can be tuned from real samples.

## Proposed Components

### `core/door_window_refine.py`

New core module responsible for 3D refinement.

Public API:

```python
@dataclass(frozen=True)
class RefinedDoorWindowSelection:
    detection_index: int
    detection: dict | None
    label: str
    coarse_mask: np.ndarray
    refined_mask: np.ndarray
    confidence: str
    point_count: int
    coarse_count: int
    reason: str
    plane_normal: np.ndarray | None
    plane_point: np.ndarray | None
    bbox_3d: np.ndarray | None


def refine_detection_selection(
    points: np.ndarray,
    uv: np.ndarray,
    clicked_idx: int,
    detections: Sequence[dict],
    pano_w: float,
    label: str | None = None,
) -> RefinedDoorWindowSelection:
    ...
```

The function should call existing bbox matching code first, then refine the selected bbox hit set.

### Geometry Helpers

The module should include small, testable helpers:

- `connected_component_from_seed(points, candidate_mask, seed_idx, radius)`
- `filter_by_seed_depth(points, component_mask, seed_idx, max_delta)`
- `fit_plane_ransac(points, threshold, iterations)`
- `score_door_window_geometry(points, label, plane)`

The helpers should not depend on Qt, ModernGL, or viewer state.

### UI Integration

`MainWindow._on_point_clicked` should change from:

```text
select_detection_region -> highlight bbox mask
```

to:

```text
refine_detection_selection -> highlight refined mask
```

The right-side status panel should show:

- detection label and score
- coarse point count
- refined point count
- confidence
- rejection or acceptance reason

If refinement fails, the UI should not pretend a precise 3D object was found. It should clear or dim the highlight and show a message such as:

```text
2D bbox matched, but 3D validation rejected it: not enough connected vertical-plane points.
```

## Error Handling

- No bbox hit: keep current behavior.
- Too few candidate points: return confidence `low` and reason `too_few_coarse_points`.
- Seed not in a connected component: return confidence `low` and reason `seed_component_empty`.
- Plane fit fails: return the connected component as a low-confidence diagnostic result, not a crash.
- Geometry score below threshold: reject refined highlight and explain which rule failed.

## Testing Strategy

Unit tests should cover synthetic point clouds:

- bbox-only behavior still finds the same coarse mask as current code.
- connected component keeps the seed component and excludes another object inside the same bbox.
- plane fitting accepts a vertical rectangular patch.
- plane fitting rejects a scattered point set.
- door/window size scoring accepts plausible dimensions and rejects implausible dimensions.
- failure cases return clear reasons rather than exceptions.

Validation scripts should run on real data:

- Given one keyframe and detection JSON, export a comparison image:
  - coarse bbox mask
  - refined 3D mask
  - rejected/outlier points
- Print counts and confidence for each detection.

## Implementation Phases

### Phase 1: Seed-Constrained Refinement

- Add `door_window_refine.py`.
- Implement bbox coarse mask, seed connected component, and depth filtering.
- Integrate refined highlight into the current click flow.
- Add unit tests for masks and failure reasons.

Expected benefit: removes many unrelated points from oversized or slightly wrong bboxes.

### Phase 2: Geometry Scoring

- Add RANSAC plane fit and principal-axis size estimation.
- Add label-aware door/window size thresholds.
- Return confidence and reason strings.
- Show confidence in the UI.

Expected benefit: rejects shelves, monitors, and cabinets that fall inside 2D detections but do not form plausible door/window geometry.

### Phase 3: Export-Quality Full LAS Refinement

- Reuse the same projection and geometry rules on full LAS data for final export.
- Keep sampled viewer extraction for interactive feedback.
- Export refined door/window regions to `.ply` or `.las`.

Expected benefit: interactive selection remains fast, while exported regions preserve more edge detail.

## Recommendation

Implement Phase 1 first. It is the smallest useful change and fits the current code structure. Then evaluate real examples from the office scan. If the refined components still include too many wrong objects, add Phase 2 plane and size scoring.

The design deliberately keeps the 2D detector as a coarse guide and the 3D point cloud as the final geometric authority. This matches the intended behavior: do not rely on the panorama bbox alone, and do not attempt fragile global pure point-cloud detection before the search space is constrained.
