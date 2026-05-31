"""Fuse 2D detection frustums with pure point-cloud geometry.

The two independent door/window judges each have a blind spot:

* The panorama detection box constrains *viewing angle* and carries a semantic
  label (door vs window), but its 3D back-projection is a metres-deep frustum
  cone that cannot tell a door from the wall behind it.
* Pure point-cloud extraction nails *depth* and planar geometry but has no
  semantics, and a click on a wall-flush door grows across the whole wall.

Fusing them takes the intersection, so each constraint covers the other's gap:

    2D frustum   -> bounds the angular extent, so growth cannot swallow a wall
    3D geometry  -> resolves the frustum's depth ambiguity, keeping one sheet
    2D label     -> supplies door/window semantics the geometry cannot infer

This module orchestrates that fusion; it owns no new geometry maths and reuses
``door_window`` (frustum membership), ``projection`` (point -> pixel), and
``pointcloud_extract`` (constrained planar growth + scoring).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.door_window import match_points_to_detections
from core.pointcloud_extract import (
    PlanarRegionSelection,
    extract_planar_region_from_seed,
)
from core.projection import project_points_to_panorama


# How close to the clicked seed's camera distance a frustum point must be to
# join the seed's depth shell (separates the near door from the far wall that
# share the same bbox cone before geometry growth runs).
DEFAULT_FRUSTUM_DEPTH_WINDOW = 0.6


@dataclass(frozen=True)
class FusedSelection:
    mask: np.ndarray
    point_count: int
    confidence: str          # high | medium | low | none
    reason: str
    label: str               # door | window | object
    source: str              # fused | pointcloud_only | none
    detection_index: int     # frustum hit, or -1
    score: float | None      # 2D detection score, or None
    plane_point: np.ndarray | None
    plane_normal: np.ndarray | None
    width_m: float | None
    height_m: float | None


def _empty(n: int, reason: str) -> FusedSelection:
    return FusedSelection(
        mask=np.zeros(n, dtype=bool),
        point_count=0,
        confidence="none",
        reason=reason,
        label="",
        source="none",
        detection_index=-1,
        score=None,
        plane_point=None,
        plane_normal=None,
        width_m=None,
        height_m=None,
    )


def frustum_candidate_mask(
    points: np.ndarray,
    cam_pos: np.ndarray,
    R_pano: np.ndarray,
    img_w: int,
    img_h: int,
    detection: dict,
    yaw_offset_deg: float = 0.0,
    seed_idx: int | None = None,
    depth_window: float | None = DEFAULT_FRUSTUM_DEPTH_WINDOW,
) -> np.ndarray:
    """Points whose projection lands inside ``detection``'s bbox (its frustum).

    When ``seed_idx`` and ``depth_window`` are given the cone is additionally
    clipped to a depth shell around the seed so the far wall stacked behind the
    door in the same bbox is removed before growing.
    """
    uv = project_points_to_panorama(
        points, cam_pos, R_pano, img_w, img_h, yaw_offset_deg=yaw_offset_deg
    )
    matches = match_points_to_detections(uv, [detection], float(img_w))
    inside = matches.match_indices >= 0
    if seed_idx is not None and depth_window is not None:
        cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
        dist = np.linalg.norm(np.asarray(points, dtype=np.float64) - cam.reshape(1, 3), axis=1)
        seed_depth = float(dist[seed_idx])
        inside = inside & (np.abs(dist - seed_depth) <= float(depth_window))
        inside[seed_idx] = True
    return inside


def fuse_detection_and_pointcloud(
    points: np.ndarray,
    seed_idx: int,
    detections: Sequence[dict],
    cam_pos: np.ndarray,
    R_pano: np.ndarray,
    img_w: int,
    img_h: int,
    yaw_offset_deg: float = 0.0,
    frustum_depth_window: float = DEFAULT_FRUSTUM_DEPTH_WINDOW,
) -> FusedSelection:
    """Extract a door/window region by fusing the clicked seed's frustum + geometry.

    Strategy A: uses ``detections`` from the currently selected keyframe. If the
    seed falls inside a detection frustum the constrained pure-point-cloud growth
    runs inside that frustum and inherits the detection label; otherwise the
    caller should fall back to the unconstrained pure-point-cloud path.
    """
    pts = np.asarray(points)
    n = len(pts)
    if seed_idx < 0 or seed_idx >= n:
        return _empty(n, "seed_out_of_range")
    if not detections or img_w <= 0 or img_h <= 0:
        return _empty(n, "no_detections")

    # 1. Which detection frustum contains the clicked seed? (smallest bbox wins,
    #    matching the 2D-guided path's nested-box preference).
    uv = project_points_to_panorama(
        pts, cam_pos, R_pano, img_w, img_h, yaw_offset_deg=yaw_offset_deg
    )
    matches = match_points_to_detections(uv, detections, float(img_w))
    det_idx = int(matches.match_indices[seed_idx])
    if det_idx < 0:
        return _empty(n, "seed_not_in_any_frustum")

    detection = detections[det_idx]
    label = str(detection.get("label", "")).lower() or None
    score = float(detection["score"]) if "score" in detection else None

    # 2. Frustum candidate set for that detection, depth-clipped around the seed.
    candidate = frustum_candidate_mask(
        pts, cam_pos, R_pano, img_w, img_h, detection,
        yaw_offset_deg=yaw_offset_deg,
        seed_idx=seed_idx,
        depth_window=frustum_depth_window,
    )

    # 3 + 4. Constrained planar growth + geometry scoring, with the 2D label as
    #        the size-check hint. Growth is bounded by the frustum candidate set,
    #        so a wall-flush door no longer spreads across the whole wall.
    region: PlanarRegionSelection = extract_planar_region_from_seed(
        pts,
        seed_idx,
        candidate_mask=candidate,
        label_hint=label,
    )

    # 5. Fuse the confidences. The frustum already agreed (seed is inside it), so
    #    a high-confidence geometry means both judges concur.
    if region.confidence == "high":
        fused_conf = "high"
        fused_reason = "fused_frustum_and_planar_geometry"
    elif region.point_count > 0:
        # Frustum matched but geometry could not confirm a clean planar door.
        fused_conf = "medium"
        fused_reason = f"frustum_only_{region.reason}"
    else:
        fused_conf = "low"
        fused_reason = region.reason

    return FusedSelection(
        mask=region.mask,
        point_count=region.point_count,
        confidence=fused_conf,
        reason=fused_reason,
        label=region.label or (label or "object"),
        source="fused",
        detection_index=det_idx,
        score=score,
        plane_point=region.plane_point,
        plane_normal=region.plane_normal,
        width_m=region.width_m,
        height_m=region.height_m,
    )


def should_highlight_fused(selection: FusedSelection) -> bool:
    return selection.point_count > 0 and not (
        selection.confidence == "low" and selection.reason.startswith("rejected_")
    )
