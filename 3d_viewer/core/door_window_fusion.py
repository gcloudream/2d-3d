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
DEFAULT_AUTO_SEED_COUNT = 24


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


def _selection_from_region(
    region: PlanarRegionSelection,
    *,
    detection_index: int,
    label_hint: str | None,
    score: float | None,
) -> FusedSelection:
    if region.confidence == "high":
        fused_conf = "high"
        fused_reason = "fused_frustum_and_planar_geometry"
    elif region.point_count > 0:
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
        label=region.label or (label_hint or "object"),
        source="fused",
        detection_index=int(detection_index),
        score=score,
        plane_point=region.plane_point,
        plane_normal=region.plane_normal,
        width_m=region.width_m,
        height_m=region.height_m,
    )


def _frustum_candidate_mask_from_uv(
    uv: np.ndarray,
    detection: dict,
    pano_w: float,
) -> np.ndarray:
    matches = match_points_to_detections(uv, [detection], float(pano_w))
    return matches.match_indices >= 0


def _depth_clipped_candidate_mask(
    base_candidate: np.ndarray,
    distances: np.ndarray,
    seed_idx: int,
    depth_window: float | None,
) -> np.ndarray:
    candidate = np.asarray(base_candidate, dtype=bool).copy()
    if depth_window is not None:
        seed_depth = float(distances[seed_idx])
        candidate &= np.abs(distances - seed_depth) <= float(depth_window)
        candidate[seed_idx] = True
    return candidate


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
    inside = _frustum_candidate_mask_from_uv(uv, detection, float(img_w))
    if seed_idx is not None and depth_window is not None:
        cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
        dist = np.linalg.norm(np.asarray(points, dtype=np.float64) - cam.reshape(1, 3), axis=1)
        inside = _depth_clipped_candidate_mask(inside, dist, seed_idx, depth_window)
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
    detection_score = float(detection["score"]) if "score" in detection else None

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
    return _selection_from_region(
        region,
        detection_index=det_idx,
        label_hint=label,
        score=detection_score,
    )


def extract_detection_region_from_bbox(
    points: np.ndarray,
    detection_index: int,
    detections: Sequence[dict],
    cam_pos: np.ndarray,
    R_pano: np.ndarray,
    img_w: int,
    img_h: int,
    yaw_offset_deg: float = 0.0,
    *,
    click_uv: tuple[float, float] | None = None,
    max_seed_count: int = DEFAULT_AUTO_SEED_COUNT,
) -> FusedSelection:
    """Extract a detection's best door/window region without a manually clicked seed."""
    pts = np.asarray(points)
    n = len(pts)
    if detection_index < 0 or detection_index >= len(detections):
        return _empty(n, "detection_index_out_of_range")
    if img_w <= 0 or img_h <= 0:
        return _empty(n, "invalid_image_size")

    detection = detections[detection_index]
    label = str(detection.get("label", "")).lower() or None
    detection_score = float(detection["score"]) if "score" in detection else None
    cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    pts64 = np.asarray(pts, dtype=np.float64)
    distances = np.linalg.norm(pts64 - cam.reshape(1, 3), axis=1)
    uv = project_points_to_panorama(
        pts, cam_pos, R_pano, img_w, img_h, yaw_offset_deg=yaw_offset_deg
    )
    target_uv = _valid_click_uv(click_uv, float(img_w))
    candidate = _frustum_candidate_mask_from_uv(uv, detection, float(img_w))
    candidate_indices = np.flatnonzero(candidate)
    if len(candidate_indices) == 0:
        return _empty(n, "bbox_no_points")

    best: FusedSelection | None = None
    for seed_idx in _auto_seed_indices(
        pts,
        candidate_indices,
        cam_pos,
        max_seed_count=max_seed_count,
        uv=uv,
        click_uv=target_uv,
        pano_w=float(img_w),
    ):
        clipped_candidate = _depth_clipped_candidate_mask(
            candidate,
            distances,
            int(seed_idx),
            DEFAULT_FRUSTUM_DEPTH_WINDOW,
        )
        region = extract_planar_region_from_seed(
            pts,
            int(seed_idx),
            candidate_mask=clipped_candidate,
            label_hint=label,
        )
        selection = _selection_from_region(
            region,
            detection_index=int(detection_index),
            label_hint=label,
            score=detection_score,
        )
        selection_score = _fused_selection_score(selection, uv=uv, click_uv=target_uv, pano_w=float(img_w))
        best_score = None if best is None else _fused_selection_score(
            best, uv=uv, click_uv=target_uv, pano_w=float(img_w)
        )
        if best_score is None or selection_score > best_score:
            best = selection

    if best is None or best.point_count <= 0:
        return _empty(n, "bbox_no_extractable_region")
    return best


def _auto_seed_indices(
    points: np.ndarray,
    candidate_indices: np.ndarray,
    cam_pos: np.ndarray,
    *,
    max_seed_count: int,
    uv: np.ndarray | None = None,
    click_uv: tuple[float, float] | None = None,
    pano_w: float | None = None,
) -> np.ndarray:
    if len(candidate_indices) <= max_seed_count:
        return candidate_indices
    cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    pts = np.asarray(points, dtype=np.float64)
    depths = np.linalg.norm(pts[candidate_indices] - cam.reshape(1, 3), axis=1)
    ordered = candidate_indices[np.argsort(depths)]
    seed_budget = max(1, int(max_seed_count))
    click_order = np.asarray([], dtype=np.int64)
    if uv is not None and click_uv is not None and pano_w is not None:
        distances = _panorama_pixel_distances(uv[candidate_indices], click_uv, float(pano_w))
        nearest = candidate_indices[np.argsort(distances)]
        click_count = max(1, seed_budget // 2)
        click_order = nearest[:click_count]

    depth_count = max(1, seed_budget - len(click_order))
    sample_positions = np.linspace(0, len(ordered) - 1, depth_count).round().astype(np.int64)
    depth_order = ordered[sample_positions]
    combined = np.concatenate([click_order, depth_order])
    return _unique_preserve_order(combined)[:seed_budget]


def _fused_selection_score(
    selection: FusedSelection,
    *,
    uv: np.ndarray | None = None,
    click_uv: tuple[float, float] | None = None,
    pano_w: float | None = None,
) -> tuple[int, int, int, int, int, int]:
    confidence_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}.get(selection.confidence, 0)
    reason_rank = 1 if selection.reason == "fused_frustum_and_planar_geometry" else 0
    label_rank = 1 if selection.label in {"door", "window"} else 0
    click_center_rank, click_spread_rank = _selection_click_ranks(
        selection, uv=uv, click_uv=click_uv, pano_w=pano_w
    )
    return (
        confidence_rank,
        reason_rank,
        label_rank,
        click_center_rank,
        click_spread_rank,
        int(selection.point_count),
    )


def _valid_click_uv(click_uv: tuple[float, float] | None, pano_w: float) -> tuple[float, float] | None:
    if click_uv is None:
        return None
    arr = np.asarray(click_uv, dtype=np.float64).reshape(-1)
    if len(arr) < 2 or not np.isfinite(arr[:2]).all():
        return None
    u = float(arr[0])
    if pano_w > 0:
        u %= float(pano_w)
    return u, float(arr[1])


def _panorama_pixel_distances(
    uv: np.ndarray,
    click_uv: tuple[float, float],
    pano_w: float,
) -> np.ndarray:
    pts_uv = np.asarray(uv, dtype=np.float64)
    target_u, target_v = click_uv
    dx = np.abs(pts_uv[:, 0] - float(target_u))
    if pano_w > 0:
        dx = np.minimum(dx, float(pano_w) - dx)
    dy = pts_uv[:, 1] - float(target_v)
    return np.hypot(dx, dy)


def _selection_click_ranks(
    selection: FusedSelection,
    *,
    uv: np.ndarray | None,
    click_uv: tuple[float, float] | None,
    pano_w: float | None,
) -> tuple[int, int]:
    if uv is None or click_uv is None or pano_w is None:
        return 0, 0
    mask = np.asarray(selection.mask, dtype=bool)
    if len(mask) != len(uv) or not mask.any():
        return 0, 0
    distances = _panorama_pixel_distances(uv[mask], click_uv, float(pano_w))
    median_dist = float(np.median(distances))
    spread_dist = float(np.percentile(distances, 90))
    return -int(round(median_dist * 1000.0)), -int(round(spread_dist * 100.0))


def _unique_preserve_order(values: np.ndarray) -> np.ndarray:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return np.asarray(ordered, dtype=np.int64)


def should_highlight_fused(selection: FusedSelection) -> bool:
    return selection.point_count > 0 and not (
        selection.confidence == "low" and selection.reason.startswith("rejected_")
    )
