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
DEFAULT_MIN_GEOMETRY_POINTS = 20
DEFAULT_MAX_PLANE_RMS_ERROR = 0.08
DEFAULT_VERTICAL_NORMAL_Z_MAX = 0.35

DIMENSION_LIMITS = {
    "door": (0.5, 1.5, 1.6, 2.6),
    "window": (0.3, 3.5, 0.3, 2.2),
}


@dataclass(frozen=True)
class PlaneFit:
    point: np.ndarray
    normal: np.ndarray
    rms_error: float


@dataclass(frozen=True)
class GeometryScore:
    confidence: str
    reason: str
    plane_point: np.ndarray | None
    plane_normal: np.ndarray | None
    width_m: float | None
    height_m: float | None
    plane_rms_error: float | None


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
    plane_point: np.ndarray | None
    plane_normal: np.ndarray | None
    width_m: float | None
    height_m: float | None
    plane_rms_error: float | None


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
        plane_point=None,
        plane_normal=None,
        width_m=None,
        height_m=None,
        plane_rms_error=None,
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


def fit_plane_least_squares(points: np.ndarray) -> PlaneFit:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) < 3:
        raise ValueError("at least 3 points are required to fit a plane")

    centroid = pts.mean(axis=0)
    centered = pts - centroid.reshape(1, 3)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise ValueError("cannot fit a plane to degenerate points")
    normal = normal / norm
    if normal[2] < 0:
        normal = -normal
    distances = centered @ normal
    rms_error = float(np.sqrt(np.mean(distances * distances)))
    return PlaneFit(point=centroid, normal=normal, rms_error=rms_error)


def estimate_planar_extent(points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(points, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n_norm = float(np.linalg.norm(n))
    if n_norm == 0.0:
        raise ValueError("normal must be non-zero")
    n = n / n_norm

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    height_axis = z_axis - float(np.dot(z_axis, n)) * n
    height_norm = float(np.linalg.norm(height_axis))
    if height_norm < 1e-6:
        centered = pts - pts.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_a = vh[0]
        axis_b = vh[1]
    else:
        axis_b = height_axis / height_norm
        axis_a = np.cross(axis_b, n)
        axis_a = axis_a / float(np.linalg.norm(axis_a))

    coord_a = pts @ axis_a
    coord_b = pts @ axis_b
    width_m = float(coord_a.max() - coord_a.min())
    height_m = float(coord_b.max() - coord_b.min())
    return width_m, height_m


def score_door_window_geometry(
    points: np.ndarray,
    label: str,
    min_points: int = DEFAULT_MIN_GEOMETRY_POINTS,
    max_plane_rms_error: float = DEFAULT_MAX_PLANE_RMS_ERROR,
    vertical_normal_z_max: float = DEFAULT_VERTICAL_NORMAL_Z_MAX,
) -> GeometryScore:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < int(min_points):
        return GeometryScore("low", "rejected_too_few_geometry_points", None, None, None, None, None)

    try:
        plane = fit_plane_least_squares(pts)
    except ValueError:
        return GeometryScore("low", "rejected_plane_fit_failed", None, None, None, None, None)

    width_m, height_m = estimate_planar_extent(pts, plane.normal)
    if plane.rms_error > float(max_plane_rms_error):
        return GeometryScore(
            "low",
            "rejected_not_planar",
            plane.point,
            plane.normal,
            width_m,
            height_m,
            plane.rms_error,
        )

    if abs(float(plane.normal[2])) > float(vertical_normal_z_max):
        return GeometryScore(
            "low",
            "rejected_not_vertical_plane",
            plane.point,
            plane.normal,
            width_m,
            height_m,
            plane.rms_error,
        )

    clean_label = str(label).lower()
    limits = DIMENSION_LIMITS.get(clean_label)
    if limits is not None:
        min_w, max_w, min_h, max_h = limits
        if not (min_w <= width_m <= max_w and min_h <= height_m <= max_h):
            return GeometryScore(
                "low",
                f"rejected_implausible_{clean_label}_size",
                plane.point,
                plane.normal,
                width_m,
                height_m,
                plane.rms_error,
            )

    accepted_label = clean_label if clean_label in DIMENSION_LIMITS else "object"
    return GeometryScore(
        "high",
        f"accepted_vertical_{accepted_label}_geometry",
        plane.point,
        plane.normal,
        width_m,
        height_m,
        plane.rms_error,
    )


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
    plane_point = None
    plane_normal = None
    width_m = None
    height_m = None
    plane_rms_error = None

    if point_count >= max(int(min_refined_points), DEFAULT_MIN_GEOMETRY_POINTS):
        geometry = score_door_window_geometry(points_arr[depth_mask], str(det.get("label", "")))
        confidence = geometry.confidence
        reason = geometry.reason
        plane_point = geometry.plane_point
        plane_normal = geometry.plane_normal
        width_m = geometry.width_m
        height_m = geometry.height_m
        plane_rms_error = geometry.plane_rms_error

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
        plane_point=plane_point,
        plane_normal=plane_normal,
        width_m=width_m,
        height_m=height_m,
        plane_rms_error=plane_rms_error,
    )
