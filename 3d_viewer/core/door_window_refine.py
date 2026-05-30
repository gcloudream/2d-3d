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
