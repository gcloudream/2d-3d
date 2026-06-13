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

import json
import logging
from dataclasses import dataclass, replace
from time import perf_counter
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
LOGGER = logging.getLogger("3d_viewer.core.door_window_fusion")


def _array_cache_key(values: np.ndarray | Sequence[float]) -> tuple[tuple[int, ...], bytes]:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return arr.shape, arr.tobytes()


def _points_cache_key(points: np.ndarray) -> tuple[int, tuple[int, ...], str, tuple[int, ...]]:
    arr = np.asarray(points)
    data_ptr = int(arr.__array_interface__["data"][0]) if arr.size else 0
    return data_ptr, tuple(arr.shape), str(arr.dtype), tuple(arr.strides)


def _bbox_cache_key(detection: dict) -> tuple[float, ...]:
    return tuple(float(v) for v in detection.get("bbox", ()))


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000.0, 3)


def _finish_diagnostics(diagnostics: dict[str, object], start: float) -> dict[str, object]:
    diagnostics["total_ms"] = _elapsed_ms(start)
    return diagnostics


def _log_fusion_result(event: str, selection: "FusedSelection"):
    diagnostics = selection.diagnostics or {}
    LOGGER.info(
        "%s reason=%s confidence=%s label=%s points=%s detection=%s diagnostics=%s",
        event,
        selection.reason,
        selection.confidence,
        selection.label,
        selection.point_count,
        selection.detection_index,
        json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
    )


class FrustumProjectionCache:
    """Cache projected panorama coordinates and bbox masks for one point cloud.

    The point cloud is treated as immutable while the cache is in use. MainWindow
    clears this cache when the keyframe, detections, or yaw calibration changes.
    """

    def __init__(self):
        self._projection_key: tuple[object, ...] | None = None
        self._projection_uv: np.ndarray | None = None
        self._candidate_masks: dict[tuple[object, ...], np.ndarray] = {}
        self._distance_key: tuple[object, ...] | None = None
        self._distances: np.ndarray | None = None

    def clear(self):
        self._projection_key = None
        self._projection_uv = None
        self._candidate_masks.clear()
        self._distance_key = None
        self._distances = None

    def project(
        self,
        points: np.ndarray,
        cam_pos: np.ndarray,
        R_pano: np.ndarray,
        img_w: int,
        img_h: int,
        yaw_offset_deg: float,
    ) -> tuple[np.ndarray, bool]:
        key = (
            _points_cache_key(points),
            _array_cache_key(cam_pos),
            _array_cache_key(R_pano),
            int(img_w),
            int(img_h),
            float(yaw_offset_deg),
        )
        if self._projection_key == key and self._projection_uv is not None:
            return self._projection_uv, True
        uv = project_points_to_panorama(
            points, cam_pos, R_pano, img_w, img_h, yaw_offset_deg=yaw_offset_deg
        )
        self._projection_key = key
        self._projection_uv = uv
        self._candidate_masks.clear()
        return uv, False

    def distances(self, points: np.ndarray, cam_pos: np.ndarray) -> tuple[np.ndarray, bool]:
        key = (_points_cache_key(points), _array_cache_key(cam_pos))
        if self._distance_key == key and self._distances is not None:
            return self._distances, True
        cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
        pts64 = np.asarray(points, dtype=np.float64)
        distances = np.linalg.norm(pts64 - cam.reshape(1, 3), axis=1)
        self._distance_key = key
        self._distances = distances
        return distances, False

    def candidate_mask(
        self,
        uv: np.ndarray,
        detection: dict,
        pano_w: float,
    ) -> tuple[np.ndarray, bool]:
        key = (self._projection_key, _points_cache_key(uv), _bbox_cache_key(detection), float(pano_w))
        cached = self._candidate_masks.get(key)
        if cached is not None:
            return cached, True
        mask = _frustum_candidate_mask_from_uv(uv, detection, float(pano_w))
        self._candidate_masks[key] = mask
        return mask, False


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
    diagnostics: dict[str, object] | None = None
    debug_masks: dict[str, np.ndarray] | None = None


def _empty(
    n: int,
    reason: str,
    diagnostics: dict[str, object] | None = None,
    debug_masks: dict[str, np.ndarray] | None = None,
) -> FusedSelection:
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
        diagnostics=diagnostics,
        debug_masks=debug_masks,
    )


def _debug_masks(
    *,
    bbox: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    final: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    if bbox is not None:
        masks["bbox"] = np.asarray(bbox, dtype=bool).copy()
    if depth is not None:
        masks["depth"] = np.asarray(depth, dtype=bool).copy()
    if final is not None:
        masks["final"] = np.asarray(final, dtype=bool).copy()
    return masks


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
    cache: FrustumProjectionCache | None = None,
) -> np.ndarray:
    """Points whose projection lands inside ``detection``'s bbox (its frustum).

    When ``seed_idx`` and ``depth_window`` are given the cone is additionally
    clipped to a depth shell around the seed so the far wall stacked behind the
    door in the same bbox is removed before growing.
    """
    cache_obj = cache if cache is not None else FrustumProjectionCache()
    uv, _ = cache_obj.project(
        np.asarray(points), cam_pos, R_pano, img_w, img_h, yaw_offset_deg
    )
    inside, _ = cache_obj.candidate_mask(uv, detection, float(img_w))
    if seed_idx is not None and depth_window is not None:
        dist, _ = cache_obj.distances(np.asarray(points), cam_pos)
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
    cache: FrustumProjectionCache | None = None,
) -> FusedSelection:
    """Extract a door/window region by fusing the clicked seed's frustum + geometry.

    Strategy A: uses ``detections`` from the currently selected keyframe. If the
    seed falls inside a detection frustum the constrained pure-point-cloud growth
    runs inside that frustum and inherits the detection label; otherwise the
    caller should fall back to the unconstrained pure-point-cloud path.
    """
    started = perf_counter()
    pts = np.asarray(points)
    n = len(pts)
    diagnostics: dict[str, object] = {
        "path": "seed",
        "point_count": int(n),
        "seed_index": int(seed_idx),
        "detection_count": int(len(detections)),
    }
    if seed_idx < 0 or seed_idx >= n:
        result = _empty(n, "seed_out_of_range", _finish_diagnostics(diagnostics, started))
        _log_fusion_result("fuse_detection_and_pointcloud", result)
        return result
    if not detections or img_w <= 0 or img_h <= 0:
        reason = "no_detections" if not detections else "invalid_image_size"
        result = _empty(n, reason, _finish_diagnostics(diagnostics, started))
        _log_fusion_result("fuse_detection_and_pointcloud", result)
        return result

    # 1. Which detection frustum contains the clicked seed? (smallest bbox wins,
    #    matching the 2D-guided path's nested-box preference).
    cache_obj = cache if cache is not None else FrustumProjectionCache()
    phase = perf_counter()
    uv, projection_cache_hit = cache_obj.project(
        pts, cam_pos, R_pano, img_w, img_h, yaw_offset_deg
    )
    diagnostics["projection_ms"] = _elapsed_ms(phase)
    diagnostics["projection_cache_hit"] = bool(projection_cache_hit)

    matches = match_points_to_detections(uv, detections, float(img_w))
    det_idx = int(matches.match_indices[seed_idx])
    if det_idx < 0:
        result = _empty(n, "seed_not_in_any_frustum", _finish_diagnostics(diagnostics, started))
        _log_fusion_result("fuse_detection_and_pointcloud", result)
        return result

    detection = detections[det_idx]
    label = str(detection.get("label", "")).lower() or None
    detection_score = float(detection["score"]) if "score" in detection else None

    # 2. Frustum candidate set for that detection, depth-clipped around the seed.
    phase = perf_counter()
    candidate, candidate_cache_hit = cache_obj.candidate_mask(uv, detection, float(img_w))
    distances, distance_cache_hit = cache_obj.distances(pts, cam_pos)
    clipped_candidate = _depth_clipped_candidate_mask(
        candidate,
        distances,
        seed_idx,
        frustum_depth_window,
    )
    diagnostics.update({
        "detection_index": int(det_idx),
        "candidate_ms": _elapsed_ms(phase),
        "candidate_cache_hit": bool(candidate_cache_hit),
        "distance_cache_hit": bool(distance_cache_hit),
        "bbox_candidate_count": int(candidate.sum()),
        "selected_depth_candidate_count": int(clipped_candidate.sum()),
        "seed_attempt_count": 1,
    })

    # 3 + 4. Constrained planar growth + geometry scoring, with the 2D label as
    #        the size-check hint. Growth is bounded by the frustum candidate set,
    #        so a wall-flush door no longer spreads across the whole wall.
    phase = perf_counter()
    region: PlanarRegionSelection = extract_planar_region_from_seed(
        pts,
        seed_idx,
        candidate_mask=clipped_candidate,
        label_hint=label,
    )
    diagnostics["extract_ms"] = _elapsed_ms(phase)

    # 5. Fuse the confidences. The frustum already agreed (seed is inside it), so
    #    a high-confidence geometry means both judges concur.
    selection = _selection_from_region(
        region,
        detection_index=det_idx,
        label_hint=label,
        score=detection_score,
    )
    selection = _clip_selection_to_detection_bbox(
        selection,
        uv=uv,
        detection=detection,
        pano_w=float(img_w),
        inside_mask=candidate,
    )
    diagnostics["final_point_count"] = int(selection.point_count)
    result = replace(
        selection,
        diagnostics=_finish_diagnostics(diagnostics, started),
        debug_masks=_debug_masks(bbox=candidate, depth=clipped_candidate, final=selection.mask),
    )
    _log_fusion_result("fuse_detection_and_pointcloud", result)
    return result


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
    cache: FrustumProjectionCache | None = None,
) -> FusedSelection:
    """Extract a detection's best door/window region without a manually clicked seed."""
    started = perf_counter()
    pts = np.asarray(points)
    n = len(pts)
    diagnostics: dict[str, object] = {
        "path": "bbox",
        "point_count": int(n),
        "detection_index": int(detection_index),
        "detection_count": int(len(detections)),
    }
    if detection_index < 0 or detection_index >= len(detections):
        result = _empty(n, "detection_index_out_of_range", _finish_diagnostics(diagnostics, started))
        _log_fusion_result("extract_detection_region_from_bbox", result)
        return result
    if img_w <= 0 or img_h <= 0:
        result = _empty(n, "invalid_image_size", _finish_diagnostics(diagnostics, started))
        _log_fusion_result("extract_detection_region_from_bbox", result)
        return result

    detection = detections[detection_index]
    label = str(detection.get("label", "")).lower() or None
    detection_score = float(detection["score"]) if "score" in detection else None
    cache_obj = cache if cache is not None else FrustumProjectionCache()

    phase = perf_counter()
    uv, projection_cache_hit = cache_obj.project(
        pts, cam_pos, R_pano, img_w, img_h, yaw_offset_deg
    )
    diagnostics["projection_ms"] = _elapsed_ms(phase)
    diagnostics["projection_cache_hit"] = bool(projection_cache_hit)

    phase = perf_counter()
    distances, distance_cache_hit = cache_obj.distances(pts, cam_pos)
    diagnostics["distance_ms"] = _elapsed_ms(phase)
    diagnostics["distance_cache_hit"] = bool(distance_cache_hit)

    target_uv = _valid_click_uv(click_uv, float(img_w))
    phase = perf_counter()
    candidate, candidate_cache_hit = cache_obj.candidate_mask(uv, detection, float(img_w))
    diagnostics["candidate_ms"] = _elapsed_ms(phase)
    diagnostics["candidate_cache_hit"] = bool(candidate_cache_hit)
    diagnostics["bbox_candidate_count"] = int(candidate.sum())
    candidate_indices = np.flatnonzero(candidate)
    if len(candidate_indices) == 0:
        result = _empty(n, "bbox_no_points", _finish_diagnostics(diagnostics, started))
        _log_fusion_result("extract_detection_region_from_bbox", result)
        return result

    best: FusedSelection | None = None
    selected_seed_idx: int | None = None
    selected_depth_candidate_count = 0
    selected_depth_candidate_mask: np.ndarray | None = None
    depth_candidate_counts: list[int] = []
    extract_ms = 0.0
    seed_indices = _auto_seed_indices(
        pts,
        candidate_indices,
        cam_pos,
        max_seed_count=max_seed_count,
        uv=uv,
        click_uv=target_uv,
        pano_w=float(img_w),
    )
    diagnostics["auto_seed_count"] = int(len(seed_indices))

    for seed_idx in seed_indices:
        clipped_candidate = _depth_clipped_candidate_mask(
            candidate,
            distances,
            int(seed_idx),
            DEFAULT_FRUSTUM_DEPTH_WINDOW,
        )
        depth_candidate_count = int(clipped_candidate.sum())
        depth_candidate_counts.append(depth_candidate_count)
        phase = perf_counter()
        region = extract_planar_region_from_seed(
            pts,
            int(seed_idx),
            candidate_mask=clipped_candidate,
            label_hint=label,
        )
        extract_ms += _elapsed_ms(phase)
        selection = _selection_from_region(
            region,
            detection_index=int(detection_index),
            label_hint=label,
            score=detection_score,
        )
        selection = _clip_selection_to_detection_bbox(
            selection,
            uv=uv,
            detection=detection,
            pano_w=float(img_w),
            inside_mask=candidate,
        )
        selection_score = _fused_selection_score(
            selection,
            uv=uv,
            click_uv=target_uv,
            pano_w=float(img_w),
            detection=detection,
            distances=distances,
            bbox_inside_mask=candidate,
        )
        best_score = None if best is None else _fused_selection_score(
            best,
            uv=uv,
            click_uv=target_uv,
            pano_w=float(img_w),
            detection=detection,
            distances=distances,
            bbox_inside_mask=candidate,
        )
        if best_score is None or selection_score > best_score:
            best = selection
            selected_seed_idx = int(seed_idx)
            selected_depth_candidate_count = depth_candidate_count
            selected_depth_candidate_mask = clipped_candidate

    diagnostics["seed_attempt_count"] = int(len(depth_candidate_counts))
    diagnostics["extract_ms"] = round(float(extract_ms), 3)
    if depth_candidate_counts:
        diagnostics["depth_candidate_min"] = int(min(depth_candidate_counts))
        diagnostics["depth_candidate_max"] = int(max(depth_candidate_counts))
        diagnostics["selected_depth_candidate_count"] = int(selected_depth_candidate_count)
    if selected_seed_idx is not None:
        diagnostics["selected_seed_index"] = int(selected_seed_idx)

    if best is None or best.point_count <= 0:
        result = _empty(
            n,
            "bbox_no_extractable_region",
            _finish_diagnostics(diagnostics, started),
            debug_masks=_debug_masks(bbox=candidate, depth=selected_depth_candidate_mask),
        )
        _log_fusion_result("extract_detection_region_from_bbox", result)
        return result
    diagnostics["final_point_count"] = int(best.point_count)
    result = replace(
        best,
        diagnostics=_finish_diagnostics(diagnostics, started),
        debug_masks=_debug_masks(bbox=candidate, depth=selected_depth_candidate_mask, final=best.mask),
    )
    _log_fusion_result("extract_detection_region_from_bbox", result)
    return result


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
    detection: dict | None = None,
    distances: np.ndarray | None = None,
    bbox_inside_mask: np.ndarray | None = None,
) -> tuple[int, ...]:
    confidence_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}.get(selection.confidence, 0)
    reason_rank = 1 if selection.reason == "fused_frustum_and_planar_geometry" else 0
    label_rank = 1 if selection.label in {"door", "window"} else 0
    click_center_rank, click_spread_rank = _selection_click_ranks(
        selection, uv=uv, click_uv=click_uv, pano_w=pano_w
    )
    bbox_inside_rank, bbox_center_rank, bbox_spread_rank = _selection_bbox_ranks(
        selection, uv=uv, detection=detection, pano_w=pano_w, inside_mask=bbox_inside_mask
    )
    depth_rank = _selection_depth_rank(selection, distances=distances)
    return (
        confidence_rank,
        reason_rank,
        label_rank,
        click_center_rank,
        click_spread_rank,
        depth_rank,
        bbox_inside_rank,
        bbox_center_rank,
        bbox_spread_rank,
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


def _clip_selection_to_detection_bbox(
    selection: FusedSelection,
    *,
    uv: np.ndarray,
    detection: dict,
    pano_w: float,
    inside_mask: np.ndarray | None = None,
) -> FusedSelection:
    mask = np.asarray(selection.mask, dtype=bool)
    if len(mask) != len(uv) or not mask.any():
        return selection
    if inside_mask is not None and len(inside_mask) == len(uv):
        inside = np.asarray(inside_mask, dtype=bool)
    else:
        inside = _frustum_candidate_mask_from_uv(uv, detection, float(pano_w))
    clipped = mask & inside
    if np.array_equal(clipped, mask):
        return selection
    return replace(selection, mask=clipped, point_count=int(clipped.sum()))


def _selection_bbox_ranks(
    selection: FusedSelection,
    *,
    uv: np.ndarray | None,
    detection: dict | None,
    pano_w: float | None,
    inside_mask: np.ndarray | None = None,
) -> tuple[int, int, int]:
    if uv is None or detection is None or pano_w is None:
        return 0, 0, 0
    mask = np.asarray(selection.mask, dtype=bool)
    if len(mask) != len(uv) or not mask.any():
        return 0, 0, 0
    if inside_mask is not None and len(inside_mask) == len(uv):
        inside = np.asarray(inside_mask, dtype=bool)
    else:
        inside = _frustum_candidate_mask_from_uv(uv, detection, float(pano_w))
    inside_ratio = float((mask & inside).sum()) / float(mask.sum())
    center_uv = _bbox_center_uv(detection["bbox"], float(pano_w))
    distances = _panorama_pixel_distances(uv[mask], center_uv, float(pano_w))
    median_dist = float(np.median(distances))
    spread_dist = float(np.percentile(distances, 90))
    return (
        int(round(inside_ratio * 1000.0)),
        -int(round(median_dist * 1000.0)),
        -int(round(spread_dist * 100.0)),
    )


def _selection_depth_rank(
    selection: FusedSelection,
    *,
    distances: np.ndarray | None,
) -> int:
    if distances is None:
        return 0
    mask = np.asarray(selection.mask, dtype=bool)
    if len(mask) != len(distances) or not mask.any():
        return 0
    median_depth = float(np.median(np.asarray(distances, dtype=np.float64)[mask]))
    return -int(round(median_depth * 1000.0))


def _bbox_center_uv(bbox: Sequence[float], pano_w: float) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if pano_w > 0 and x2 < x1:
        width = (pano_w - x1) + x2
        u = (x1 + width / 2.0) % pano_w
    else:
        u = (x1 + x2) / 2.0
    return u, (y1 + y2) / 2.0


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
    reason = str(selection.reason)
    if selection.point_count <= 0:
        return False
    if str(selection.confidence) != "high":
        return False
    if reason.startswith("rejected_") or reason.startswith("frustum_only_") or "too_few" in reason:
        return False
    return True
