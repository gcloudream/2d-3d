"""3D point-cloud refinement for door/window bbox selections."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from core.door_window import match_points_to_detections


DEFAULT_COMPONENT_RADIUS = 0.22
DEFAULT_DEPTH_DELTA = 0.35
DEFAULT_MIN_REFINED_POINTS = 80
DEFAULT_MIN_GEOMETRY_POINTS = 20
DEFAULT_MAX_PLANE_RMS_ERROR = 0.08
DEFAULT_VERTICAL_NORMAL_Z_MAX = 0.35
DEFAULT_PLANE_INLIER_THRESHOLD = 0.06
DEFAULT_RANSAC_ITERATIONS = 120
DEFAULT_MIN_PLANE_INLIER_RATIO = 0.45
DEFAULT_MIN_RECTANGULAR_COVERAGE = 0.20
# Final highlight is thinned to a tight band around the fitted door/window
# plane. The RANSAC inlier threshold above stays loose for a robust fit, but the
# displayed set must be a thin sheet: a thick slab looks coherent when viewed
# from the panorama center (down the line of sight) yet fans out into apparent
# "scatter" in the oblique global point-cloud view.
DEFAULT_PLANE_DISPLAY_THICKNESS = 0.04
# Step 1 — depth-cone clip. A door/window bbox is an angular window, so the
# matched points form a cone that can be many metres deep (front object + far
# wall in the same box). Before growing a component we clip that cone to a slab
# around the clicked seed's camera distance so growth stays on the seed surface.
DEFAULT_CONE_DEPTH_WINDOW = 0.6
# Step 2 — seed-plane thinning. After the component is grown we keep only points
# within this perpendicular distance of the seed-local plane. Unlike a
# camera-distance shell this actually flattens the depth thickness along a wall.
DEFAULT_SEED_PLANE_DELTA = 0.08
DEFAULT_SEED_PLANE_FIT_RADIUS = 0.6
# Step 3 — normal-consistency prune. Local surface normals stop the component
# from bleeding across the door frame onto the floor / ceiling / side wall,
# which is what inflates the measured width past the door size limits.
DEFAULT_NORMAL_RADIUS = 0.25
DEFAULT_MAX_NORMAL_ANGLE_DEG = 35.0
DEFAULT_NORMAL_MIN_NEIGHBORS = 6
# Final step — isolated-point removal (statistical outlier removal). Points that
# survive the bbox / plane-distance / normal filters can still be off the
# door/window body: stragglers that are coplanar with the same wall and within
# the angular box but spatially disconnected from the body. They read as random
# "scatter" in the oblique global view. A point is dropped when it has fewer
# than ``MIN_NEIGHBORS`` neighbours within ``RADIUS``.
DEFAULT_ISOLATION_RADIUS = 0.10
DEFAULT_ISOLATION_MIN_NEIGHBORS = 4
# Density-adaptive isolation: a point is kept when its companion count is at
# least this fraction of the selection's median companion count (capped by
# MIN_NEIGHBORS). This auto-scales the outlier test to the cloud's sampling
# density so a uniformly sparse but coherent body is not wiped out.
DEFAULT_ISOLATION_DENSITY_FRACTION = 0.35

DIMENSION_LIMITS = {
    "door": (0.5, 1.5, 1.6, 2.6),
    "window": (0.3, 3.5, 0.3, 2.2),
}


@dataclass(frozen=True)
class PlaneFit:
    point: np.ndarray
    normal: np.ndarray
    rms_error: float
    inlier_mask: np.ndarray | None = None
    inlier_ratio: float = 1.0


@dataclass(frozen=True)
class GeometryScore:
    confidence: str
    reason: str
    plane_point: np.ndarray | None
    plane_normal: np.ndarray | None
    width_m: float | None
    height_m: float | None
    plane_rms_error: float | None
    inlier_ratio: float | None = None
    rectangular_coverage: float | None = None
    inlier_mask: np.ndarray | None = None


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


def clip_bbox_cone_by_seed_depth(
    points: np.ndarray,
    candidate_mask: np.ndarray,
    seed_idx: int,
    cam_pos: np.ndarray,
    depth_window: float = DEFAULT_CONE_DEPTH_WINDOW,
) -> np.ndarray:
    """Clip the bbox-matched cone to a depth slab around the seed.

    The 2D bbox only constrains viewing angle, so ``candidate_mask`` is a cone
    that can contain several surfaces stacked in depth (a near object and the
    far wall behind it). Keeping only points whose camera distance is within
    ``depth_window`` of the clicked seed removes those other surfaces before
    region growing, so growth cannot hop onto them.
    """
    points64 = np.asarray(points, dtype=np.float64)
    candidate = np.asarray(candidate_mask, dtype=bool)
    if len(points64) != len(candidate):
        raise ValueError(f"points length {len(points64)} != candidate mask length {len(candidate)}")
    if seed_idx < 0 or seed_idx >= len(points64) or not candidate[seed_idx]:
        return np.zeros(len(points64), dtype=bool)

    cam = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    distances = np.linalg.norm(points64 - cam.reshape(1, 3), axis=1)
    seed_depth = float(distances[seed_idx])
    within = np.abs(distances - seed_depth) <= float(depth_window)
    return candidate & within


def estimate_seed_plane(
    points: np.ndarray,
    mask: np.ndarray,
    seed_idx: int,
    radius: float = DEFAULT_SEED_PLANE_FIT_RADIUS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit a local plane to masked points near the seed.

    Returns ``(point_on_plane, unit_normal)`` or ``None`` when there are not
    enough neighbours to fit a plane. The plane is anchored at the seed so the
    perpendicular-distance filter is centred on the surface the user clicked.
    """
    points64 = np.asarray(points, dtype=np.float64)
    sel = np.asarray(mask, dtype=bool)
    if seed_idx < 0 or seed_idx >= len(points64) or not sel[seed_idx]:
        return None

    indices = np.flatnonzero(sel)
    tree = cKDTree(points64[indices])
    local = tree.query_ball_point(points64[seed_idx], r=float(radius))
    if len(local) < 3:
        local = list(range(len(indices)))
    neighbours = points64[indices[np.asarray(local, dtype=np.int64)]]
    if len(neighbours) < 3:
        return None
    try:
        plane = fit_plane_least_squares(neighbours)
    except ValueError:
        return None
    return points64[seed_idx].copy(), plane.normal


def filter_by_seed_plane(
    points: np.ndarray,
    component_mask: np.ndarray,
    seed_idx: int,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    max_delta: float = DEFAULT_SEED_PLANE_DELTA,
) -> np.ndarray:
    """Keep component points within ``max_delta`` of the seed-local plane.

    This replaces the camera-distance shell for flattening the selection: for a
    wall facing the camera the shell barely thins anything, whereas the
    perpendicular plane distance directly removes the off-plane wall halo.
    """
    points64 = np.asarray(points, dtype=np.float64)
    component = np.asarray(component_mask, dtype=bool)
    if len(points64) != len(component):
        raise ValueError(f"points length {len(points64)} != component mask length {len(component)}")
    if seed_idx < 0 or seed_idx >= len(points64) or not component[seed_idx]:
        return np.zeros(len(points64), dtype=bool)

    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return component.copy()
    normal = normal / norm
    offsets = np.abs((points64 - np.asarray(plane_point, dtype=np.float64).reshape(1, 3)) @ normal)
    return component & (offsets <= float(max_delta))


def filter_by_normal_consistency(
    points: np.ndarray,
    component_mask: np.ndarray,
    seed_idx: int,
    reference_normal: np.ndarray,
    radius: float = DEFAULT_NORMAL_RADIUS,
    max_angle_deg: float = DEFAULT_MAX_NORMAL_ANGLE_DEG,
    min_neighbors: int = DEFAULT_NORMAL_MIN_NEIGHBORS,
) -> np.ndarray:
    """Drop component points whose local surface normal disagrees with the door.

    Estimates each point's normal from its neighbours (inside the component) and
    keeps points whose normal is within ``max_angle_deg`` of ``reference_normal``.
    This stops the selection from bleeding across the door frame onto the
    perpendicular floor / ceiling / return wall. Points with too few neighbours
    to estimate a normal are kept (sparse door edges should not be discarded).
    """
    points64 = np.asarray(points, dtype=np.float64)
    component = np.asarray(component_mask, dtype=bool)
    if len(points64) != len(component):
        raise ValueError(f"points length {len(points64)} != component mask length {len(component)}")
    if seed_idx < 0 or seed_idx >= len(points64) or not component[seed_idx]:
        return np.zeros(len(points64), dtype=bool)

    ref = np.asarray(reference_normal, dtype=np.float64).reshape(3)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm == 0.0:
        return component.copy()
    ref = ref / ref_norm

    indices = np.flatnonzero(component)
    if len(indices) < 3:
        return component.copy()
    comp_points = points64[indices]
    tree = cKDTree(comp_points)
    cos_limit = float(np.cos(np.deg2rad(float(max_angle_deg))))

    keep_local = np.ones(len(indices), dtype=bool)
    neighbor_lists = tree.query_ball_point(comp_points, r=float(radius))
    for i, neigh in enumerate(neighbor_lists):
        if len(neigh) < int(min_neighbors):
            continue  # too sparse to trust a normal -> keep
        local_pts = comp_points[np.asarray(neigh, dtype=np.int64)]
        centered = local_pts - local_pts.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        nrm = float(np.linalg.norm(normal))
        if nrm == 0.0:
            continue
        normal = normal / nrm
        if abs(float(normal @ ref)) < cos_limit:
            keep_local[i] = False

    # never drop the seed itself
    seed_local = int(np.searchsorted(indices, seed_idx))
    if 0 <= seed_local < len(indices) and indices[seed_local] == seed_idx:
        keep_local[seed_local] = True

    result = np.zeros(len(points64), dtype=bool)
    result[indices[keep_local]] = True
    return result


def remove_isolated_points(
    points: np.ndarray,
    mask: np.ndarray,
    seed_idx: int | None = None,
    radius: float = DEFAULT_ISOLATION_RADIUS,
    min_neighbors: int = DEFAULT_ISOLATION_MIN_NEIGHBORS,
    density_fraction: float = DEFAULT_ISOLATION_DENSITY_FRACTION,
) -> np.ndarray:
    """Drop spatially isolated points from a selection (statistical outlier removal).

    A point is kept when it has enough companions within ``radius``. The keep
    threshold is density-adaptive: ``max(min_neighbors, density_fraction * median
    companions)`` capped so it never exceeds the body's own median. This removes
    stragglers that pass the bbox / plane-distance / normal filters yet are
    disconnected from the door/window body (the dots that look like random
    scatter in the oblique global view), while adapting to the cloud's sampling
    density so a sparse-but-coherent body is not wiped out.

    The seed point (if given and selected) is always kept. When the selection is
    too small to judge density it is returned unchanged.
    """
    pts = np.asarray(points, dtype=np.float64)
    sel = np.asarray(mask, dtype=bool)
    if len(pts) != len(sel):
        raise ValueError(f"points length {len(pts)} != mask length {len(sel)}")

    indices = np.flatnonzero(sel)
    if len(indices) <= int(min_neighbors):
        return sel.copy()

    sel_points = pts[indices]
    tree = cKDTree(sel_points)
    # query_ball_point includes the point itself, so subtract 1 for companions.
    companions = tree.query_ball_point(sel_points, r=float(radius), return_length=True) - 1

    median_companions = float(np.median(companions))
    # If even typical body points are sparse at this radius, we cannot reliably
    # tell outliers from a uniformly sparse (but coherent) body — skip removal.
    if median_companions < float(min_neighbors):
        return sel.copy()

    # Adaptive threshold: a point is isolated if it has far fewer companions than
    # the typical body point. Never demand more than the body's own median.
    threshold = min(
        float(min_neighbors),
        max(1.0, float(density_fraction) * median_companions),
    )
    keep_local = companions >= threshold

    if seed_idx is not None and 0 <= seed_idx < len(pts) and sel[seed_idx]:
        seed_local = int(np.searchsorted(indices, seed_idx))
        if 0 <= seed_local < len(indices) and indices[seed_local] == seed_idx:
            keep_local[seed_local] = True

    if not keep_local.any():
        return sel.copy()  # never return empty due to over-aggressive pruning

    result = np.zeros(len(pts), dtype=bool)
    result[indices[keep_local]] = True
    return result


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


def _plane_from_three_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    normal = normal / norm
    if normal[2] < 0:
        normal = -normal
    return a, normal


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float = DEFAULT_PLANE_INLIER_THRESHOLD,
    iterations: int = DEFAULT_RANSAC_ITERATIONS,
) -> PlaneFit:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    n = len(pts)
    if n < 3:
        raise ValueError("at least 3 points are required to fit a plane")

    rng = np.random.default_rng(0)
    best_mask: np.ndarray | None = None
    best_count = -1
    best_rms = float("inf")

    for _ in range(int(iterations)):
        sample = rng.choice(n, size=3, replace=False)
        candidate = _plane_from_three_points(pts[sample[0]], pts[sample[1]], pts[sample[2]])
        if candidate is None:
            continue
        plane_point, normal = candidate
        distances = np.abs((pts - plane_point.reshape(1, 3)) @ normal)
        mask = distances <= float(threshold)
        count = int(mask.sum())
        if count < 3:
            continue
        rms = float(np.sqrt(np.mean(distances[mask] * distances[mask])))
        if count > best_count or (count == best_count and rms < best_rms):
            best_mask = mask
            best_count = count
            best_rms = rms

    if best_mask is None:
        raise ValueError("cannot fit a plane to degenerate points")

    first_refit = fit_plane_least_squares(pts[best_mask])
    distances = np.abs((pts - first_refit.point.reshape(1, 3)) @ first_refit.normal)
    inlier_mask = distances <= float(threshold)
    final_refit = fit_plane_least_squares(pts[inlier_mask])
    return PlaneFit(
        point=final_refit.point,
        normal=final_refit.normal,
        rms_error=final_refit.rms_error,
        inlier_mask=inlier_mask,
        inlier_ratio=float(inlier_mask.sum()) / float(n),
    )


def _plane_axes(normal: np.ndarray, points: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n_norm = float(np.linalg.norm(n))
    if n_norm == 0.0:
        raise ValueError("normal must be non-zero")
    n = n / n_norm

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    height_axis = z_axis - float(np.dot(z_axis, n)) * n
    height_norm = float(np.linalg.norm(height_axis))
    if height_norm < 1e-6:
        if points is None:
            raise ValueError("points are required when the plane is horizontal")
        pts = np.asarray(points, dtype=np.float64)
        centered = pts - pts.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_a = vh[0]
        axis_b = vh[1]
    else:
        axis_b = height_axis / height_norm
        axis_a = np.cross(axis_b, n)
        axis_a = axis_a / float(np.linalg.norm(axis_a))
    return axis_a, axis_b


def _planar_coordinates(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    axis_a, axis_b = _plane_axes(normal, pts)
    return np.column_stack([pts @ axis_a, pts @ axis_b])


def estimate_planar_extent(points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    coords = _planar_coordinates(points, normal)
    coord_a = coords[:, 0]
    coord_b = coords[:, 1]
    width_m = float(coord_a.max() - coord_a.min())
    height_m = float(coord_b.max() - coord_b.min())
    return width_m, height_m


def rectangular_coverage(points: np.ndarray, normal: np.ndarray) -> float:
    coords = _planar_coordinates(points, normal)
    width_m = float(coords[:, 0].max() - coords[:, 0].min())
    height_m = float(coords[:, 1].max() - coords[:, 1].min())
    bbox_area = width_m * height_m
    if len(coords) < 3 or bbox_area <= 1e-9:
        return 0.0
    try:
        hull = ConvexHull(coords)
    except QhullError:
        return 0.0
    return float(hull.volume) / bbox_area


def score_door_window_geometry(
    points: np.ndarray,
    label: str,
    min_points: int = DEFAULT_MIN_GEOMETRY_POINTS,
    max_plane_rms_error: float = DEFAULT_MAX_PLANE_RMS_ERROR,
    vertical_normal_z_max: float = DEFAULT_VERTICAL_NORMAL_Z_MAX,
    min_plane_inlier_ratio: float = DEFAULT_MIN_PLANE_INLIER_RATIO,
    min_rectangular_coverage: float = DEFAULT_MIN_RECTANGULAR_COVERAGE,
) -> GeometryScore:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < int(min_points):
        return GeometryScore("low", "rejected_too_few_geometry_points", None, None, None, None, None)

    try:
        plane = fit_plane_ransac(pts)
    except ValueError:
        try:
            plane = fit_plane_least_squares(pts)
        except ValueError:
            return GeometryScore("low", "rejected_plane_fit_failed", None, None, None, None, None)

    inlier_mask = plane.inlier_mask if plane.inlier_mask is not None else np.ones(len(pts), dtype=bool)
    inlier_points = pts[inlier_mask]
    width_m, height_m = estimate_planar_extent(inlier_points, plane.normal)
    coverage = rectangular_coverage(inlier_points, plane.normal)
    if plane.inlier_ratio < float(min_plane_inlier_ratio):
        return GeometryScore(
            "low",
            "rejected_low_plane_inlier_ratio",
            plane.point,
            plane.normal,
            width_m,
            height_m,
            plane.rms_error,
            plane.inlier_ratio,
            coverage,
            inlier_mask,
        )

    if plane.rms_error > float(max_plane_rms_error):
        return GeometryScore(
            "low",
            "rejected_not_planar",
            plane.point,
            plane.normal,
            width_m,
            height_m,
            plane.rms_error,
            plane.inlier_ratio,
            coverage,
            inlier_mask,
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
            plane.inlier_ratio,
            coverage,
            inlier_mask,
        )

    if coverage < float(min_rectangular_coverage):
        return GeometryScore(
            "low",
            "rejected_low_rectangular_coverage",
            plane.point,
            plane.normal,
            width_m,
            height_m,
            plane.rms_error,
            plane.inlier_ratio,
            coverage,
            inlier_mask,
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
                plane.inlier_ratio,
                coverage,
                inlier_mask,
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
        plane.inlier_ratio,
        coverage,
        inlier_mask,
    )


def should_highlight_refined_selection(selection: RefinedDoorWindowSelection) -> bool:
    return selection.point_count > 0 and not (
        selection.confidence == "low" and selection.reason.startswith("rejected_")
    )


def select_plane_band(
    points: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    thickness: float = DEFAULT_PLANE_DISPLAY_THICKNESS,
) -> np.ndarray:
    """Mask points within ``thickness`` metres of the fitted plane.

    Used to thin the displayed door/window selection into a single sheet so it
    no longer fans out (apparent "scatter") when seen from the oblique global
    point-cloud camera.
    """
    pts = np.asarray(points, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return np.ones(len(pts), dtype=bool)
    normal = normal / norm
    offsets = np.abs((pts - np.asarray(plane_point, dtype=np.float64).reshape(1, 3)) @ normal)
    return offsets <= float(thickness)


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
    cone_depth_window: float = DEFAULT_CONE_DEPTH_WINDOW,
    seed_plane_delta: float = DEFAULT_SEED_PLANE_DELTA,
    normal_max_angle_deg: float = DEFAULT_MAX_NORMAL_ANGLE_DEG,
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
    # Step 1: clip the deep bbox cone to a depth slab around the clicked seed so
    # region growing cannot hop onto a different surface stacked in the same box.
    cone_mask = clip_bbox_cone_by_seed_depth(
        points_arr,
        coarse_mask,
        clicked_idx,
        cam_pos=cam_pos,
        depth_window=cone_depth_window,
    )
    component_mask = connected_component_from_seed(
        points_arr,
        cone_mask,
        clicked_idx,
        radius=component_radius,
    )
    # Step 2: thin along the seed-local plane normal (flattens the wall halo that
    # a camera-distance shell leaves behind), with the shell kept as a fallback.
    seed_plane = estimate_seed_plane(points_arr, component_mask, clicked_idx)
    if seed_plane is not None:
        seed_plane_point, seed_plane_normal = seed_plane
        depth_mask = filter_by_seed_plane(
            points_arr,
            component_mask,
            clicked_idx,
            seed_plane_point,
            seed_plane_normal,
            max_delta=seed_plane_delta,
        )
        # Step 3: prune points whose local normal disagrees with the seed plane,
        # so the component does not bleed across the frame onto floor/ceiling.
        depth_mask = filter_by_normal_consistency(
            points_arr,
            depth_mask,
            clicked_idx,
            seed_plane_normal,
            max_angle_deg=normal_max_angle_deg,
        )
    else:
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
        depth_indices = np.flatnonzero(depth_mask)
        geometry = score_door_window_geometry(points_arr[depth_indices], str(det.get("label", "")))
        confidence = geometry.confidence
        reason = geometry.reason
        plane_point = geometry.plane_point
        plane_normal = geometry.plane_normal
        width_m = geometry.width_m
        height_m = geometry.height_m
        plane_rms_error = geometry.plane_rms_error
        if confidence == "high" and geometry.inlier_mask is not None:
            refined_mask = np.zeros(n, dtype=bool)
            refined_mask[depth_indices[np.asarray(geometry.inlier_mask, dtype=bool)]] = True
            # Thin the accepted set to a tight band around the fitted plane so the
            # highlight is a single sheet. The RANSAC inlier band (~0.06 m) keeps
            # off-plane stragglers that look fine down the panorama line of sight
            # but fan out into apparent scatter in the oblique global view.
            if plane_point is not None and plane_normal is not None:
                band = select_plane_band(points_arr, plane_point, plane_normal)
                thinned = refined_mask & band
                if thinned.any():
                    refined_mask = thinned
            # Drop spatially isolated stragglers that survive the plane band but
            # sit disconnected from the door/window body (apparent scatter).
            refined_mask = remove_isolated_points(points_arr, refined_mask, clicked_idx)
            depth_mask = refined_mask
            point_count = int(depth_mask.sum())

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
