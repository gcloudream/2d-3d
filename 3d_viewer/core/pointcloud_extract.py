"""Pure point-cloud door/window extraction (no 2D detection required).

The 2D-guided path in ``door_window_refine`` uses a panorama detection box as a
coarse region generator. This module does the opposite: the user clicks a point
directly in the global point-cloud view and we grow the connected *coplanar*
region around it, then score it as a door/window-like vertical planar patch.

It needs no panorama, camera pose, or detection box — only the world points and
the clicked index. All geometry primitives are reused from
``door_window_refine`` so the two paths stay consistent.

Limitation by design: a door that is perfectly flush with its wall is coplanar
with the wall, so pure geometry cannot separate them. The bounded growth extent
keeps the selection local, and the plane-distance threshold separates any door
panel that is recessed or protruding from the wall (most real doors are).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.door_window_refine import (
    DIMENSION_LIMITS,
    connected_component_from_seed,
    filter_by_normal_consistency,
    fit_plane_least_squares,
    remove_isolated_points,
    score_door_window_geometry,
    select_plane_band,
)


# Radius of the seed neighbourhood used to estimate the local surface plane.
DEFAULT_SEED_NEIGHBOR_RADIUS = 0.25
# Connection radius for region growing across the planar surface.
DEFAULT_CONNECT_RADIUS = 0.15
# Perpendicular distance to the seed plane a point may have to be a candidate.
DEFAULT_PLANE_DELTA = 0.05
# Bounding sphere radius around the seed. This caps how far growth can spread
# across a flat surface. A door/window diagonal is ~2.3-3 m, so ~2 m keeps the
# region at door/window scale; without it a click on a door flush with its wall
# would grow across the entire coplanar wall (see module docstring limitation).
DEFAULT_MAX_EXTENT = 2.0
# Local-normal disagreement that prunes a point off the door/window plane.
DEFAULT_NORMAL_MAX_ANGLE_DEG = 30.0
# Minimum grown points before the region is worth scoring.
DEFAULT_MIN_REGION_POINTS = 40
# Final display band around the fitted plane (matches the 2D-guided path).
DEFAULT_DISPLAY_THICKNESS = 0.04


@dataclass(frozen=True)
class PlanarRegionSelection:
    mask: np.ndarray
    point_count: int
    confidence: str
    reason: str
    label: str
    plane_point: np.ndarray | None
    plane_normal: np.ndarray | None
    width_m: float | None
    height_m: float | None


def _empty(n: int, reason: str) -> PlanarRegionSelection:
    return PlanarRegionSelection(
        mask=np.zeros(n, dtype=bool),
        point_count=0,
        confidence="none",
        reason=reason,
        label="",
        plane_point=None,
        plane_normal=None,
        width_m=None,
        height_m=None,
    )


def infer_label_from_dimensions(width_m: float | None, height_m: float | None) -> str:
    """Classify a planar patch as door/window/object from its extent."""
    if width_m is None or height_m is None:
        return "object"
    for label, (min_w, max_w, min_h, max_h) in DIMENSION_LIMITS.items():
        if min_w <= width_m <= max_w and min_h <= height_m <= max_h:
            return label
    return "object"


def should_highlight_planar_region(selection: PlanarRegionSelection) -> bool:
    return selection.point_count > 0 and not (
        selection.confidence == "low" and selection.reason.startswith("rejected_")
    )


def extract_planar_region_from_seed(
    points: np.ndarray,
    seed_idx: int,
    *,
    candidate_mask: np.ndarray | None = None,
    label_hint: str | None = None,
    seed_neighbor_radius: float = DEFAULT_SEED_NEIGHBOR_RADIUS,
    connect_radius: float = DEFAULT_CONNECT_RADIUS,
    plane_delta: float = DEFAULT_PLANE_DELTA,
    max_extent: float = DEFAULT_MAX_EXTENT,
    normal_max_angle_deg: float = DEFAULT_NORMAL_MAX_ANGLE_DEG,
    min_region_points: int = DEFAULT_MIN_REGION_POINTS,
    display_thickness: float = DEFAULT_DISPLAY_THICKNESS,
) -> PlanarRegionSelection:
    """Grow and score a coplanar region around ``seed_idx``.

    Pure 3D when ``candidate_mask`` is ``None``. When a ``candidate_mask`` is
    supplied (e.g. the points inside a 2D detection frustum) region growing is
    restricted to it, which keeps a click on a wall-flush door from spreading
    across the whole coplanar wall — the fusion path uses this.

    ``label_hint`` (``"door"`` / ``"window"``) makes the geometry size check use
    that class's dimension limits instead of guessing from the extent.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if seed_idx < 0 or seed_idx >= n:
        return _empty(n, "seed_out_of_range")

    constraint = None
    if candidate_mask is not None:
        constraint = np.asarray(candidate_mask, dtype=bool)
        if len(constraint) != n:
            raise ValueError(f"candidate_mask length {len(constraint)} != points length {n}")
        if not constraint[seed_idx]:
            return _empty(n, "seed_outside_candidate_mask")

    seed = pts[seed_idx]
    dist_to_seed = np.linalg.norm(pts - seed.reshape(1, 3), axis=1)

    # 1. Estimate the local surface plane from the seed neighbourhood.
    near = dist_to_seed <= float(seed_neighbor_radius)
    if constraint is not None:
        near = near & constraint
    if int(near.sum()) < 3:
        return _empty(n, "too_few_seed_neighbors")
    try:
        seed_plane = fit_plane_least_squares(pts[near])
    except ValueError:
        return _empty(n, "seed_plane_fit_failed")
    seed_normal = seed_plane.normal

    # 2. Build the coplanar candidate set inside a bounded sphere, then grow the
    #    connected component from the seed across that surface.
    within_sphere = dist_to_seed <= float(max_extent)
    plane_offset = np.abs((pts - seed.reshape(1, 3)) @ seed_normal)
    candidate = within_sphere & (plane_offset <= float(plane_delta))
    if constraint is not None:
        candidate = candidate & constraint
    candidate[seed_idx] = True
    component = connected_component_from_seed(
        pts, candidate, seed_idx, radius=connect_radius
    )

    region_count = int(component.sum())
    if region_count < int(min_region_points):
        return PlanarRegionSelection(
            mask=component,
            point_count=region_count,
            confidence="low",
            reason="too_few_region_points",
            label="object",
            plane_point=seed,
            plane_normal=seed_normal,
            width_m=None,
            height_m=None,
        )

    # 3. Prune points whose local normal disagrees with the door/window plane,
    #    so the region does not bleed across the frame onto floor / ceiling.
    component = filter_by_normal_consistency(
        pts, component, seed_idx, seed_normal, max_angle_deg=normal_max_angle_deg
    )

    # 4. Score the grown region with the shared door/window geometry rules.
    indices = np.flatnonzero(component)
    score_label = label_hint if label_hint else "object"
    geometry = score_door_window_geometry(pts[indices], score_label)
    if geometry.confidence == "high" and geometry.plane_normal is not None:
        band = select_plane_band(
            pts, geometry.plane_point, geometry.plane_normal, thickness=display_thickness
        )
        final = component & band
        if not final.any():
            final = component
        # Drop spatially isolated stragglers (coplanar, in-frustum, but
        # disconnected from the body) that read as scatter in the global view.
        final = remove_isolated_points(pts, final, seed_idx)
        label = label_hint or infer_label_from_dimensions(geometry.width_m, geometry.height_m)
        return PlanarRegionSelection(
            mask=final,
            point_count=int(final.sum()),
            confidence="high",
            reason="accepted_vertical_planar_region",
            label=label,
            plane_point=geometry.plane_point,
            plane_normal=geometry.plane_normal,
            width_m=geometry.width_m,
            height_m=geometry.height_m,
        )

    return PlanarRegionSelection(
        mask=component,
        point_count=int(component.sum()),
        confidence=geometry.confidence,
        reason=geometry.reason,
        label=label_hint or "object",
        plane_point=geometry.plane_point if geometry.plane_point is not None else seed,
        plane_normal=geometry.plane_normal if geometry.plane_normal is not None else seed_normal,
        width_m=geometry.width_m,
        height_m=geometry.height_m,
    )
