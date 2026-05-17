"""Geometry helpers for drawing panorama detection boxes."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def split_wrapped_bbox(bbox: Sequence[float], pano_w: float) -> list[list[float]]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 >= x1:
        return [[x1, y1, x2, y2]]
    return [[x1, y1, pano_w, y2], [0.0, y1, x2, y2]]


def bbox_edge_samples(
    bbox: Sequence[float],
    pano_w: float,
    samples_per_edge: int = 32,
) -> np.ndarray:
    """Return pairs of (u, v) points suitable for GL_LINES."""
    n = max(2, int(samples_per_edge))
    segments: list[np.ndarray] = []
    for x1, y1, x2, y2 in split_wrapped_bbox(bbox, pano_w):
        top = np.column_stack([np.linspace(x1, x2, n), np.full(n, y1)])
        right = np.column_stack([np.full(n, x2), np.linspace(y1, y2, n)])
        bottom = np.column_stack([np.linspace(x2, x1, n), np.full(n, y2)])
        left = np.column_stack([np.full(n, x1), np.linspace(y1, y2, n)])
        for edge in (top, right, bottom, left):
            for a, b in zip(edge[:-1], edge[1:]):
                segments.extend([a, b])
    return np.asarray(segments, dtype=np.float32)


def uv_to_pano_local_dirs(
    uv: np.ndarray,
    img_w: float,
    img_h: float,
    yaw_offset_deg: float = 0.0,
) -> np.ndarray:
    """Convert display-image pixels to unit directions in pano local space."""
    u = (uv[:, 0].astype(np.float64) - img_w * yaw_offset_deg / 360.0) % img_w
    v = uv[:, 1].astype(np.float64)
    theta = (u / img_w) * (2.0 * np.pi)
    phi = (v / img_h) * np.pi

    sin_phi = np.sin(phi)
    dirs = np.column_stack([
        -sin_phi * np.sin(theta),
        -sin_phi * np.cos(theta),
        np.cos(phi),
    ])
    return dirs.astype(np.float32)
