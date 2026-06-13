"""屏幕空间 hover 吸附：把可见点投到 NDC，找到鼠标附近最近的点。"""
from __future__ import annotations

import numpy as np


def _project_points_to_screen_with_depth(
    points: np.ndarray, mvp: np.ndarray, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (screen_xy_int32 [N,2], visible_mask, ndc_z)。"""
    n = len(points)
    homo = np.concatenate([points, np.ones((n, 1), dtype=np.float32)], axis=1)
    clip = homo @ mvp.T  # [N,4]
    w = clip[:, 3]
    valid = w > 1e-6
    ndc = np.zeros_like(clip[:, :3])
    ndc[valid] = clip[valid, :3] / w[valid, None]
    inside = valid & (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1) \
                  & (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1) \
                  & (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1)
    sx = ((ndc[:, 0] + 1.0) * 0.5 * width).astype(np.int32)
    sy = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * height).astype(np.int32)
    return np.column_stack([sx, sy]), inside, ndc[:, 2]


def project_points_to_screen(
    points: np.ndarray, mvp: np.ndarray, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (screen_xy_int32 [N,2], visible_mask)。
    screen_xy 仅 visible 项有效。
    """
    sxy, inside, _ = _project_points_to_screen_with_depth(points, mvp, width, height)
    return sxy, inside


def _nearest_from_projected(
    sxy: np.ndarray,
    inside: np.ndarray,
    depth: np.ndarray,
    mouse_x: int,
    mouse_y: int,
    max_dist_px: int,
) -> int:
    if not np.any(inside):
        return -1

    visible_idx = np.flatnonzero(inside)
    visible_xy = sxy[visible_idx].astype(np.int64, copy=False)
    dx = visible_xy[:, 0] - int(mouse_x)
    dy = visible_xy[:, 1] - int(mouse_y)
    d2 = dx * dx + dy * dy
    candidates = d2 <= max_dist_px * max_dist_px
    if not np.any(candidates):
        return -1
    candidate_idx = visible_idx[candidates]
    candidate_d2 = d2[candidates]
    candidate_depth = depth[candidate_idx]
    best = int(np.lexsort((candidate_d2, candidate_depth))[0])
    return int(candidate_idx[best])


class ScreenPointIndex:
    """One-view cache of point projections used by hover and click picking."""

    def __init__(self):
        self._key: tuple[int, int, int, int, bytes] | None = None
        self._sxy: np.ndarray | None = None
        self._inside: np.ndarray | None = None
        self._depth: np.ndarray | None = None

    def clear(self):
        self._key = None
        self._sxy = None
        self._inside = None
        self._depth = None

    def _cache_key(self, points: np.ndarray, mvp: np.ndarray, width: int, height: int):
        pts = np.asarray(points)
        matrix = np.asarray(mvp, dtype=np.float32)
        return (id(pts), len(pts), int(width), int(height), matrix.tobytes())

    def _projection(self, points: np.ndarray, mvp: np.ndarray, width: int, height: int):
        key = self._cache_key(points, mvp, width, height)
        if key != self._key:
            self._sxy, self._inside, self._depth = _project_points_to_screen_with_depth(
                points, mvp, width, height,
            )
            self._key = key
        return self._sxy, self._inside, self._depth

    def find_nearest(
        self,
        points: np.ndarray,
        mvp: np.ndarray,
        width: int,
        height: int,
        mouse_x: int,
        mouse_y: int,
        max_dist_px: int = 12,
    ) -> int:
        if len(points) == 0:
            return -1
        sxy, inside, depth = self._projection(points, mvp, width, height)
        return _nearest_from_projected(sxy, inside, depth, mouse_x, mouse_y, max_dist_px)


def find_nearest_to_mouse(
    points: np.ndarray, mvp: np.ndarray, width: int, height: int,
    mouse_x: int, mouse_y: int, max_dist_px: int = 12,
) -> int:
    """返回吸附到的点 index，没有则 -1。
    在鼠标半径内优先选当前视角更靠前的点，同深度时再按屏幕距离兜底。
    30 万点 numpy 化查询通常约 20–30 ms，对 30ms 节流的 hover 来说够用。
    """
    if len(points) == 0:
        return -1
    sxy, inside, depth = _project_points_to_screen_with_depth(points, mvp, width, height)
    return _nearest_from_projected(sxy, inside, depth, mouse_x, mouse_y, max_dist_px)
