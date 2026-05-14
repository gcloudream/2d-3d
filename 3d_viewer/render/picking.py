"""屏幕空间 hover 吸附：把可见点投到 NDC，找到鼠标附近最近的点。"""
from __future__ import annotations

import numpy as np


def project_points_to_screen(
    points: np.ndarray, mvp: np.ndarray, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (screen_xy_int32 [N,2], visible_mask)。
    screen_xy 仅 visible 项有效。
    """
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
    return np.column_stack([sx, sy]), inside


def find_nearest_to_mouse(
    points: np.ndarray, mvp: np.ndarray, width: int, height: int,
    mouse_x: int, mouse_y: int, max_dist_px: int = 12,
) -> int:
    """返回吸附到的点 index，没有则 -1。
    简单实现：算所有点到鼠标的屏幕距离平方，取最小。
    30 万点 numpy 化查询通常约 20–30 ms，对 30ms 节流的 hover 来说够用。
    """
    if len(points) == 0:
        return -1
    sxy, inside = project_points_to_screen(points, mvp, width, height)
    if not np.any(inside):
        return -1

    visible_idx = np.flatnonzero(inside)
    visible_xy = sxy[visible_idx].astype(np.int64, copy=False)
    dx = visible_xy[:, 0] - int(mouse_x)
    dy = visible_xy[:, 1] - int(mouse_y)
    d2 = dx * dx + dy * dy
    best = int(np.argmin(d2))
    if d2[best] > max_dist_px * max_dist_px:
        return -1
    return int(visible_idx[best])
