"""坐标系工具:
   - rotation_from_angle: roll/pitch/yaw -> 3x3 (与 算法例子/projectToPanoramic.py 完全一致)
   - world_to_camera_local: 把世界坐标变换到相机本地（球面贴图所用）
"""
from __future__ import annotations

import numpy as np


def rotation_from_angle(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """复刻 算法例子/projectToPanoramic.py 里 get_rotation_from_angle 的公式。"""
    roll = -roll
    pitch = -pitch

    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = np.cos(roll) * np.cos(yaw) + np.sin(pitch) * np.sin(roll) * np.sin(yaw)
    R[0, 1] = -np.cos(roll) * np.sin(yaw) + np.sin(pitch) * np.sin(roll) * np.cos(yaw)
    R[0, 2] = np.cos(pitch) * np.sin(roll)

    R[1, 0] = np.cos(pitch) * np.sin(yaw)
    R[1, 1] = np.cos(pitch) * np.cos(yaw)
    R[1, 2] = -np.sin(pitch)

    R[2, 0] = -np.sin(roll) * np.cos(yaw) + np.sin(pitch) * np.cos(roll) * np.sin(yaw)
    R[2, 1] = np.sin(roll) * np.sin(yaw) + np.sin(pitch) * np.cos(roll) * np.cos(yaw)
    R[2, 2] = np.cos(roll) * np.cos(pitch)
    return R


def world_to_camera_local(points: np.ndarray, cam_pos: np.ndarray, R_pano: np.ndarray) -> np.ndarray:
    """把世界点变到 pano 局部系（与 projectToPanoramic.coordinate_to_pixel 之前那一步一致）。"""
    return (R_pano @ (points.astype(np.float64) - cam_pos.reshape(1, 3)).T).T


def project_points_to_panorama(
    points: np.ndarray,
    cam_pos: np.ndarray,
    R_pano: np.ndarray,
    img_w: int,
    img_h: int,
    yaw_offset_deg: float = 0.0,
) -> np.ndarray:
    """把世界点批量投影到 equirectangular 全景像素坐标。

    yaw_offset_deg=0 时复刻 算法例子/projectToPanoramic.py 的 coordinate_to_pixel。
    非 0 时复刻 pano_sphere shader 对 u_pix 的水平采样偏移。
    """
    local = world_to_camera_local(points, cam_pos, R_pano)
    x = local[:, 0]
    y = local[:, 1]
    z = local[:, 2]

    tm_h = np.zeros_like(x, dtype=np.float64)
    np.divide(-x, y, out=tm_h, where=y != 0)
    tm_h = np.arctan(tm_h)

    left = x < 0
    front = y > 0
    u = np.empty_like(x, dtype=np.float64)
    u[left & front] = img_h - img_h * tm_h[left & front] / np.pi
    u[left & ~front] = img_h * (-tm_h[left & ~front]) / np.pi
    u[~left & front] = img_h - img_h * tm_h[~left & front] / np.pi
    u[~left & ~front] = img_w - img_h * tm_h[~left & ~front] / np.pi
    u = (np.floor(u + 0.5) + img_w * yaw_offset_deg / 360.0) % img_w

    horizontal_dist = np.hypot(x, y)
    tm_v = np.full_like(z, np.pi / 2.0, dtype=np.float64)
    np.divide(horizontal_dist, z, out=tm_v, where=z != 0)
    tm_v = np.where(z != 0, np.arctan(tm_v), tm_v)
    tm_v = np.where(tm_v < 0, tm_v + np.pi, tm_v)
    v = np.floor(img_h * tm_v / np.pi + 0.5) % img_h

    return np.column_stack([u, v])
