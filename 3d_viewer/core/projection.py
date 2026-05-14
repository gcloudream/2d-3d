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
