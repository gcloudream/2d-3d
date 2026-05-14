"""单一 Camera 对象，被全景球与点云共享。

Keyframe 模式：position 由外部设置，用户操作只改 yaw/pitch/fov。
"""
from __future__ import annotations

import numpy as np

from core.projection import rotation_from_angle


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Camera:
    def __init__(self):
        self.position = np.zeros(3, dtype=np.float64)
        self.yaw_deg = 0.0      # 绕世界 +Z 轴 (LiDAR 的天向，朝上)，0=朝 +X
        self.pitch_deg = 0.0    # 上下，正向上
        self.fov_deg = 75.0
        self.znear = 0.05
        self.zfar = 500.0
        # 由外部 set_keyframe 注入：当前 keyframe 的 R_pano (世界 -> pano 局部)
        # 视角默认对齐 keyframe 的"前方"方向
        self._R_pano: np.ndarray | None = None
        self._initial_yaw_offset = 0.0  # 跳到 keyframe 时把 yaw=0 视作朝 keyframe 的前方

    # ------- keyframe 切换 -------

    def set_keyframe(self, position: np.ndarray, roll: float, pitch: float, yaw: float):
        """跳到一个新的全景采样点。重置 yaw/pitch 到该 keyframe 的"前方"。"""
        self.position = np.asarray(position, dtype=np.float64).copy()
        self._R_pano = rotation_from_angle(roll, pitch, yaw)
        # 默认让用户视角朝向 +X（世界），不强行对齐 keyframe 朝向。
        # 若需要对齐 keyframe 自身朝向，可从 R_pano 推算并赋 yaw_deg.
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0

    @property
    def R_pano(self) -> np.ndarray | None:
        return self._R_pano

    # ------- 鼠标交互 -------

    def orbit(self, delta_yaw_deg: float, delta_pitch_deg: float):
        self.yaw_deg = (self.yaw_deg + delta_yaw_deg) % 360.0
        self.pitch_deg = max(-89.0, min(89.0, self.pitch_deg + delta_pitch_deg))

    def zoom(self, delta_fov: float):
        self.fov_deg = max(20.0, min(110.0, self.fov_deg + delta_fov))

    # ------- 矩阵 -------

    def look_dir(self) -> np.ndarray:
        """相机视线方向（世界系，单位向量）。yaw=0,pitch=0 -> +X 方向。"""
        y = np.deg2rad(self.yaw_deg)
        p = np.deg2rad(self.pitch_deg)
        cp = np.cos(p)
        return np.array([cp * np.cos(y), cp * np.sin(y), np.sin(p)], dtype=np.float64)

    def view_matrix(self) -> np.ndarray:
        """4x4 view（world -> camera），与 OpenGL 约定一致：相机看 -Z。
        我们用 +Z up，相机看向 look_dir()。
        """
        eye = self.position
        target = eye + self.look_dir()
        up = np.array([0.0, 0.0, 1.0])
        f = _normalize(target - eye)
        s = _normalize(np.cross(f, up))
        if np.linalg.norm(s) < 1e-8:  # 视线和 up 平行
            s = np.array([1.0, 0.0, 0.0])
        u = np.cross(s, f)
        M = np.eye(4, dtype=np.float64)
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = -eye
        return (M @ T).astype(np.float32)

    def proj_matrix(self, aspect: float) -> np.ndarray:
        f = 1.0 / np.tan(np.deg2rad(self.fov_deg) / 2.0)
        n, fz = self.znear, self.zfar
        P = np.zeros((4, 4), dtype=np.float64)
        P[0, 0] = f / aspect
        P[1, 1] = f
        P[2, 2] = (fz + n) / (n - fz)
        P[2, 3] = (2 * fz * n) / (n - fz)
        P[3, 2] = -1.0
        return P.astype(np.float32)
