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
        self._keyframe_yaw_deg = 0.0  # 当前 keyframe 的"自然前方"yaw，供 reset_view 使用

    # ------- keyframe 切换 -------

    def set_keyframe(self, position: np.ndarray, roll: float, pitch: float, yaw: float):
        """跳到新的全景采样点，自动把视角对齐到全景的"自然前方"。

        当全景水平校准为 -90° 时，全景中心列对应 pano local +X 方向。
        该方向在世界系中为 R_pano^T @ (1,0,0) = R_pano[0,:]，
        对应的 viewer yaw = atan2(R[0,1], R[0,0])。
        """
        self.position = np.asarray(position, dtype=np.float64).copy()
        R = rotation_from_angle(roll, pitch, yaw)
        self._keyframe_yaw_deg = float(np.degrees(np.arctan2(R[0, 1], R[0, 0])))
        self.yaw_deg = self._keyframe_yaw_deg
        self.pitch_deg = 0.0

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
        raw_s = np.cross(f, up)
        if np.linalg.norm(raw_s) < 1e-6:  # 视线和 up 平行
            s = np.array([1.0, 0.0, 0.0])
        else:
            s = _normalize(raw_s)
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
