"""Global orbit camera for inspecting the whole point cloud."""
from __future__ import annotations

import numpy as np

from render.camera import _normalize


class OrbitCamera:
    def __init__(self):
        self.target = np.zeros(3, dtype=np.float64)
        self.radius = 1.0
        self.distance = 5.0
        self.yaw_deg = -35.0
        self.pitch_deg = -35.0
        self.fov_deg = 55.0
        self.znear = 0.05
        self.zfar = 1000.0

    def fit_to_points(self, points: np.ndarray):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        self.target = (mn + mx) * 0.5
        self.radius = max(1.0, float(np.linalg.norm(mx - mn) * 0.5))
        self.distance = self.radius * 2.4
        self.znear = max(0.02, self.radius / 500.0)
        self.zfar = max(1000.0, self.radius * 20.0)

    @property
    def position(self) -> np.ndarray:
        return self.target - self.look_dir() * self.distance

    def reset_view(self):
        self.yaw_deg = -35.0
        self.pitch_deg = -35.0
        self.distance = self.radius * 2.4

    def orbit(self, delta_yaw_deg: float, delta_pitch_deg: float):
        self.yaw_deg = (self.yaw_deg + delta_yaw_deg) % 360.0
        self.pitch_deg = max(-85.0, min(85.0, self.pitch_deg + delta_pitch_deg))

    def zoom(self, delta: float):
        factor = 1.0 + float(delta) * 0.12
        factor = max(0.2, min(5.0, factor))
        self.distance = max(self.radius * 0.05, min(self.radius * 20.0, self.distance * factor))

    def look_dir(self) -> np.ndarray:
        y = np.deg2rad(self.yaw_deg)
        p = np.deg2rad(self.pitch_deg)
        cp = np.cos(p)
        return np.array([cp * np.cos(y), cp * np.sin(y), np.sin(p)], dtype=np.float64)

    def view_matrix(self) -> np.ndarray:
        eye = self.position
        target = self.target
        up = np.array([0.0, 0.0, 1.0])
        f = _normalize(target - eye)
        raw_s = np.cross(f, up)
        if np.linalg.norm(raw_s) < 1e-6:
            s = np.array([1.0, 0.0, 0.0])
        else:
            s = _normalize(raw_s)
        u = np.cross(s, f)
        m = np.eye(4, dtype=np.float64)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        t = np.eye(4, dtype=np.float64)
        t[:3, 3] = -eye
        return (m @ t).astype(np.float32)

    def proj_matrix(self, aspect: float) -> np.ndarray:
        f = 1.0 / np.tan(np.deg2rad(self.fov_deg) / 2.0)
        n, fz = self.znear, self.zfar
        p = np.zeros((4, 4), dtype=np.float64)
        p[0, 0] = f / aspect
        p[1, 1] = f
        p[2, 2] = (fz + n) / (n - fz)
        p[2, 3] = (2 * fz * n) / (n - fz)
        p[3, 2] = -1.0
        return p.astype(np.float32)
