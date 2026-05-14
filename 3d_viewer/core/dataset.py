"""数据加载: 解析 camera_pos.cam, 加载点云抽样。

复用 desktop_viewer/data.py 的解析约定，但接口稍简化以匹配新查看器。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np


@dataclass(frozen=True)
class CameraPose:
    image_name: str
    x: float
    y: float
    z: float
    roll: float   # rad
    pitch: float  # rad
    yaw: float    # rad
    timestamp: float | None = None

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class Dataset:
    data_root: Path
    camera_file: Path
    image_dir: Path
    pointcloud_file: Path
    poses: list[CameraPose]
    points: np.ndarray              # (N, 3) float32
    colors: np.ndarray              # (N, 3) uint8
    total_points: int               # 原始点数
    sample_step: int                # 抽样步长


def find_default_dataset(workspace: Path) -> "Dataset | None":
    """从工作区里找到第一个看起来合理的数据集，返回 Dataset 配置（不加载）。"""
    cfg = _default_config(workspace)
    if cfg is None:
        return None
    return cfg


def _default_config(workspace: Path) -> "_Config | None":
    # 在 workspace 下找第一个有 CAM/camera_pos.cam 的目录
    for root in sorted(workspace.iterdir()):
        if not root.is_dir():
            continue
        cam = root / "CAM" / "camera_pos.cam"
        if not cam.exists():
            continue
        las_candidates = [
            root / "LAS_Rgb" / f"{root.name}_rgb_0.las",
            root / "LAS_resample" / f"{root.name}_rgb_0.las",
            root / "LAS" / f"{root.name}.las",
        ]
        las = next((p for p in las_candidates if p.exists()), None)
        if las is None:
            continue
        return _Config(root, cam, root / "CAM", las)
    return None


@dataclass
class _Config:
    data_root: Path
    camera_file: Path
    image_dir: Path
    pointcloud_file: Path


def parse_camera_file(camera_file: Path) -> list[CameraPose]:
    poses: list[CameraPose] = []
    with camera_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            timestamp = float(parts[9]) if len(parts) > 9 else None
            poses.append(CameraPose(
                image_name=parts[0],
                x=float(parts[3]), y=float(parts[4]), z=float(parts[5]),
                roll=float(parts[6]), pitch=float(parts[7]), yaw=float(parts[8]),
                timestamp=timestamp,
            ))
    # 同一 image_name 可能有多行，保留首次出现
    seen: set[str] = set()
    unique: list[CameraPose] = []
    for p in poses:
        if p.image_name in seen:
            continue
        seen.add(p.image_name)
        unique.append(p)
    return unique


def load_las_sample(las_file: Path, max_points: int = 300_000) -> tuple[np.ndarray, np.ndarray, int, int]:
    """读 LAS 抽样，返回 (points, colors, total_points, sample_step)。

    优先用 RGB，没有的话用 intensity 灰度。
    """
    with laspy.open(las_file) as reader:
        total = int(reader.header.point_count)
        step = max(1, int(np.ceil(total / max_points)))
        pts: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        for chunk in reader.chunk_iterator(1_000_000):
            sel = chunk[::step]
            xyz = np.column_stack((sel.x, sel.y, sel.z)).astype(np.float32)
            pts.append(xyz)

            if all(hasattr(sel, n) for n in ("red", "green", "blue")):
                rgb = np.column_stack((sel.red, sel.green, sel.blue))
                if rgb.max(initial=0) > 255:
                    rgb = rgb / 65535.0 * 255.0
                cols.append(rgb.astype(np.uint8))
            elif hasattr(sel, "intensity"):
                inten = sel.intensity.astype(np.float32)
                lo, hi = float(inten.min(initial=0)), float(inten.max(initial=1))
                gray = ((inten - lo) / max(hi - lo, 1.0) * 255.0).astype(np.uint8)
                cols.append(np.column_stack((gray, gray, gray)))
            else:
                cols.append(np.full((len(sel), 3), 180, dtype=np.uint8))
    points = np.concatenate(pts, 0) if pts else np.empty((0, 3), np.float32)
    colors = np.concatenate(cols, 0) if cols else np.empty((0, 3), np.uint8)
    return points, colors, total, step


def load_dataset(cfg: _Config, max_points: int = 300_000) -> Dataset:
    poses = parse_camera_file(cfg.camera_file)
    if not poses:
        raise RuntimeError(f"no poses parsed from {cfg.camera_file}")
    points, colors, total, step = load_las_sample(cfg.pointcloud_file, max_points)
    return Dataset(
        data_root=cfg.data_root,
        camera_file=cfg.camera_file,
        image_dir=cfg.image_dir,
        pointcloud_file=cfg.pointcloud_file,
        poses=poses,
        points=points, colors=colors,
        total_points=total, sample_step=step,
    )
