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
    roll: float
    pitch: float
    yaw: float
    timestamp: float | None = None

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class DatasetConfig:
    data_root: Path
    camera_file: Path
    image_dir: Path
    pointcloud_file: Path


@dataclass
class PointCloudSample:
    points: np.ndarray
    colors: np.ndarray
    total_points: int
    sample_step: int


def default_dataset_config(workspace: Path) -> DatasetConfig:
    root = workspace / "20260129135824"
    return DatasetConfig(
        data_root=root,
        camera_file=root / "CAM" / "camera_pos.cam",
        image_dir=root / "CAM",
        pointcloud_file=root / "LAS_Rgb" / "20260129135824_rgb_0.las",
    )


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
            poses.append(
                CameraPose(
                    image_name=parts[0],
                    x=float(parts[3]),
                    y=float(parts[4]),
                    z=float(parts[5]),
                    roll=float(parts[6]),
                    pitch=float(parts[7]),
                    yaw=float(parts[8]),
                    timestamp=timestamp,
                )
            )
    return poses


def unique_camera_poses(poses: list[CameraPose]) -> list[CameraPose]:
    """Keep one row per image name while preserving file order."""
    seen: set[str] = set()
    unique: list[CameraPose] = []
    for pose in poses:
        if pose.image_name in seen:
            continue
        seen.add(pose.image_name)
        unique.append(pose)
    return unique


def validate_images(poses: list[CameraPose], image_dir: Path) -> list[str]:
    return [pose.image_name for pose in poses if not (image_dir / pose.image_name).exists()]


def load_las_sample(pointcloud_file: Path, max_points: int = 250_000) -> PointCloudSample:
    """Read a bounded point cloud sample suitable for live desktop rendering."""
    with laspy.open(pointcloud_file) as reader:
        total = int(reader.header.point_count)
        step = max(1, int(np.ceil(total / max_points)))

        point_chunks: list[np.ndarray] = []
        color_chunks: list[np.ndarray] = []

        for chunk in reader.chunk_iterator(1_000_000):
            selected = chunk[::step]
            xyz = np.column_stack((selected.x, selected.y, selected.z)).astype(np.float32)
            point_chunks.append(xyz)

            if all(hasattr(selected, name) for name in ("red", "green", "blue")):
                rgb = np.column_stack((selected.red, selected.green, selected.blue))
                if rgb.max(initial=0) > 255:
                    rgb = rgb / 65535.0 * 255.0
                color_chunks.append(rgb.astype(np.uint8))
            elif hasattr(selected, "intensity"):
                intensity = selected.intensity.astype(np.float32)
                lo = float(intensity.min(initial=0))
                hi = float(intensity.max(initial=1))
                span = max(hi - lo, 1.0)
                gray = ((intensity - lo) / span * 255.0).astype(np.uint8)
                color_chunks.append(np.column_stack((gray, gray, gray)))
            else:
                gray = np.full((len(selected), 3), 180, dtype=np.uint8)
                color_chunks.append(gray)

    points = np.concatenate(point_chunks, axis=0) if point_chunks else np.empty((0, 3), dtype=np.float32)
    colors = np.concatenate(color_chunks, axis=0) if color_chunks else np.empty((0, 3), dtype=np.uint8)
    return PointCloudSample(points=points, colors=colors, total_points=total, sample_step=step)
