"""Persistence for door/window openings extracted from point-cloud selections."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WallOpening:
    id: str
    label: str
    source_image: str
    seed_index: int
    point_count: int
    confidence: str
    reason: str
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    width_m: float | None
    height_m: float | None
    z_min: float
    z_max: float
    detection_index: int
    score: float | None


def wall_openings_path(workspace: Path, data_root: Path) -> Path:
    return Path(workspace) / "out" / "wall_openings" / f"{Path(data_root).name}_openings.json"


def wall_opening_events_path(workspace: Path, data_root: Path) -> Path:
    return Path(workspace) / "out" / "wall_openings" / f"{Path(data_root).name}_events.jsonl"


def clear_wall_opening_session_files(workspace: Path, data_root: Path) -> list[Path]:
    removed: list[Path] = []
    for path in (wall_openings_path(workspace, data_root), wall_opening_events_path(workspace, data_root)):
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def append_wall_opening_event(workspace: Path, data_root: Path, event: dict) -> Path:
    path = wall_opening_events_path(workspace, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        fh.write("\n")
    return path


def load_wall_openings(workspace: Path, data_root: Path) -> list[WallOpening]:
    path = wall_openings_path(workspace, data_root)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [_opening_from_json(item) for item in payload.get("openings", [])]


def save_wall_openings(workspace: Path, data_root: Path, openings: list[WallOpening]) -> Path:
    path = wall_openings_path(workspace, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"openings": [asdict(item) for item in openings]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def append_wall_opening(workspace: Path, data_root: Path, opening: WallOpening) -> WallOpening:
    existing = load_wall_openings(workspace, data_root)
    for item in existing:
        if _is_duplicate_opening(item, opening):
            return item
    label = opening.label or "opening"
    saved = WallOpening(
        id=opening.id or f"{label}-{len(existing) + 1:04d}",
        label=opening.label,
        source_image=opening.source_image,
        seed_index=opening.seed_index,
        point_count=opening.point_count,
        confidence=opening.confidence,
        reason=opening.reason,
        center=opening.center,
        normal=opening.normal,
        bbox_min=opening.bbox_min,
        bbox_max=opening.bbox_max,
        width_m=opening.width_m,
        height_m=opening.height_m,
        z_min=opening.z_min,
        z_max=opening.z_max,
        detection_index=opening.detection_index,
        score=opening.score,
    )
    save_wall_openings(workspace, data_root, [*existing, saved])
    return saved


def _is_duplicate_opening(existing: WallOpening, opening: WallOpening) -> bool:
    if existing.label != opening.label:
        return False
    if existing.source_image != opening.source_image:
        return False
    if existing.detection_index != opening.detection_index:
        return False
    return (
        _xyz_close(existing.center, opening.center)
        and _xyz_close(existing.bbox_min, opening.bbox_min)
        and _xyz_close(existing.bbox_max, opening.bbox_max)
    )


def _xyz_close(a: tuple[float, float, float], b: tuple[float, float, float], tolerance: float = 0.02) -> bool:
    return all(abs(float(x) - float(y)) <= tolerance for x, y in zip(a, b))


def opening_from_selection(
    points: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
    source_image: str,
    seed_index: int,
    confidence: str,
    reason: str,
    plane_point: np.ndarray | None,
    plane_normal: np.ndarray | None,
    width_m: float | None,
    height_m: float | None,
    detection_index: int = -1,
    score: float | None = None,
) -> WallOpening:
    selected = np.asarray(points, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if len(selected) == 0:
        raise ValueError("cannot record an opening without selected points")
    bbox_min = selected.min(axis=0)
    bbox_max = selected.max(axis=0)
    center = selected.mean(axis=0)
    normal = np.asarray(plane_normal if plane_normal is not None else [0.0, 0.0, 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm > 0.0:
        normal = normal / norm
    return WallOpening(
        id="",
        label=label or "object",
        source_image=source_image,
        seed_index=int(seed_index),
        point_count=int(len(selected)),
        confidence=confidence,
        reason=reason,
        center=_rounded_xyz(center),
        normal=_rounded_xyz(normal),
        bbox_min=_rounded_xyz(bbox_min),
        bbox_max=_rounded_xyz(bbox_max),
        width_m=None if width_m is None else round(float(width_m), 5),
        height_m=None if height_m is None else round(float(height_m), 5),
        z_min=round(float(bbox_min[2]), 5),
        z_max=round(float(bbox_max[2]), 5),
        detection_index=int(detection_index),
        score=None if score is None else round(float(score), 5),
    )


def _rounded_xyz(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(round(float(v), 5) for v in values.reshape(3))


def _opening_from_json(item: dict) -> WallOpening:
    payload = dict(item)
    for key in ("center", "normal", "bbox_min", "bbox_max"):
        payload[key] = tuple(float(v) for v in payload[key])
    return WallOpening(**payload)
