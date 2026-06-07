"""Top-down wall-image outputs from sampled room point clouds."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


DEFAULT_RESOLUTION_M = 0.05
DEFAULT_HEIGHT_PERCENTILES = (20.0, 85.0)
DEFAULT_MIN_WALL_RUN_M = 1.2
DEFAULT_MAX_WALL_GAP_M = 0.20
DEFAULT_WEAK_DENSITY_PERCENTILE = 60.0
DEFAULT_STRONG_DENSITY_PERCENTILE = 75.0
DEFAULT_MIN_WEAK_VERTICAL_SPAN_M = 0.55
DEFAULT_MIN_STRONG_VERTICAL_SPAN_M = 1.00
DEFAULT_HEIGHT_LAYER_COUNT = 8
DEFAULT_MIN_WEAK_HEIGHT_LAYERS = 2
DEFAULT_MIN_STRONG_HEIGHT_LAYERS = 3
DEFAULT_MIN_COMPONENT_ASPECT = 3.0
DEFAULT_MAX_BLOB_FILL_RATIO = 0.22


@dataclass(frozen=True)
class WallImageResult:
    image_path: Path
    preserved_image_path: Path
    metadata_path: Path
    width_px: int
    height_px: int
    resolution_m: float
    selected_points: int
    total_points: int
    z_min: float
    z_max: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    wall_pixel_count: int


def wall_image_output_dir(workspace: Path) -> Path:
    return workspace / "out" / "wall_images"


def generate_wall_density_image(
    workspace: Path,
    data_root: Path,
    points: np.ndarray,
    *,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    height_percentiles: tuple[float, float] = DEFAULT_HEIGHT_PERCENTILES,
    min_wall_run_m: float = DEFAULT_MIN_WALL_RUN_M,
    max_wall_gap_m: float = DEFAULT_MAX_WALL_GAP_M,
) -> WallImageResult:
    """Create top-down density and preserved wall-density PNGs.

    The density image is the stable MVP output. The preserved image keeps weak
    wall evidence connected to stronger wall seeds, while suppressing isolated
    furniture-like blobs. It intentionally stays image-like instead of drawing
    inferred vector wall lines, so uncertain wall evidence remains visible.
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) == 0:
        raise ValueError("point cloud is empty")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")

    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        raise ValueError("point cloud contains no finite XYZ points")
    finite_points = pts[finite].astype(np.float64, copy=False)

    z = finite_points[:, 2]
    lo_pct, hi_pct = height_percentiles
    if not (0.0 <= lo_pct < hi_pct <= 100.0):
        raise ValueError("height_percentiles must be an increasing pair in [0, 100]")
    z_min = float(np.percentile(z, lo_pct))
    z_max = float(np.percentile(z, hi_pct))
    if z_max <= z_min:
        z_min = float(z.min())
        z_max = float(z.max())
    selected = finite_points[(z >= z_min) & (z <= z_max)]
    if len(selected) == 0:
        raise ValueError("height filter selected no points")

    x_min = float(finite_points[:, 0].min())
    x_max = float(finite_points[:, 0].max())
    y_min = float(finite_points[:, 1].min())
    y_max = float(finite_points[:, 1].max())
    width_px = max(1, int(np.ceil((x_max - x_min) / resolution_m)) + 1)
    height_px = max(1, int(np.ceil((y_max - y_min) / resolution_m)) + 1)

    ix = np.clip(((selected[:, 0] - x_min) / resolution_m).astype(np.int64), 0, width_px - 1)
    iy = np.clip(((selected[:, 1] - y_min) / resolution_m).astype(np.int64), 0, height_px - 1)
    grid = np.zeros((height_px, width_px), dtype=np.uint32)
    np.add.at(grid, (iy, ix), 1)
    height_layers = _height_layer_counts(
        height_px,
        width_px,
        ix,
        iy,
        selected[:, 2],
        z_min,
        z_max,
        layer_count=DEFAULT_HEIGHT_LAYER_COUNT,
    )

    full_ix = np.clip(((finite_points[:, 0] - x_min) / resolution_m).astype(np.int64), 0, width_px - 1)
    full_iy = np.clip(((finite_points[:, 1] - y_min) / resolution_m).astype(np.int64), 0, height_px - 1)
    z_grid_min = np.full((height_px, width_px), np.inf, dtype=np.float32)
    z_grid_max = np.full((height_px, width_px), -np.inf, dtype=np.float32)
    np.minimum.at(z_grid_min, (full_iy, full_ix), finite_points[:, 2].astype(np.float32))
    np.maximum.at(z_grid_max, (full_iy, full_ix), finite_points[:, 2].astype(np.float32))
    vertical_span = z_grid_max - z_grid_min
    vertical_span[~np.isfinite(vertical_span)] = 0.0

    density = np.log1p(grid.astype(np.float32))
    peak = float(density.max())
    if peak > 0.0:
        density = density / peak * 255.0
    image = Image.fromarray(np.flipud(density.astype(np.uint8)), mode="L")
    image = ImageOps.autocontrast(image)

    out_dir = wall_image_output_dir(workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = data_root.name or "dataset"
    image_path = out_dir / f"{stem}_wall_density.png"
    preserved_image_path = out_dir / f"{stem}_wall_preserved.png"
    metadata_path = out_dir / f"{stem}_wall_density.json"
    image.save(image_path)

    wall_mask = preserve_wall_density_grid(
        grid,
        vertical_span,
        height_layers,
        resolution_m=resolution_m,
        min_wall_run_m=min_wall_run_m,
        max_wall_gap_m=max_wall_gap_m,
    )
    preserved_image = render_preserved_wall_density_image(
        grid,
        wall_mask,
    )
    preserved_image.save(preserved_image_path)

    result = WallImageResult(
        image_path=image_path,
        preserved_image_path=preserved_image_path,
        metadata_path=metadata_path,
        width_px=width_px,
        height_px=height_px,
        resolution_m=float(resolution_m),
        selected_points=int(len(selected)),
        total_points=int(len(finite_points)),
        z_min=z_min,
        z_max=z_max,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        wall_pixel_count=int(wall_mask.sum()),
    )
    metadata_path.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "preserved_image": str(preserved_image_path),
                "data_root": str(data_root),
                "resolution_m": result.resolution_m,
                "width_px": result.width_px,
                "height_px": result.height_px,
                "selected_points": result.selected_points,
                "total_points": result.total_points,
                "wall_pixel_count": result.wall_pixel_count,
                "height_percentiles": [float(lo_pct), float(hi_pct)],
                "z_band": [result.z_min, result.z_max],
                "bounds": {
                    "x": [result.x_min, result.x_max],
                    "y": [result.y_min, result.y_max],
                },
                "mode": "topdown_density_and_preserved_wall_density",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def preserve_wall_density_grid(
    grid: np.ndarray,
    vertical_span: np.ndarray | None = None,
    height_layers: np.ndarray | None = None,
    *,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    min_wall_run_m: float = DEFAULT_MIN_WALL_RUN_M,
    max_wall_gap_m: float = DEFAULT_MAX_WALL_GAP_M,
    weak_density_percentile: float = DEFAULT_WEAK_DENSITY_PERCENTILE,
    strong_density_percentile: float = DEFAULT_STRONG_DENSITY_PERCENTILE,
    min_weak_vertical_span_m: float = DEFAULT_MIN_WEAK_VERTICAL_SPAN_M,
    min_strong_vertical_span_m: float = DEFAULT_MIN_STRONG_VERTICAL_SPAN_M,
    min_weak_height_layers: int = DEFAULT_MIN_WEAK_HEIGHT_LAYERS,
    min_strong_height_layers: int = DEFAULT_MIN_STRONG_HEIGHT_LAYERS,
) -> np.ndarray:
    """Preserve weak wall lines connected to stronger wall evidence."""
    counts = np.asarray(grid)
    if counts.ndim != 2:
        raise ValueError("grid must be a 2D array")
    if counts.size == 0:
        return np.zeros_like(counts, dtype=bool)

    nonzero_counts = counts[counts > 0]
    if len(nonzero_counts) == 0:
        return np.zeros_like(counts, dtype=bool)

    min_len_px = max(3, int(np.ceil(min_wall_run_m / resolution_m)))
    max_gap_px = max(1, int(np.ceil(max_wall_gap_m / resolution_m)))
    weak_threshold = max(1, int(np.percentile(nonzero_counts, weak_density_percentile)))
    strong_threshold = max(1, int(np.percentile(nonzero_counts, strong_density_percentile)))
    weak_density = counts >= weak_threshold
    strong_density = counts >= strong_threshold

    if vertical_span is not None:
        span = np.asarray(vertical_span, dtype=np.float32)
        if span.shape != counts.shape:
            raise ValueError("vertical_span shape must match grid shape")
        weak_geometry = span >= float(min_weak_vertical_span_m)
        strong_geometry = span >= float(min_strong_vertical_span_m)
    else:
        weak_geometry = counts > 0
        strong_geometry = counts > 0

    if height_layers is not None:
        layers = np.asarray(height_layers)
        if layers.shape != counts.shape:
            raise ValueError("height_layers shape must match grid shape")
        weak_geometry = weak_geometry | (layers >= int(min_weak_height_layers))
        strong_geometry = strong_geometry | (layers >= int(min_strong_height_layers))

    weak = weak_density & weak_geometry
    strong = strong_density & strong_geometry
    weak_lines = (
        _axis_run_mask(weak, axis=1, min_len_px=min_len_px, max_gap_px=max_gap_px)
        | _axis_run_mask(weak, axis=0, min_len_px=min_len_px, max_gap_px=max_gap_px)
    )
    strong_lines = (
        _axis_run_mask(strong, axis=1, min_len_px=max(3, min_len_px // 2), max_gap_px=max_gap_px)
        | _axis_run_mask(strong, axis=0, min_len_px=max(3, min_len_px // 2), max_gap_px=max_gap_px)
    )
    # Only line-supported weak evidence participates in hysteresis. Raw weak
    # pixels are too broad in this scan and can connect walls to furniture.
    weak_mask = weak_lines | strong
    seed = strong | (strong_lines & weak_mask)
    if seed.any():
        propagated = ndimage.binary_propagation(
            ndimage.binary_dilation(seed, structure=np.ones((3, 3), dtype=bool)) & weak_mask,
            mask=ndimage.binary_closing(weak_mask, structure=np.ones((3, 3), dtype=bool)),
        )
        wall_mask = propagated | weak_lines
    else:
        wall_mask = weak_lines
    wall_mask = wall_mask & (counts > 0)
    wall_mask = _remove_small_components(wall_mask, min_area=max(3, min_len_px))
    return wall_mask


def render_preserved_wall_density_image(
    grid: np.ndarray,
    wall_mask: np.ndarray,
) -> Image.Image:
    """Render original density only where preserved wall-like pixels remain."""
    counts = np.asarray(grid)
    density = np.log1p(counts.astype(np.float32))
    peak = float(density.max())
    if peak > 0.0:
        density = density / peak * 255.0
    filtered = density * np.asarray(wall_mask, dtype=np.float32)
    image = Image.fromarray(np.flipud(filtered.astype(np.uint8)), mode="L")
    return ImageOps.autocontrast(image)


# Backwards-compatible aliases for older tests/imports in this local project.
filter_wall_density_grid = preserve_wall_density_grid
render_filtered_wall_density_image = render_preserved_wall_density_image


def _close_1d_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    arr = np.asarray(values, dtype=bool).copy()
    if max_gap <= 0 or not arr.any():
        return arr
    runs = _runs_from_bool(~arr)
    for start, end in runs:
        if start == 0 or end == len(arr):
            continue
        if end - start <= max_gap:
            arr[start:end] = True
    return arr


def _height_layer_counts(
    height_px: int,
    width_px: int,
    ix: np.ndarray,
    iy: np.ndarray,
    z: np.ndarray,
    z_min: float,
    z_max: float,
    *,
    layer_count: int,
) -> np.ndarray:
    if layer_count <= 0 or z_max <= z_min:
        return np.zeros((height_px, width_px), dtype=np.uint8)
    layers = np.clip(
        np.floor((np.asarray(z, dtype=np.float64) - z_min) / (z_max - z_min) * layer_count).astype(np.int64),
        0,
        layer_count - 1,
    )
    occupied = np.zeros((layer_count, height_px, width_px), dtype=bool)
    occupied[layers, iy, ix] = True
    return occupied.sum(axis=0).astype(np.uint8)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(labels.ravel())
    keep = np.zeros(count + 1, dtype=bool)
    keep[1:] = areas[1:] >= int(min_area)
    return keep[labels]


def _remove_blob_like_components(
    mask: np.ndarray,
    *,
    min_aspect: float,
    max_fill_ratio: float,
) -> np.ndarray:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    keep = np.zeros(count + 1, dtype=bool)
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        if len(xs) == 0:
            continue
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        bbox_area = max(1, width * height)
        aspect = max(width, height) / max(1, min(width, height))
        fill_ratio = len(xs) / bbox_area
        keep[label] = aspect >= float(min_aspect) or fill_ratio <= float(max_fill_ratio)
    return keep[labels]


def _axis_run_mask(
    mask: np.ndarray,
    *,
    axis: int,
    min_len_px: int,
    max_gap_px: int,
) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    out = np.zeros_like(arr, dtype=bool)
    if axis == 1:
        for row in range(arr.shape[0]):
            trace = _close_1d_gaps(arr[row, :], max_gap_px)
            for start, end in _runs_from_bool(trace):
                if end - start >= min_len_px:
                    out[row, start:end] = True
        return out
    if axis == 0:
        for col in range(arr.shape[1]):
            trace = _close_1d_gaps(arr[:, col], max_gap_px)
            for start, end in _runs_from_bool(trace):
                if end - start >= min_len_px:
                    out[start:end, col] = True
        return out
    raise ValueError("axis must be 0 or 1")


def _runs_from_bool(values: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(values, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(arr.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(arr)))
    return runs
