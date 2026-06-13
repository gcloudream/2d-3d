"""Prototype 3D wall model generation from preserved top-down wall evidence."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from core.wall_image import (
    DEFAULT_HEIGHT_LAYER_COUNT,
    DEFAULT_HEIGHT_PERCENTILES,
    DEFAULT_RESOLUTION_M,
    _height_layer_counts,
    preserve_wall_density_grid,
    wall_image_output_dir,
)
from core.wall_openings import WallOpening, append_wall_opening_event


DEFAULT_WALL_THICKNESS_M = 0.14
DEFAULT_MIN_MODEL_WALL_LENGTH_M = 2.0
DEFAULT_STRIP_WIDTH_M = 0.18
DEFAULT_MIN_WALL_POINT_COUNT = 80
DEFAULT_MIN_WALL_HEIGHT_SPAN_M = 0.8
DEFAULT_MAX_WALL_RIDGE_WIDTH_M = 0.35
DEFAULT_BOUNDARY_TOLERANCE_M = 0.45
DEFAULT_CONNECTION_TOLERANCE_M = 0.65
DEFAULT_MIN_MAJOR_WALL_LENGTH_M = 4.0
DEFAULT_MIN_CONNECTED_WALL_LENGTH_M = 1.0
DEFAULT_MIN_RETURN_WALL_LENGTH_M = 1.0
DEFAULT_MIN_BOUNDARY_NOTCH_LENGTH_M = 0.45
DEFAULT_MAX_RETURN_WALL_LENGTH_M = 2.2
DEFAULT_RETURN_WALL_TOUCH_TOLERANCE_M = 0.35
DEFAULT_BOUNDARY_NOTCH_TOLERANCE_M = 0.75
DEFAULT_MIN_RETURN_WALL_POINT_COUNT = 180
DEFAULT_MIN_RETURN_WALL_HEIGHT_SPAN_M = 1.2
OBJ_MATERIAL_COLORS = {
    "wall": (0.62, 0.78, 0.82, 1.0),
    "opening_window": (0.05, 0.88, 0.36, 1.0),
    "opening_door": (1.0, 0.58, 0.08, 1.0),
    "opening": (0.95, 0.86, 0.16, 1.0),
}


@dataclass(frozen=True)
class WallSegment:
    orientation: str
    x1: float
    y1: float
    x2: float
    y2: float
    z_min: float
    z_max: float
    length_m: float
    point_count: int
    height_span_m: float


@dataclass(frozen=True)
class WallModelResult:
    obj_path: Path
    metadata_path: Path
    preview_path: Path
    topdown_preview_path: Path
    segment_count: int
    vertex_count: int
    face_count: int
    matched_opening_count: int = 0
    projected_opening_count: int = 0
    unmatched_opening_count: int = 0


@dataclass(frozen=True)
class WallLineDraft:
    wall_lines_path: Path
    topdown_preview_path: Path
    segments: list[WallSegment]
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    resolution_m: float
    grid_shape: tuple[int, int]
    grid: np.ndarray | None = None
    wall_mask: np.ndarray | None = None
    source: str = "auto_detected_wall_lines"

    @property
    def segment_count(self) -> int:
        return len(self.segments)


@dataclass(frozen=True)
class WallOpeningMarker:
    opening_id: str
    label: str
    segment_index: int
    orientation: str
    wall_coord: float
    axis_min: float
    axis_max: float
    z_min: float
    z_max: float
    side: float


@dataclass(frozen=True)
class _GridEvidence:
    grid: np.ndarray
    wall_mask: np.ndarray
    points: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    wall_x_min: float
    wall_x_max: float
    wall_y_min: float
    wall_y_max: float
    resolution_m: float
    z_model_min: float
    z_model_max: float


def wall_model_output_dir(workspace: Path) -> Path:
    return workspace / "out" / "wall_models"


def generate_wall_model(
    workspace: Path,
    data_root: Path,
    points: np.ndarray,
    *,
    openings: Iterable[WallOpening] | None = None,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
    min_wall_length_m: float = DEFAULT_MIN_MODEL_WALL_LENGTH_M,
    strip_width_m: float = DEFAULT_STRIP_WIDTH_M,
) -> WallModelResult:
    """Generate a simple OBJ wall model from sampled point-cloud evidence."""
    draft = generate_wall_lines(
        workspace,
        data_root,
        points,
        resolution_m=resolution_m,
        min_wall_length_m=min_wall_length_m,
        strip_width_m=strip_width_m,
    )
    return export_wall_model_from_wall_lines(
        workspace,
        data_root,
        draft,
        openings=openings,
        resolution_m=resolution_m,
        wall_thickness_m=wall_thickness_m,
    )


def generate_wall_lines(
    workspace: Path,
    data_root: Path,
    points: np.ndarray,
    *,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    min_wall_length_m: float = DEFAULT_MIN_MODEL_WALL_LENGTH_M,
    strip_width_m: float = DEFAULT_STRIP_WIDTH_M,
) -> WallLineDraft:
    """Generate an editable wall-line draft without exporting the OBJ model."""
    evidence, segments = _detect_wall_segments(
        points,
        resolution_m=resolution_m,
        min_wall_length_m=min_wall_length_m,
        strip_width_m=strip_width_m,
    )
    out_dir = wall_model_output_dir(workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(data_root).name or "dataset"
    wall_lines_path = out_dir / f"{stem}_wall_lines.json"
    topdown_preview_path = out_dir / f"{stem}_wall_lines_topdown.png"
    draft = WallLineDraft(
        wall_lines_path=wall_lines_path,
        topdown_preview_path=topdown_preview_path,
        segments=list(segments),
        x_min=evidence.x_min,
        y_min=evidence.y_min,
        x_max=evidence.x_max,
        y_max=evidence.y_max,
        resolution_m=resolution_m,
        grid_shape=tuple(int(v) for v in evidence.grid.shape),
        grid=evidence.grid,
        wall_mask=evidence.wall_mask,
    )
    render_wall_line_draft_preview(draft).save(topdown_preview_path)
    save_wall_line_draft(
        workspace,
        data_root,
        draft.segments,
        x_min=draft.x_min,
        y_min=draft.y_min,
        x_max=draft.x_max,
        y_max=draft.y_max,
        resolution_m=draft.resolution_m,
        source=draft.source,
        grid_shape=draft.grid_shape,
        topdown_preview_path=topdown_preview_path,
    )
    return draft


def _detect_wall_segments(
    points: np.ndarray,
    *,
    resolution_m: float,
    min_wall_length_m: float,
    strip_width_m: float,
) -> tuple[_GridEvidence, list[WallSegment]]:
    evidence = _build_grid_evidence(
        points,
        resolution_m=resolution_m,
    )
    candidate_min_wall_length_m = min(min_wall_length_m, DEFAULT_MIN_CONNECTED_WALL_LENGTH_M)
    raw_segments = extract_axis_aligned_wall_segments(
        evidence.wall_mask,
        x_min=evidence.x_min,
        y_min=evidence.y_min,
        resolution_m=resolution_m,
        min_wall_length_m=candidate_min_wall_length_m,
    )
    raw_segments.extend(
        extract_boundary_outline_wall_segments(
            evidence.wall_mask,
            x_min=evidence.x_min,
            y_min=evidence.y_min,
            resolution_m=resolution_m,
            min_wall_length_m=candidate_min_wall_length_m,
        )
    )
    raw_segments = _merge_similar_segments(
        raw_segments,
        resolution_m=resolution_m,
        min_length_m=candidate_min_wall_length_m,
    )
    segments = verify_wall_segments_with_points(
        raw_segments,
        evidence.points,
        fallback_z_min=evidence.z_model_min,
        fallback_z_max=evidence.z_model_max,
        strip_width_m=strip_width_m,
    )
    segments = filter_wall_segments_by_network(
        segments,
        x_min=evidence.wall_x_min,
        x_max=evidence.wall_x_max,
        y_min=evidence.wall_y_min,
        y_max=evidence.wall_y_max,
    )
    contour_segments = extract_contour_wall_segments(
        evidence.wall_mask,
        x_min=evidence.x_min,
        y_min=evidence.y_min,
        resolution_m=resolution_m,
        min_wall_length_m=DEFAULT_MIN_BOUNDARY_NOTCH_LENGTH_M,
    )
    contour_segments = verify_wall_segments_with_points(
        contour_segments,
        evidence.points,
        fallback_z_min=evidence.z_model_min,
        fallback_z_max=evidence.z_model_max,
        strip_width_m=strip_width_m,
        min_point_count=DEFAULT_MIN_RETURN_WALL_POINT_COUNT,
        min_height_span_m=DEFAULT_MIN_RETURN_WALL_HEIGHT_SPAN_M,
        max_segments=None,
    )
    segments.extend(
        recover_short_return_wall_segments(
            contour_segments,
            segments,
            x_min=evidence.wall_x_min,
            x_max=evidence.wall_x_max,
            y_min=evidence.wall_y_min,
            y_max=evidence.wall_y_max,
        )
    )
    segments = snap_wall_segment_endpoints(
        segments,
        tolerance_m=DEFAULT_CONNECTION_TOLERANCE_M,
    )
    segments = complete_parallel_return_bridges(segments)
    segments = complete_boundary_corner_gaps(
        segments,
        evidence.points,
        x_min=evidence.wall_x_min,
        x_max=evidence.wall_x_max,
        y_min=evidence.wall_y_min,
        y_max=evidence.wall_y_max,
        fallback_z_min=evidence.z_model_min,
        fallback_z_max=evidence.z_model_max,
        strip_width_m=strip_width_m,
    )
    segments = snap_wall_segment_endpoints(
        segments,
        tolerance_m=DEFAULT_CONNECTION_TOLERANCE_M,
    )
    return evidence, segments


def export_wall_model_from_wall_lines(
    workspace: Path,
    data_root: Path,
    wall_lines: WallLineDraft | Iterable[WallSegment],
    *,
    openings: Iterable[WallOpening] | None = None,
    resolution_m: float | None = None,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
) -> WallModelResult:
    """Export OBJ/preview artifacts from confirmed wall-line geometry."""
    if isinstance(wall_lines, WallLineDraft):
        draft = wall_lines
        segments = list(draft.segments)
        effective_resolution_m = float(resolution_m or draft.resolution_m)
    else:
        segments = [_normalize_wall_segment(segment) for segment in wall_lines]
        effective_resolution_m = float(resolution_m or DEFAULT_RESOLUTION_M)
        x_min, y_min, x_max, y_max = _segments_world_bounds(segments, margin_m=effective_resolution_m * 4)
        out_dir = wall_model_output_dir(workspace)
        stem = Path(data_root).name or "dataset"
        draft = WallLineDraft(
            wall_lines_path=out_dir / f"{stem}_wall_lines.json",
            topdown_preview_path=out_dir / f"{stem}_wall_lines_topdown.png",
            segments=segments,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            resolution_m=effective_resolution_m,
            grid_shape=_grid_shape_from_bounds(x_min, y_min, x_max, y_max, effective_resolution_m),
            source="provided_wall_lines",
        )
    opening_records = list(openings or [])
    opening_markers, unmatched_openings = match_wall_openings_to_segments(
        opening_records,
        segments,
    )
    projected_openings = _projected_openings_from_unmatched(opening_records, unmatched_openings)
    if opening_records:
        append_wall_opening_event(
            workspace,
            data_root,
            {
                "event": "generate_wall_model_opening_match",
                "segment_count": len(segments),
                "matched_opening_count": len(opening_markers),
                "projected_opening_count": len(projected_openings),
                "unmatched_opening_count": len(unmatched_openings),
                "matched_openings": [marker.__dict__ for marker in opening_markers],
                "projected_opening_ids": [opening.id for opening in projected_openings],
                "unmatched_openings": unmatched_openings,
            },
        )

    out_dir = wall_model_output_dir(workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = data_root.name or "dataset"
    obj_path = out_dir / f"{stem}_walls.obj"
    metadata_path = out_dir / f"{stem}_walls.json"
    preview_path = out_dir / f"{stem}_walls_preview.png"
    topdown_preview_path = out_dir / f"{stem}_walls_topdown.png"

    vertices, faces = wall_segments_to_mesh(segments, wall_thickness_m=wall_thickness_m)
    face_materials = ["wall"] * len(faces)
    for marker in opening_markers:
        marker_vertices, marker_faces = opening_markers_to_mesh(
            [marker],
            wall_thickness_m=wall_thickness_m,
        )
        marker_base = len(vertices)
        vertices.extend(marker_vertices)
        faces.extend(tuple(index + marker_base for index in face) for face in marker_faces)
        face_materials.extend([_opening_marker_material(marker.label)] * len(marker_faces))
    write_obj(obj_path, vertices, faces, face_materials=face_materials)
    _write_metadata(
        metadata_path,
        obj_path,
        preview_path,
        topdown_preview_path,
        segments,
        vertex_count=len(vertices),
        face_count=len(faces),
        wall_thickness_m=wall_thickness_m,
        resolution_m=effective_resolution_m,
        opening_markers=opening_markers,
        projected_openings=projected_openings,
        unmatched_openings=unmatched_openings,
        source_wall_lines=draft.wall_lines_path,
    )
    render_wall_line_draft_preview(
        draft,
        opening_markers=opening_markers,
        projected_openings=projected_openings,
    ).save(topdown_preview_path)
    render_wall_model_preview(segments, wall_thickness_m=wall_thickness_m).save(preview_path)

    return WallModelResult(
        obj_path=obj_path,
        metadata_path=metadata_path,
        preview_path=preview_path,
        topdown_preview_path=topdown_preview_path,
        segment_count=len(segments),
        vertex_count=len(vertices),
        face_count=len(faces),
        matched_opening_count=len(opening_markers),
        projected_opening_count=len(projected_openings),
        unmatched_opening_count=len(unmatched_openings),
    )


def save_wall_line_draft(
    workspace: Path,
    data_root: Path,
    segments: Iterable[WallSegment],
    *,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    resolution_m: float,
    source: str = "manual_wall_lines",
    grid_shape: tuple[int, int] | None = None,
    topdown_preview_path: Path | None = None,
) -> Path:
    """Persist editable wall-line geometry as the source of truth for OBJ export."""
    out_dir = wall_model_output_dir(workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(data_root).name or "dataset"
    wall_lines_path = out_dir / f"{stem}_wall_lines.json"
    preview_path = topdown_preview_path or out_dir / f"{stem}_wall_lines_topdown.png"
    normalized = [_normalize_wall_segment(segment) for segment in segments]
    shape = grid_shape or _grid_shape_from_bounds(x_min, y_min, x_max, y_max, resolution_m)
    payload = {
        "schema_version": 1,
        "source": source,
        "data_root": str(data_root),
        "topdown_preview": str(preview_path),
        "resolution_m": float(resolution_m),
        "bounds": {
            "x_min": float(x_min),
            "y_min": float(y_min),
            "x_max": float(x_max),
            "y_max": float(y_max),
        },
        "grid_shape": [int(shape[0]), int(shape[1])],
        "segment_count": len(normalized),
        "segments": [
            {
                "id": f"wall-{index + 1:04d}",
                "edited": source != "auto_detected_wall_lines",
                **_wall_segment_to_dict(segment),
            }
            for index, segment in enumerate(normalized)
        ],
    }
    wall_lines_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return wall_lines_path


def load_wall_line_draft(path: Path) -> WallLineDraft:
    """Load editable wall-line geometry saved by :func:`save_wall_line_draft`."""
    draft_path = Path(path)
    payload = json.loads(draft_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported wall line draft schema")
    bounds = payload.get("bounds") or {}
    resolution_m = float(payload.get("resolution_m", DEFAULT_RESOLUTION_M))
    segments = [
        _wall_segment_from_mapping(raw)
        for raw in payload.get("segments", [])
    ]
    grid_shape_raw = payload.get("grid_shape")
    if grid_shape_raw and len(grid_shape_raw) == 2:
        grid_shape = (int(grid_shape_raw[0]), int(grid_shape_raw[1]))
    else:
        grid_shape = _grid_shape_from_bounds(
            float(bounds.get("x_min", 0.0)),
            float(bounds.get("y_min", 0.0)),
            float(bounds.get("x_max", 0.0)),
            float(bounds.get("y_max", 0.0)),
            resolution_m,
        )
    return WallLineDraft(
        wall_lines_path=draft_path,
        topdown_preview_path=Path(payload.get("topdown_preview", draft_path.with_name(f"{draft_path.stem}_topdown.png"))),
        segments=segments,
        x_min=float(bounds.get("x_min", 0.0)),
        y_min=float(bounds.get("y_min", 0.0)),
        x_max=float(bounds.get("x_max", 0.0)),
        y_max=float(bounds.get("y_max", 0.0)),
        resolution_m=resolution_m,
        grid_shape=grid_shape,
        source=str(payload.get("source", "loaded_wall_lines")),
    )


def render_wall_line_draft_preview(
    draft: WallLineDraft,
    *,
    opening_markers: list[WallOpeningMarker] | None = None,
    projected_openings: list[WallOpening] | None = None,
) -> Image.Image:
    """Render the current editable wall-line draft on its top-down backdrop."""
    return _render_wall_line_draft(
        draft,
        segments=draft.segments,
        opening_markers=opening_markers,
        projected_openings=projected_openings,
    )


def render_wall_line_draft_background(draft: WallLineDraft) -> Image.Image:
    """Render only the non-editable top-down evidence layer for line editing."""
    return _render_wall_line_draft(draft, segments=[])


def _render_wall_line_draft(
    draft: WallLineDraft,
    *,
    segments: list[WallSegment],
    opening_markers: list[WallOpeningMarker] | None = None,
    projected_openings: list[WallOpening] | None = None,
) -> Image.Image:
    if draft.grid is not None and draft.wall_mask is not None:
        return render_wall_model_topdown_preview(
            draft.grid,
            draft.wall_mask,
            segments,
            draft.x_min,
            draft.y_min,
            draft.resolution_m,
            opening_markers=opening_markers,
            projected_openings=projected_openings,
        )
    height_px, width_px = draft.grid_shape
    grid = np.zeros((max(1, height_px), max(1, width_px)), dtype=np.uint32)
    wall_mask = np.zeros_like(grid, dtype=bool)
    return render_wall_model_topdown_preview(
        grid,
        wall_mask,
        segments,
        draft.x_min,
        draft.y_min,
        draft.resolution_m,
        opening_markers=opening_markers,
        projected_openings=projected_openings,
    )


def extract_axis_aligned_wall_segments(
    wall_mask: np.ndarray,
    *,
    x_min: float,
    y_min: float,
    resolution_m: float,
    min_wall_length_m: float,
    max_gap_m: float = 0.35,
    max_wall_ridge_width_m: float = DEFAULT_MAX_WALL_RIDGE_WIDTH_M,
) -> list[WallSegment]:
    """Extract thin horizontal/vertical wall ridges from a 2D wall mask."""
    mask = np.asarray(wall_mask, dtype=bool)
    min_len_px = max(3, int(np.ceil(min_wall_length_m / resolution_m)))
    max_gap_px = max(1, int(np.ceil(max_gap_m / resolution_m)))
    max_width_px = max(1, int(np.ceil(max_wall_ridge_width_m / resolution_m)))
    segments = _collect_thin_axis_runs(
        mask,
        orientation="vertical",
        x_min=x_min,
        y_min=y_min,
        resolution_m=resolution_m,
        min_len_px=min_len_px,
        max_gap_px=max_gap_px,
        max_width_px=max_width_px,
    )
    segments.extend(
        _collect_thin_axis_runs(
            mask,
            orientation="horizontal",
            x_min=x_min,
            y_min=y_min,
            resolution_m=resolution_m,
            min_len_px=min_len_px,
            max_gap_px=max_gap_px,
            max_width_px=max_width_px,
        )
    )

    return _merge_similar_segments(
        segments,
        resolution_m=resolution_m,
        min_length_m=min_wall_length_m,
    )


def extract_boundary_outline_wall_segments(
    wall_mask: np.ndarray,
    *,
    x_min: float,
    y_min: float,
    resolution_m: float,
    min_wall_length_m: float,
    max_gap_m: float = 0.25,
    envelope_tolerance_px: int = 2,
) -> list[WallSegment]:
    """Extract axis-aligned outer-envelope wall candidates from filled evidence."""
    mask = np.asarray(wall_mask, dtype=bool)
    if not mask.any():
        return []
    min_len_px = max(3, int(np.ceil(min_wall_length_m / resolution_m)))
    max_gap_px = max(1, int(np.ceil(max_gap_m / resolution_m)))
    segments: list[WallSegment] = []

    col_has = mask.any(axis=0)
    cols = np.flatnonzero(col_has)
    if len(cols):
        top_rows = np.asarray([np.flatnonzero(mask[:, col]).max() for col in cols], dtype=np.int64)
        bottom_rows = np.asarray([np.flatnonzero(mask[:, col]).min() for col in cols], dtype=np.int64)
        segments.extend(
            _outline_segments_from_envelope(
                positions=cols,
                envelope=top_rows,
                orientation="horizontal",
                x_min=x_min,
                y_min=y_min,
                resolution_m=resolution_m,
                axis_size=mask.shape[1],
                min_len_px=min_len_px,
                max_gap_px=max_gap_px,
                envelope_tolerance_px=envelope_tolerance_px,
            )
        )
        segments.extend(
            _outline_segments_from_envelope(
                positions=cols,
                envelope=bottom_rows,
                orientation="horizontal",
                x_min=x_min,
                y_min=y_min,
                resolution_m=resolution_m,
                axis_size=mask.shape[1],
                min_len_px=min_len_px,
                max_gap_px=max_gap_px,
                envelope_tolerance_px=envelope_tolerance_px,
            )
        )

    row_has = mask.any(axis=1)
    rows = np.flatnonzero(row_has)
    if len(rows):
        left_cols = np.asarray([np.flatnonzero(mask[row, :]).min() for row in rows], dtype=np.int64)
        right_cols = np.asarray([np.flatnonzero(mask[row, :]).max() for row in rows], dtype=np.int64)
        segments.extend(
            _outline_segments_from_envelope(
                positions=rows,
                envelope=left_cols,
                orientation="vertical",
                x_min=x_min,
                y_min=y_min,
                resolution_m=resolution_m,
                axis_size=mask.shape[0],
                min_len_px=min_len_px,
                max_gap_px=max_gap_px,
                envelope_tolerance_px=envelope_tolerance_px,
            )
        )
        segments.extend(
            _outline_segments_from_envelope(
                positions=rows,
                envelope=right_cols,
                orientation="vertical",
                x_min=x_min,
                y_min=y_min,
                resolution_m=resolution_m,
                axis_size=mask.shape[0],
                min_len_px=min_len_px,
                max_gap_px=max_gap_px,
                envelope_tolerance_px=envelope_tolerance_px,
            )
        )

    return _merge_similar_segments(
        segments,
        resolution_m=resolution_m,
        min_length_m=min_wall_length_m,
    )


def extract_contour_wall_segments(
    wall_mask: np.ndarray,
    *,
    x_min: float,
    y_min: float,
    resolution_m: float,
    min_wall_length_m: float = DEFAULT_MIN_RETURN_WALL_LENGTH_M,
    max_gap_m: float = 0.20,
) -> list[WallSegment]:
    """Extract short line candidates from the contour of preserved wall evidence."""
    mask = np.asarray(wall_mask, dtype=bool)
    if not mask.any():
        return []
    contour = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    min_len_px = max(3, int(np.ceil(min_wall_length_m / resolution_m)))
    max_gap_px = max(1, int(np.ceil(max_gap_m / resolution_m)))
    max_width_px = max(1, int(np.ceil(max(DEFAULT_RESOLUTION_M, DEFAULT_MAX_WALL_RIDGE_WIDTH_M * 0.6) / resolution_m)))
    segments = _collect_thin_axis_runs(
        contour,
        orientation="vertical",
        x_min=x_min,
        y_min=y_min,
        resolution_m=resolution_m,
        min_len_px=min_len_px,
        max_gap_px=max_gap_px,
        max_width_px=max_width_px,
    )
    segments.extend(
        _collect_thin_axis_runs(
            contour,
            orientation="horizontal",
            x_min=x_min,
            y_min=y_min,
            resolution_m=resolution_m,
            min_len_px=min_len_px,
            max_gap_px=max_gap_px,
            max_width_px=max_width_px,
        )
    )
    return _merge_similar_segments(
        segments,
        resolution_m=resolution_m,
        coord_tolerance_m=0.12,
        max_gap_m=max_gap_m,
        min_length_m=min_wall_length_m,
    )


def verify_wall_segments_with_points(
    segments: Iterable[WallSegment],
    points: np.ndarray,
    *,
    fallback_z_min: float,
    fallback_z_max: float,
    strip_width_m: float = DEFAULT_STRIP_WIDTH_M,
    min_point_count: int = DEFAULT_MIN_WALL_POINT_COUNT,
    min_height_span_m: float = DEFAULT_MIN_WALL_HEIGHT_SPAN_M,
    max_segments: int | None = 80,
) -> list[WallSegment]:
    pts = np.asarray(points, dtype=np.float64)
    verified: list[WallSegment] = []
    half = float(strip_width_m) / 2.0
    for seg in segments:
        if seg.orientation == "vertical":
            y_lo, y_hi = sorted((seg.y1, seg.y2))
            in_strip = (
                (np.abs(pts[:, 0] - seg.x1) <= half)
                & (pts[:, 1] >= y_lo - half)
                & (pts[:, 1] <= y_hi + half)
            )
        else:
            x_lo, x_hi = sorted((seg.x1, seg.x2))
            in_strip = (
                (np.abs(pts[:, 1] - seg.y1) <= half)
                & (pts[:, 0] >= x_lo - half)
                & (pts[:, 0] <= x_hi + half)
            )
        candidate = pts[in_strip]
        if len(candidate) < min_point_count:
            continue
        z_lo = float(np.percentile(candidate[:, 2], 10.0))
        z_hi = float(np.percentile(candidate[:, 2], 95.0))
        height_span = z_hi - z_lo
        if height_span < min_height_span_m:
            continue
        z_min = min(z_lo, fallback_z_min)
        z_max = max(z_hi, fallback_z_max)
        verified.append(
            WallSegment(
                seg.orientation,
                seg.x1,
                seg.y1,
                seg.x2,
                seg.y2,
                z_min,
                z_max,
                seg.length_m,
                int(len(candidate)),
                float(height_span),
            )
        )
    verified.sort(key=lambda item: item.length_m, reverse=True)
    if max_segments is None:
        return verified
    return verified[:max_segments]


def filter_wall_segments_by_network(
    segments: Iterable[WallSegment],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    boundary_tolerance_m: float = DEFAULT_BOUNDARY_TOLERANCE_M,
    connection_tolerance_m: float = DEFAULT_CONNECTION_TOLERANCE_M,
    min_major_wall_length_m: float = DEFAULT_MIN_MAJOR_WALL_LENGTH_M,
    min_connected_wall_length_m: float = DEFAULT_MIN_CONNECTED_WALL_LENGTH_M,
) -> list[WallSegment]:
    """Keep likely room walls and drop isolated short interior clutter lines."""
    candidates = list(segments)
    if not candidates:
        return []

    accepted: list[WallSegment] = []
    anchors: list[WallSegment] = []
    pending: list[WallSegment] = []
    for seg in candidates:
        if seg.length_m >= min_major_wall_length_m or _segment_near_boundary(
            seg,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            tolerance_m=boundary_tolerance_m,
        ):
            accepted.append(seg)
            anchors.append(seg)
        elif seg.length_m >= min_connected_wall_length_m:
            pending.append(seg)

    for seg in pending:
        if any(_segments_touch(seg, wall, connection_tolerance_m) for wall in anchors):
            accepted.append(seg)

    accepted.sort(key=lambda item: item.length_m, reverse=True)
    return accepted


def recover_short_return_wall_segments(
    candidates: Iterable[WallSegment],
    anchors: Iterable[WallSegment],
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    min_length_m: float = DEFAULT_MIN_RETURN_WALL_LENGTH_M,
    max_length_m: float = DEFAULT_MAX_RETURN_WALL_LENGTH_M,
    touch_tolerance_m: float = DEFAULT_RETURN_WALL_TOUCH_TOLERANCE_M,
    boundary_tolerance_m: float = DEFAULT_BOUNDARY_TOLERANCE_M,
) -> list[WallSegment]:
    """Recover short wall returns that directly attach to the trusted wall network."""
    anchor_list = list(anchors)
    recovered: list[WallSegment] = []
    for candidate in candidates:
        is_tiny_boundary_notch = False
        if candidate.length_m < min_length_m or candidate.length_m > max_length_m:
            has_notch_anchor_support = _has_endpoint_anchor_support(
                candidate,
                anchor_list,
                tolerance_m=DEFAULT_BOUNDARY_NOTCH_TOLERANCE_M,
            )
            is_tiny_boundary_notch = (
                DEFAULT_MIN_BOUNDARY_NOTCH_LENGTH_M <= candidate.length_m < min_length_m
                and _segment_near_boundary(
                    candidate,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    tolerance_m=DEFAULT_BOUNDARY_NOTCH_TOLERANCE_M,
                )
                and has_notch_anchor_support
            )
            if not is_tiny_boundary_notch:
                continue
        if candidate.point_count < DEFAULT_MIN_RETURN_WALL_POINT_COUNT:
            continue
        if candidate.height_span_m < DEFAULT_MIN_RETURN_WALL_HEIGHT_SPAN_M:
            continue
        if _segment_duplicates_any(candidate, anchor_list + recovered):
            continue
        touches_anchor = any(_segments_touch(candidate, anchor, touch_tolerance_m) for anchor in anchor_list)
        near_boundary = _segment_near_boundary(
            candidate,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            tolerance_m=boundary_tolerance_m,
        )
        if (
            (touches_anchor and _has_parallel_anchor_support(candidate, anchor_list))
            or (
                is_tiny_boundary_notch
                and _has_endpoint_anchor_support(
                    candidate,
                    anchor_list,
                    tolerance_m=DEFAULT_BOUNDARY_NOTCH_TOLERANCE_M,
                )
            )
            or (near_boundary and any(_segments_touch(candidate, anchor, DEFAULT_CONNECTION_TOLERANCE_M) for anchor in anchor_list))
        ):
            recovered.append(candidate)
    recovered.sort(key=lambda item: item.length_m, reverse=True)
    return recovered


def wall_mask_world_bounds(
    wall_mask: np.ndarray,
    *,
    x_min: float,
    y_min: float,
    resolution_m: float,
) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(np.asarray(wall_mask, dtype=bool))
    if len(xs) == 0:
        return (float(x_min), float(x_min), float(y_min), float(y_min))
    x0 = x_min + (int(xs.min()) + 0.5) * resolution_m
    x1 = x_min + (int(xs.max()) + 0.5) * resolution_m
    y0 = y_min + (int(ys.min()) + 0.5) * resolution_m
    y1 = y_min + (int(ys.max()) + 0.5) * resolution_m
    return (round(float(x0), 10), round(float(x1), 10), round(float(y0), 10), round(float(y1), 10))


def snap_wall_segment_endpoints(
    segments: Iterable[WallSegment],
    *,
    tolerance_m: float = DEFAULT_CONNECTION_TOLERANCE_M,
) -> list[WallSegment]:
    """Extend near-miss wall endpoints to nearby perpendicular wall segments."""
    source = list(segments)
    snapped: list[WallSegment] = []
    for seg in source:
        if seg.orientation == "vertical":
            y1, y2 = sorted((seg.y1, seg.y2))
            y1 = _snap_vertical_endpoint(seg.x1, y1, source, tolerance_m)
            y2 = _snap_vertical_endpoint(seg.x1, y2, source, tolerance_m)
            y1, y2 = sorted((y1, y2))
            snapped.append(_copy_segment_geometry(seg, seg.x1, y1, seg.x1, y2))
        else:
            x1, x2 = sorted((seg.x1, seg.x2))
            x1 = _snap_horizontal_endpoint(x1, seg.y1, source, tolerance_m)
            x2 = _snap_horizontal_endpoint(x2, seg.y1, source, tolerance_m)
            x1, x2 = sorted((x1, x2))
            snapped.append(_copy_segment_geometry(seg, x1, seg.y1, x2, seg.y2))
    snapped.sort(key=lambda item: item.length_m, reverse=True)
    return snapped


def complete_parallel_return_bridges(
    segments: Iterable[WallSegment],
    *,
    min_spacing_m: float = 0.4,
    max_spacing_m: float = 1.4,
    endpoint_tolerance_m: float = 0.35,
) -> list[WallSegment]:
    source = list(segments)
    completed = list(source)
    verticals = [seg for seg in source if seg.orientation == "vertical"]
    for idx, left in enumerate(verticals):
        for right in verticals[idx + 1:]:
            x0, x1 = sorted((left.x1, right.x1))
            spacing = x1 - x0
            if not (min_spacing_m <= spacing <= max_spacing_m):
                continue
            left_y0, left_y1 = sorted((left.y1, left.y2))
            right_y0, right_y1 = sorted((right.y1, right.y2))
            for y_left, y_right in ((left_y0, right_y0), (left_y1, right_y1)):
                if abs(y_left - y_right) > endpoint_tolerance_m:
                    continue
                y = (y_left + y_right) / 2.0
                bridge = WallSegment(
                    "horizontal",
                    x0,
                    y,
                    x1,
                    y,
                    min(left.z_min, right.z_min),
                    max(left.z_max, right.z_max),
                    spacing,
                    min(left.point_count, right.point_count),
                    min(left.height_span_m, right.height_span_m),
                )
                if not _segment_duplicates_any(bridge, completed, coord_tolerance_m=0.15, overlap_ratio=0.7):
                    completed.append(bridge)
    completed.sort(key=lambda item: item.length_m, reverse=True)
    return completed


def complete_boundary_corner_gaps(
    segments: Iterable[WallSegment],
    points: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    fallback_z_min: float,
    fallback_z_max: float,
    strip_width_m: float = DEFAULT_STRIP_WIDTH_M,
    min_gap_m: float = DEFAULT_MIN_BOUNDARY_NOTCH_LENGTH_M,
    max_gap_m: float = 1.8,
    coord_tolerance_m: float = DEFAULT_CONNECTION_TOLERANCE_M,
    boundary_tolerance_m: float = DEFAULT_BOUNDARY_TOLERANCE_M,
    min_point_count: int = 120,
    min_height_span_m: float = DEFAULT_MIN_RETURN_WALL_HEIGHT_SPAN_M,
) -> list[WallSegment]:
    source = list(segments)
    completed = list(source)
    horizontals = [
        seg for seg in source
        if seg.orientation == "horizontal"
        and _segment_near_boundary(
            seg,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            tolerance_m=boundary_tolerance_m,
        )
    ]
    verticals = [
        seg for seg in source
        if seg.orientation == "vertical"
        and _segment_near_boundary(
            seg,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            tolerance_m=boundary_tolerance_m,
        )
    ]
    candidates: list[WallSegment] = []
    for horizontal in horizontals:
        hx0, hx1 = sorted((horizontal.x1, horizontal.x2))
        for h_endpoint in (hx0, hx1):
            for vertical in verticals:
                if abs(vertical.x1 - h_endpoint) > coord_tolerance_m:
                    continue
                vy0, vy1 = sorted((vertical.y1, vertical.y2))
                for v_endpoint in (vy0, vy1):
                    gap = abs(v_endpoint - horizontal.y1)
                    if not (min_gap_m <= gap <= max_gap_m):
                        continue
                    for candidate_x in (vertical.x1, h_endpoint):
                        candidate = WallSegment(
                            "vertical",
                            candidate_x,
                            horizontal.y1,
                            candidate_x,
                            v_endpoint,
                            0.0,
                            0.0,
                            gap,
                            0,
                            0.0,
                        )
                        if _segment_duplicates_any(
                            candidate,
                            completed + candidates,
                            coord_tolerance_m=0.05,
                        ):
                            continue
                        candidates.append(candidate)

    verified = verify_wall_segments_with_points(
        candidates,
        points,
        fallback_z_min=fallback_z_min,
        fallback_z_max=fallback_z_max,
        strip_width_m=strip_width_m,
        min_point_count=min_point_count,
        min_height_span_m=min_height_span_m,
        max_segments=None,
    )
    completed.extend(verified)
    completed.sort(key=lambda item: item.length_m, reverse=True)
    return completed


def match_wall_openings_to_segments(
    openings: Iterable[WallOpening],
    segments: list[WallSegment],
    *,
    max_plane_distance_m: float = 0.35,
    span_tolerance_m: float = 0.25,
) -> tuple[list[WallOpeningMarker], list[dict]]:
    matched: list[WallOpeningMarker] = []
    unmatched: list[dict] = []
    for opening_index, opening in enumerate(openings):
        best = None
        best_distance = float("inf")
        center = np.asarray(opening.center, dtype=np.float64)
        bbox_min = np.asarray(opening.bbox_min, dtype=np.float64)
        bbox_max = np.asarray(opening.bbox_max, dtype=np.float64)
        for index, seg in enumerate(segments):
            if seg.orientation not in {"vertical", "horizontal"}:
                continue
            if seg.orientation == "vertical":
                axis_min, axis_max = sorted((float(bbox_min[1]), float(bbox_max[1])))
                seg_min, seg_max = sorted((seg.y1, seg.y2))
                distance = abs(float(center[0]) - seg.x1)
                side = 1.0 if float(center[0]) >= seg.x1 else -1.0
                wall_coord = seg.x1
            else:
                axis_min, axis_max = sorted((float(bbox_min[0]), float(bbox_max[0])))
                seg_min, seg_max = sorted((seg.x1, seg.x2))
                distance = abs(float(center[1]) - seg.y1)
                side = 1.0 if float(center[1]) >= seg.y1 else -1.0
                wall_coord = seg.y1
            if distance > max_plane_distance_m:
                continue
            if axis_max < seg_min - span_tolerance_m or axis_min > seg_max + span_tolerance_m:
                continue
            if distance < best_distance:
                best_distance = distance
                best = (index, seg, wall_coord, max(seg_min, axis_min), min(seg_max, axis_max), side)
        if best is None:
            unmatched.append(
                {
                    "opening_index": opening_index,
                    "id": opening.id,
                    "label": opening.label,
                    "reason": "no_matching_wall_segment",
                }
            )
            continue
        index, seg, wall_coord, axis_min, axis_max, side = best
        if axis_max <= axis_min:
            unmatched.append(
                {
                    "opening_index": opening_index,
                    "id": opening.id,
                    "label": opening.label,
                    "reason": "empty_projected_opening_span",
                }
            )
            continue
        matched.append(
            WallOpeningMarker(
                opening_id=opening.id,
                label=opening.label,
                segment_index=index,
                orientation=seg.orientation,
                wall_coord=float(wall_coord),
                axis_min=round(float(axis_min), 5),
                axis_max=round(float(axis_max), 5),
                z_min=round(max(float(opening.z_min), float(seg.z_min)), 5),
                z_max=round(min(float(opening.z_max), float(seg.z_max)), 5),
                side=float(side),
            )
        )
    return matched, unmatched


def opening_markers_to_mesh(
    markers: Iterable[WallOpeningMarker],
    *,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
    frame_width_m: float = 0.04,
    marker_depth_m: float = 0.025,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for marker in markers:
        axis_min, axis_max = sorted((marker.axis_min, marker.axis_max))
        z_min, z_max = sorted((marker.z_min, marker.z_max))
        if axis_max <= axis_min or z_max <= z_min:
            continue
        fw = min(frame_width_m, (axis_max - axis_min) / 2.0, (z_max - z_min) / 2.0)
        strips = [
            (axis_min, axis_max, z_min, z_min + fw),
            (axis_min, axis_max, z_max - fw, z_max),
            (axis_min, axis_min + fw, z_min, z_max),
            (axis_max - fw, axis_max, z_min, z_max),
        ]
        for a1, a2, z1, z2 in strips:
            _append_marker_strip_box(
                vertices,
                faces,
                marker,
                a1,
                a2,
                z1,
                z2,
                wall_thickness_m=wall_thickness_m,
                marker_depth_m=marker_depth_m,
            )
        _append_marker_fill_panel(
            vertices,
            faces,
            marker,
            axis_min,
            axis_max,
            z_min,
            z_max,
            wall_thickness_m=wall_thickness_m,
            marker_depth_m=marker_depth_m,
        )
    return vertices, faces


def _projected_openings_from_unmatched(
    openings: list[WallOpening],
    unmatched_openings: list[dict],
) -> list[WallOpening]:
    projected: list[WallOpening] = []
    for item in unmatched_openings:
        opening_index = item.get("opening_index")
        if not isinstance(opening_index, int):
            continue
        if 0 <= opening_index < len(openings):
            opening = openings[opening_index]
            if _is_projectable_unmatched_opening(opening):
                projected.append(opening)
    return projected


def _is_projectable_unmatched_opening(opening: WallOpening) -> bool:
    label = (opening.label or "").lower()
    if label not in {"door", "window"}:
        return False
    reason = (opening.reason or "").lower()
    if "too_few" in reason or "rejected_" in reason:
        return False
    confidence = (opening.confidence or "").lower()
    return confidence == "high" or reason.startswith(("accepted_", "fused_"))


def _opening_marker_material(label: str) -> str:
    normalized = (label or "").lower()
    if normalized == "window":
        return "opening_window"
    if normalized == "door":
        return "opening_door"
    return "opening"


def _append_marker_strip_box(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int, int]],
    marker: WallOpeningMarker,
    axis_min: float,
    axis_max: float,
    z_min: float,
    z_max: float,
    *,
    wall_thickness_m: float,
    marker_depth_m: float,
) -> None:
    face_offset = marker.side * (wall_thickness_m / 2.0 + 0.01)
    depth = marker.side * marker_depth_m
    if marker.orientation == "vertical":
        x1 = marker.wall_coord + face_offset
        x2 = marker.wall_coord + face_offset + depth
        y1, y2 = axis_min, axis_max
    else:
        x1, x2 = axis_min, axis_max
        y1 = marker.wall_coord + face_offset
        y2 = marker.wall_coord + face_offset + depth
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    base = len(vertices) + 1
    vertices.extend([
        (x1, y1, z_min), (x2, y1, z_min), (x2, y2, z_min), (x1, y2, z_min),
        (x1, y1, z_max), (x2, y1, z_max), (x2, y2, z_max), (x1, y2, z_max),
    ])
    faces.extend([
        (base + 0, base + 1, base + 2, base + 3),
        (base + 4, base + 7, base + 6, base + 5),
        (base + 0, base + 4, base + 5, base + 1),
        (base + 1, base + 5, base + 6, base + 2),
        (base + 2, base + 6, base + 7, base + 3),
        (base + 3, base + 7, base + 4, base + 0),
    ])


def _append_marker_fill_panel(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int, int]],
    marker: WallOpeningMarker,
    axis_min: float,
    axis_max: float,
    z_min: float,
    z_max: float,
    *,
    wall_thickness_m: float,
    marker_depth_m: float,
) -> None:
    offset = marker.side * (wall_thickness_m / 2.0 + 0.01 + marker_depth_m + 0.003)
    base = len(vertices) + 1
    if marker.orientation == "vertical":
        x = marker.wall_coord + offset
        vertices.extend([
            (x, axis_min, z_min),
            (x, axis_max, z_min),
            (x, axis_max, z_max),
            (x, axis_min, z_max),
        ])
    else:
        y = marker.wall_coord + offset
        vertices.extend([
            (axis_min, y, z_min),
            (axis_max, y, z_min),
            (axis_max, y, z_max),
            (axis_min, y, z_max),
        ])
    faces.append((base + 0, base + 1, base + 2, base + 3))


def wall_segments_to_mesh(
    segments: Iterable[WallSegment],
    *,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    half = float(wall_thickness_m) / 2.0
    for raw_segment in segments:
        seg = _normalize_wall_segment(raw_segment)
        dx = seg.x2 - seg.x1
        dy = seg.y2 - seg.y1
        length = float((dx * dx + dy * dy) ** 0.5)
        if length <= 1e-9:
            continue
        normal_x = -dy / length
        normal_y = dx / length
        z1, z2 = sorted((seg.z_min, seg.z_max))
        p1_left = (seg.x1 + normal_x * half, seg.y1 + normal_y * half, z1)
        p2_left = (seg.x2 + normal_x * half, seg.y2 + normal_y * half, z1)
        p2_right = (seg.x2 - normal_x * half, seg.y2 - normal_y * half, z1)
        p1_right = (seg.x1 - normal_x * half, seg.y1 - normal_y * half, z1)
        base = len(vertices) + 1
        vertices.extend([
            p1_left,
            p2_left,
            p2_right,
            p1_right,
            (p1_left[0], p1_left[1], z2),
            (p2_left[0], p2_left[1], z2),
            (p2_right[0], p2_right[1], z2),
            (p1_right[0], p1_right[1], z2),
        ])
        faces.extend([
            (base + 0, base + 1, base + 2, base + 3),
            (base + 4, base + 7, base + 6, base + 5),
            (base + 0, base + 4, base + 5, base + 1),
            (base + 1, base + 5, base + 6, base + 2),
            (base + 2, base + 6, base + 7, base + 3),
            (base + 3, base + 7, base + 4, base + 0),
        ])
    return vertices, faces


def write_obj(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int, int]],
    *,
    face_materials: Iterable[str] | None = None,
) -> None:
    material_names = list(face_materials) if face_materials is not None else ["wall"] * len(faces)
    if len(material_names) != len(faces):
        raise ValueError("face_materials must have the same length as faces")
    material_names = [
        name if name in OBJ_MATERIAL_COLORS else "opening"
        for name in material_names
    ]
    mtl_path = path.with_suffix(".mtl")
    _write_mtl(mtl_path, OBJ_MATERIAL_COLORS)

    lines = [
        "# prototype wall model generated from point-cloud wall evidence",
        f"mtllib {mtl_path.name}",
    ]
    for x, y, z in vertices:
        lines.append(f"v {x:.5f} {y:.5f} {z:.5f}")
    current_material = None
    for material, face in zip(material_names, faces):
        if material != current_material:
            lines.append(f"usemtl {material}")
            current_material = material
        lines.append("f " + " ".join(str(i) for i in face))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mtl(path: Path, materials: dict[str, tuple[float, float, float, float]]) -> None:
    lines = ["# materials for wall/opening OBJ preview"]
    for name, color in materials.items():
        r, g, b = color[:3]
        lines.extend([
            f"newmtl {name}",
            f"Ka {r:.4f} {g:.4f} {b:.4f}",
            f"Kd {r:.4f} {g:.4f} {b:.4f}",
            "Ks 0.0800 0.0800 0.0800",
            "Ns 24.0000",
            "illum 2",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def render_wall_model_topdown_preview(
    grid: np.ndarray,
    wall_mask: np.ndarray,
    segments: list[WallSegment],
    x_min: float,
    y_min: float,
    resolution_m: float,
    *,
    opening_markers: list[WallOpeningMarker] | None = None,
    projected_openings: list[WallOpening] | None = None,
) -> Image.Image:
    density = np.log1p(np.asarray(grid, dtype=np.float32))
    if float(density.max()) > 0.0:
        density = density / float(density.max()) * 160.0
    base = np.dstack([density, density, density]).astype(np.uint8)
    base[np.asarray(wall_mask, dtype=bool)] = np.array([130, 190, 210], dtype=np.uint8)
    image = Image.fromarray(np.flipud(base), mode="RGB")
    draw = ImageDraw.Draw(image)
    height = grid.shape[0]
    for seg in segments:
        x1 = (seg.x1 - x_min) / resolution_m
        y1 = height - 1 - (seg.y1 - y_min) / resolution_m
        x2 = (seg.x2 - x_min) / resolution_m
        y2 = height - 1 - (seg.y2 - y_min) / resolution_m
        draw.line((x1, y1, x2, y2), fill=(255, 70, 40), width=3)
    for marker in opening_markers or []:
        color = (40, 230, 110) if marker.label == "window" else (255, 205, 40)
        if marker.orientation == "vertical":
            x = (marker.wall_coord - x_min) / resolution_m
            y1 = height - 1 - (marker.axis_min - y_min) / resolution_m
            y2 = height - 1 - (marker.axis_max - y_min) / resolution_m
            draw.line((x, y1, x, y2), fill=color, width=7)
            draw.line((x, y1, x, y2), fill=(10, 20, 15), width=2)
        else:
            x1 = (marker.axis_min - x_min) / resolution_m
            x2 = (marker.axis_max - x_min) / resolution_m
            y = height - 1 - (marker.wall_coord - y_min) / resolution_m
            draw.line((x1, y, x2, y), fill=color, width=7)
            draw.line((x1, y, x2, y), fill=(10, 20, 15), width=2)
    for opening in projected_openings or []:
        ox1 = (opening.bbox_min[0] - x_min) / resolution_m
        ox2 = (opening.bbox_max[0] - x_min) / resolution_m
        oy1 = height - 1 - (opening.bbox_min[1] - y_min) / resolution_m
        oy2 = height - 1 - (opening.bbox_max[1] - y_min) / resolution_m
        rect = (min(ox1, ox2), min(oy1, oy2), max(ox1, ox2), max(oy1, oy2))
        draw.rectangle(rect, outline=(255, 150, 35), width=4)
        draw.rectangle(rect, outline=(20, 15, 5), width=1)
    return image


def render_wall_model_preview(
    segments: list[WallSegment],
    *,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
    size: tuple[int, int] = (960, 720),
) -> Image.Image:
    if not segments:
        return Image.new("RGB", size, (245, 247, 248))
    vertices, faces = wall_segments_to_mesh(segments, wall_thickness_m=wall_thickness_m)
    pts = np.asarray(vertices, dtype=np.float64)

    def project(p: np.ndarray) -> tuple[float, float]:
        x, y, z = p
        return (x - y, (x + y) * 0.42 - z * 1.15)

    projected = np.asarray([project(p) for p in pts], dtype=np.float64)
    margin = 60
    span = projected.max(axis=0) - projected.min(axis=0)
    scale = min((size[0] - 2 * margin) / max(span[0], 1e-6), (size[1] - 2 * margin) / max(span[1], 1e-6))
    projected = (projected - projected.min(axis=0)) * scale + margin
    projected[:, 1] = size[1] - projected[:, 1]

    image = Image.new("RGB", size, (245, 247, 248))
    draw = ImageDraw.Draw(image)
    face_depths = []
    for face in faces:
        idx = [i - 1 for i in face]
        depth = float(np.mean(pts[idx, 0] + pts[idx, 1] + pts[idx, 2] * 0.2))
        face_depths.append((depth, idx))
    for _, idx in sorted(face_depths):
        poly = [tuple(projected[i]) for i in idx]
        avg_z = float(np.mean(pts[idx, 2]))
        shade = int(np.clip(175 + avg_z * 14, 120, 220))
        draw.polygon(poly, fill=(shade, shade + 10, shade + 16), outline=(70, 82, 90))
    return ImageOps.expand(image, border=1, fill=(220, 224, 228))


def _build_grid_evidence(
    points: np.ndarray,
    *,
    resolution_m: float,
    height_percentiles: tuple[float, float] = DEFAULT_HEIGHT_PERCENTILES,
) -> _GridEvidence:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError("points must have shape (N, 3) and not be empty")
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        raise ValueError("point cloud contains no finite XYZ points")
    finite_points = pts[finite].astype(np.float64, copy=False)
    z = finite_points[:, 2]
    z_band_min = float(np.percentile(z, height_percentiles[0]))
    z_band_max = float(np.percentile(z, height_percentiles[1]))
    selected = finite_points[(z >= z_band_min) & (z <= z_band_max)]

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
        z_band_min,
        z_band_max,
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
    wall_mask = preserve_wall_density_grid(
        grid,
        vertical_span,
        height_layers,
        resolution_m=resolution_m,
        min_wall_run_m=DEFAULT_MIN_MODEL_WALL_LENGTH_M,
        max_wall_gap_m=0.25,
    )
    wall_x_min, wall_x_max, wall_y_min, wall_y_max = wall_mask_world_bounds(
        wall_mask,
        x_min=x_min,
        y_min=y_min,
        resolution_m=resolution_m,
    )
    return _GridEvidence(
        grid=grid,
        wall_mask=wall_mask,
        points=finite_points,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        wall_x_min=wall_x_min,
        wall_x_max=wall_x_max,
        wall_y_min=wall_y_min,
        wall_y_max=wall_y_max,
        resolution_m=resolution_m,
        z_model_min=float(np.percentile(z, 10.0)),
        z_model_max=float(np.percentile(z, 95.0)),
    )


def _collect_thin_axis_runs(
    mask: np.ndarray,
    *,
    orientation: str,
    x_min: float,
    y_min: float,
    resolution_m: float,
    min_len_px: int,
    max_gap_px: int,
    max_width_px: int,
) -> list[WallSegment]:
    """Collect long axis-aligned runs whose perpendicular support stays thin."""
    segments: list[WallSegment] = []
    if orientation == "vertical":
        for col in range(mask.shape[1]):
            trace = _close_1d_gaps(mask[:, col], max_gap_px)
            for start, end in _runs_from_bool(trace):
                if end - start < min_len_px:
                    continue
                if not _run_is_thin(mask, orientation, col, start, end, max_width_px):
                    continue
                x = x_min + (col + 0.5) * resolution_m
                y1 = y_min + (start + 0.5) * resolution_m
                y2 = y_min + (end - 0.5) * resolution_m
                segments.append(WallSegment("vertical", x, y1, x, y2, 0.0, 0.0, abs(y2 - y1), 0, 0.0))
        return segments

    if orientation == "horizontal":
        for row in range(mask.shape[0]):
            trace = _close_1d_gaps(mask[row, :], max_gap_px)
            for start, end in _runs_from_bool(trace):
                if end - start < min_len_px:
                    continue
                if not _run_is_thin(mask, orientation, row, start, end, max_width_px):
                    continue
                x1 = x_min + (start + 0.5) * resolution_m
                x2 = x_min + (end - 0.5) * resolution_m
                y = y_min + (row + 0.5) * resolution_m
                segments.append(WallSegment("horizontal", x1, y, x2, y, 0.0, 0.0, abs(x2 - x1), 0, 0.0))
        return segments

    raise ValueError("orientation must be vertical or horizontal")


def _outline_segments_from_envelope(
    *,
    positions: np.ndarray,
    envelope: np.ndarray,
    orientation: str,
    x_min: float,
    y_min: float,
    resolution_m: float,
    axis_size: int,
    min_len_px: int,
    max_gap_px: int,
    envelope_tolerance_px: int,
) -> list[WallSegment]:
    segments: list[WallSegment] = []
    for coord in np.unique(envelope):
        trace = np.zeros(axis_size, dtype=bool)
        near = np.abs(envelope - int(coord)) <= int(envelope_tolerance_px)
        trace[positions[near]] = True
        trace = _close_1d_gaps(trace, max_gap_px)
        for start, end in _runs_from_bool(trace):
            if end - start < min_len_px:
                continue
            if orientation == "horizontal":
                x1 = x_min + (start + 0.5) * resolution_m
                x2 = x_min + (end - 0.5) * resolution_m
                y = y_min + (int(coord) + 0.5) * resolution_m
                segments.append(WallSegment("horizontal", x1, y, x2, y, 0.0, 0.0, abs(x2 - x1), 0, 0.0))
            elif orientation == "vertical":
                x = x_min + (int(coord) + 0.5) * resolution_m
                y1 = y_min + (start + 0.5) * resolution_m
                y2 = y_min + (end - 0.5) * resolution_m
                segments.append(WallSegment("vertical", x, y1, x, y2, 0.0, 0.0, abs(y2 - y1), 0, 0.0))
            else:
                raise ValueError("orientation must be vertical or horizontal")
    return segments


def _segment_near_boundary(
    seg: WallSegment,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    tolerance_m: float,
) -> bool:
    if seg.orientation == "vertical":
        return abs(seg.x1 - x_min) <= tolerance_m or abs(seg.x1 - x_max) <= tolerance_m
    return abs(seg.y1 - y_min) <= tolerance_m or abs(seg.y1 - y_max) <= tolerance_m


def _normalize_wall_segment(segment: WallSegment) -> WallSegment:
    orientation = str(segment.orientation)
    if orientation == "vertical":
        x = float((segment.x1 + segment.x2) / 2.0)
        y1 = float(segment.y1)
        y2 = float(segment.y2)
        length = abs(y2 - y1)
        x1 = x2 = x
    elif orientation == "horizontal":
        y = float((segment.y1 + segment.y2) / 2.0)
        x1 = float(segment.x1)
        x2 = float(segment.x2)
        length = abs(x2 - x1)
        y1 = y2 = y
    elif orientation == "free":
        x1 = float(segment.x1)
        y1 = float(segment.y1)
        x2 = float(segment.x2)
        y2 = float(segment.y2)
        length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    else:
        x1 = float(segment.x1)
        y1 = float(segment.y1)
        x2 = float(segment.x2)
        y2 = float(segment.y2)
        orientation = _wall_segment_orientation_from_points(x1, y1, x2, y2)
        length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if length <= 0.0:
        raise ValueError("wall segment length must be positive")
    return WallSegment(
        orientation,
        x1,
        y1,
        x2,
        y2,
        float(segment.z_min),
        float(segment.z_max),
        float(length),
        int(segment.point_count),
        float(segment.height_span_m),
    )


def _wall_segment_orientation_from_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    tolerance: float = 1e-6,
) -> str:
    if abs(x2 - x1) <= tolerance:
        return "vertical"
    if abs(y2 - y1) <= tolerance:
        return "horizontal"
    return "free"


def _wall_segment_to_dict(segment: WallSegment) -> dict:
    normalized = _normalize_wall_segment(segment)
    return {
        "orientation": normalized.orientation,
        "x1": normalized.x1,
        "y1": normalized.y1,
        "x2": normalized.x2,
        "y2": normalized.y2,
        "z_min": normalized.z_min,
        "z_max": normalized.z_max,
        "length_m": normalized.length_m,
        "point_count": normalized.point_count,
        "height_span_m": normalized.height_span_m,
    }


def _wall_segment_from_mapping(raw: Mapping[str, object]) -> WallSegment:
    orientation = str(raw.get("orientation", ""))
    segment = WallSegment(
        orientation,
        float(raw.get("x1", 0.0)),
        float(raw.get("y1", 0.0)),
        float(raw.get("x2", 0.0)),
        float(raw.get("y2", 0.0)),
        float(raw.get("z_min", 0.0)),
        float(raw.get("z_max", 2.6)),
        float(raw.get("length_m", 0.0)),
        int(raw.get("point_count", 0)),
        float(raw.get("height_span_m", 0.0)),
    )
    return _normalize_wall_segment(segment)


def _segments_world_bounds(
    segments: Iterable[WallSegment],
    *,
    margin_m: float,
) -> tuple[float, float, float, float]:
    values = list(segments)
    if not values:
        return -1.0, -1.0, 1.0, 1.0
    xs: list[float] = []
    ys: list[float] = []
    for segment in values:
        xs.extend([float(segment.x1), float(segment.x2)])
        ys.extend([float(segment.y1), float(segment.y2)])
    return (
        min(xs) - margin_m,
        min(ys) - margin_m,
        max(xs) + margin_m,
        max(ys) + margin_m,
    )


def _grid_shape_from_bounds(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    resolution_m: float,
) -> tuple[int, int]:
    resolution = max(float(resolution_m), 1e-6)
    width_px = max(1, int(np.ceil((float(x_max) - float(x_min)) / resolution)) + 1)
    height_px = max(1, int(np.ceil((float(y_max) - float(y_min)) / resolution)) + 1)
    return height_px, width_px


def _snap_vertical_endpoint(
    x: float,
    y: float,
    segments: list[WallSegment],
    tolerance_m: float,
) -> float:
    best_y = y
    best_distance = float(tolerance_m)
    for other in segments:
        if other.orientation != "horizontal":
            continue
        x0, x1 = sorted((other.x1, other.x2))
        distance = abs(other.y1 - y)
        if distance <= best_distance and x0 - tolerance_m <= x <= x1 + tolerance_m:
            best_y = other.y1
            best_distance = distance
    return best_y


def _snap_horizontal_endpoint(
    x: float,
    y: float,
    segments: list[WallSegment],
    tolerance_m: float,
) -> float:
    best_x = x
    best_distance = float(tolerance_m)
    for other in segments:
        if other.orientation != "vertical":
            continue
        y0, y1 = sorted((other.y1, other.y2))
        distance = abs(other.x1 - x)
        if distance <= best_distance and y0 - tolerance_m <= y <= y1 + tolerance_m:
            best_x = other.x1
            best_distance = distance
    return best_x


def _copy_segment_geometry(
    source: WallSegment,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> WallSegment:
    if source.orientation == "vertical":
        length = abs(y2 - y1)
    elif source.orientation == "horizontal":
        length = abs(x2 - x1)
    else:
        length = float(((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2) ** 0.5)
    return WallSegment(
        source.orientation,
        float(x1),
        float(y1),
        float(x2),
        float(y2),
        source.z_min,
        source.z_max,
        float(length),
        source.point_count,
        source.height_span_m,
    )


def _segments_touch(a: WallSegment, b: WallSegment, tolerance_m: float) -> bool:
    if a.orientation == b.orientation:
        if a.orientation == "vertical":
            if abs(a.x1 - b.x1) > tolerance_m:
                return False
            a0, a1 = sorted((a.y1, a.y2))
            b0, b1 = sorted((b.y1, b.y2))
        else:
            if abs(a.y1 - b.y1) > tolerance_m:
                return False
            a0, a1 = sorted((a.x1, a.x2))
            b0, b1 = sorted((b.x1, b.x2))
        return max(a0, b0) <= min(a1, b1) + tolerance_m

    vertical = a if a.orientation == "vertical" else b
    horizontal = b if a.orientation == "vertical" else a
    vx = vertical.x1
    vy0, vy1 = sorted((vertical.y1, vertical.y2))
    hx0, hx1 = sorted((horizontal.x1, horizontal.x2))
    hy = horizontal.y1
    return (
        hx0 - tolerance_m <= vx <= hx1 + tolerance_m
        and vy0 - tolerance_m <= hy <= vy1 + tolerance_m
    )


def _segment_duplicates_any(
    candidate: WallSegment,
    segments: list[WallSegment],
    *,
    coord_tolerance_m: float = DEFAULT_MAX_WALL_RIDGE_WIDTH_M,
    overlap_ratio: float = 0.5,
) -> bool:
    for seg in segments:
        if candidate.orientation != seg.orientation:
            continue
        if candidate.orientation == "vertical":
            if abs(candidate.x1 - seg.x1) > coord_tolerance_m:
                continue
            a0, a1 = sorted((candidate.y1, candidate.y2))
            b0, b1 = sorted((seg.y1, seg.y2))
        else:
            if abs(candidate.y1 - seg.y1) > coord_tolerance_m:
                continue
            a0, a1 = sorted((candidate.x1, candidate.x2))
            b0, b1 = sorted((seg.x1, seg.x2))
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        if overlap >= min(candidate.length_m, seg.length_m) * overlap_ratio:
            return True
    return False


def _has_endpoint_anchor_support(
    candidate: WallSegment,
    anchors: list[WallSegment],
    *,
    tolerance_m: float,
) -> bool:
    candidate_endpoints = ((candidate.x1, candidate.y1), (candidate.x2, candidate.y2))
    tolerance_sq = float(tolerance_m) ** 2
    for anchor in anchors:
        anchor_endpoints = ((anchor.x1, anchor.y1), (anchor.x2, anchor.y2))
        for cx, cy in candidate_endpoints:
            for ax, ay in anchor_endpoints:
                if (cx - ax) ** 2 + (cy - ay) ** 2 <= tolerance_sq:
                    return True
    return False


def _has_parallel_anchor_support(
    candidate: WallSegment,
    anchors: list[WallSegment],
    *,
    min_spacing_m: float = 0.4,
    max_spacing_m: float = 1.4,
    endpoint_tolerance_m: float = 0.45,
) -> bool:
    for anchor in anchors:
        if candidate.orientation != anchor.orientation:
            continue
        if candidate.orientation == "vertical":
            spacing = abs(candidate.x1 - anchor.x1)
            c0, c1 = sorted((candidate.y1, candidate.y2))
            a0, a1 = sorted((anchor.y1, anchor.y2))
        else:
            spacing = abs(candidate.y1 - anchor.y1)
            c0, c1 = sorted((candidate.x1, candidate.x2))
            a0, a1 = sorted((anchor.x1, anchor.x2))
        if not (min_spacing_m <= spacing <= max_spacing_m):
            continue
        endpoint_aligned = abs(c0 - a0) <= endpoint_tolerance_m or abs(c1 - a1) <= endpoint_tolerance_m
        overlapping = max(c0, a0) <= min(c1, a1) + endpoint_tolerance_m
        if endpoint_aligned and overlapping:
            return True
    return False


def _run_is_thin(
    mask: np.ndarray,
    orientation: str,
    coord: int,
    start: int,
    end: int,
    max_width_px: int,
) -> bool:
    widths: list[int] = []
    step = max(1, (end - start) // 80)
    for along in range(start, end, step):
        if orientation == "vertical":
            if not mask[along, coord]:
                continue
            widths.append(_contiguous_width(mask[along, :], coord))
        else:
            if not mask[coord, along]:
                continue
            widths.append(_contiguous_width(mask[:, along], coord))
    if not widths:
        return False
    values = np.asarray(widths, dtype=np.float32)
    return bool(
        float(np.median(values)) <= max_width_px
        and float(np.percentile(values, 90.0)) <= max_width_px * 1.5
    )


def _contiguous_width(values: np.ndarray, center: int) -> int:
    arr = np.asarray(values, dtype=bool)
    left = center
    while left > 0 and arr[left - 1]:
        left -= 1
    right = center
    while right + 1 < len(arr) and arr[right + 1]:
        right += 1
    return right - left + 1


def _merge_similar_segments(
    segments: list[WallSegment],
    *,
    resolution_m: float,
    coord_tolerance_m: float = DEFAULT_MAX_WALL_RIDGE_WIDTH_M,
    max_gap_m: float = 0.45,
    min_length_m: float | None = None,
) -> list[WallSegment]:
    merged: list[WallSegment] = []
    for orientation in ("vertical", "horizontal"):
        same = [s for s in segments if s.orientation == orientation]
        if orientation == "vertical":
            same.sort(key=lambda s: (s.x1, s.y1))
        else:
            same.sort(key=lambda s: (s.y1, s.x1))
        groups: list[list[WallSegment]] = []
        for seg in same:
            coord = seg.x1 if orientation == "vertical" else seg.y1
            if not groups:
                groups.append([seg])
                continue
            group_coord = float(np.mean([s.x1 if orientation == "vertical" else s.y1 for s in groups[-1]]))
            if abs(coord - group_coord) <= coord_tolerance_m:
                groups[-1].append(seg)
            else:
                groups.append([seg])
        for group in groups:
            if not group:
                continue
            if orientation == "vertical":
                coord = float(np.mean([s.x1 for s in group]))
                intervals = sorted((min(s.y1, s.y2), max(s.y1, s.y2)) for s in group)
            else:
                coord = float(np.mean([s.y1 for s in group]))
                intervals = sorted((min(s.x1, s.x2), max(s.x1, s.x2)) for s in group)
            cur_start, cur_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= cur_end + max_gap_m:
                    cur_end = max(cur_end, end)
                    continue
                merged.append(_segment_from_interval(orientation, coord, cur_start, cur_end))
                cur_start, cur_end = start, end
            merged.append(_segment_from_interval(orientation, coord, cur_start, cur_end))
    requested_min_length = DEFAULT_MIN_MODEL_WALL_LENGTH_M * 0.8 if min_length_m is None else float(min_length_m)
    min_length = max(requested_min_length, resolution_m * 3)
    return [s for s in merged if s.length_m >= min_length]


def _segment_from_interval(orientation: str, coord: float, start: float, end: float) -> WallSegment:
    if orientation == "vertical":
        return WallSegment("vertical", coord, start, coord, end, 0.0, 0.0, abs(end - start), 0, 0.0)
    return WallSegment("horizontal", start, coord, end, coord, 0.0, 0.0, abs(end - start), 0, 0.0)


def _write_metadata(
    path: Path,
    obj_path: Path,
    preview_path: Path,
    topdown_preview_path: Path,
    segments: list[WallSegment],
    *,
    vertex_count: int,
    face_count: int,
    wall_thickness_m: float,
    resolution_m: float,
    opening_markers: list[WallOpeningMarker] | None = None,
    projected_openings: list[WallOpening] | None = None,
    unmatched_openings: list[dict] | None = None,
    source_wall_lines: Path | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "obj": str(obj_path),
                "preview": str(preview_path),
                "topdown_preview": str(topdown_preview_path),
                "source_wall_lines": str(source_wall_lines) if source_wall_lines else None,
                "segment_count": len(segments),
                "vertex_count": vertex_count,
                "face_count": face_count,
                "wall_thickness_m": wall_thickness_m,
                "resolution_m": resolution_m,
                "segments": [seg.__dict__ for seg in segments],
                "matched_opening_count": len(opening_markers or []),
                "projected_opening_count": len(projected_openings or []),
                "unmatched_opening_count": len(unmatched_openings or []),
                "opening_markers": [marker.__dict__ for marker in (opening_markers or [])],
                "projected_openings": [
                    _projected_opening_metadata(opening) for opening in (projected_openings or [])
                ],
                "unmatched_openings": unmatched_openings or [],
                "mode": "prototype_ridge_outline_network_filtered_snapped_wall_mesh_from_preserved_density",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _projected_opening_metadata(opening: WallOpening) -> dict:
    return {
        "id": opening.id,
        "label": opening.label,
        "source_image": opening.source_image,
        "confidence": opening.confidence,
        "reason": opening.reason,
        "center": list(opening.center),
        "bbox_min": list(opening.bbox_min),
        "bbox_max": list(opening.bbox_max),
        "z_min": opening.z_min,
        "z_max": opening.z_max,
    }


def _close_1d_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    arr = np.asarray(values, dtype=bool).copy()
    if max_gap <= 0 or not arr.any():
        return arr
    for start, end in _runs_from_bool(~arr):
        if start == 0 or end == len(arr):
            continue
        if end - start <= max_gap:
            arr[start:end] = True
    return arr


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
