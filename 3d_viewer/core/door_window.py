"""Door/window detection matching helpers for projected panorama points."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DetectionMatchResult:
    match_indices: np.ndarray
    hit_mask: np.ndarray


@dataclass(frozen=True)
class DetectionSelection:
    detection_index: int
    detection: dict | None
    label: str
    score: float | None
    mask: np.ndarray
    point_count: int


def _bbox_area(bbox: Sequence[float], pano_w: float) -> float:
    x1, y1, x2, y2 = bbox
    if x2 < x1:
        width = (pano_w - x1) + x2
    else:
        width = x2 - x1
    return max(0.0, width) * max(0.0, y2 - y1)


def _inside_bbox(uv: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    u = uv[:, 0]
    v = uv[:, 1]
    if x2 < x1:
        in_x = (u >= x1) | (u <= x2)
    else:
        in_x = (u >= x1) & (u <= x2)
    in_y = (v >= y1) & (v <= y2)
    return in_x & in_y


def match_points_to_detections(
    uv: np.ndarray,
    detections: Sequence[dict],
    pano_w: float,
) -> DetectionMatchResult:
    """Match projected panorama points to detections.

    If several detections contain the same point, the smallest bbox wins. This
    matches the intended door-with-window behavior from the proposal.
    """
    match_indices = np.full(len(uv), -1, dtype=np.int32)
    if len(uv) == 0 or not detections:
        return DetectionMatchResult(match_indices, match_indices >= 0)

    order = sorted(
        range(len(detections)),
        key=lambda i: _bbox_area(detections[i]["bbox"], pano_w),
    )
    for det_idx in order:
        inside = _inside_bbox(uv, detections[det_idx]["bbox"])
        take = inside & (match_indices < 0)
        match_indices[take] = det_idx

    return DetectionMatchResult(match_indices, match_indices >= 0)


def select_detection_region(
    clicked_idx: int,
    uv: np.ndarray,
    detections: Sequence[dict],
    pano_w: float,
) -> DetectionSelection:
    """Return the full point mask for the detection containing clicked_idx."""
    empty = np.zeros(len(uv), dtype=bool)
    if clicked_idx < 0 or clicked_idx >= len(uv) or not detections:
        return DetectionSelection(-1, None, "", None, empty, 0)

    matches = match_points_to_detections(uv, detections, pano_w)
    det_idx = int(matches.match_indices[clicked_idx])
    if det_idx < 0:
        return DetectionSelection(-1, None, "", None, empty, 0)

    mask = matches.match_indices == det_idx
    det = detections[det_idx]
    return DetectionSelection(
        detection_index=det_idx,
        detection=det,
        label=str(det.get("label", "")),
        score=float(det["score"]) if "score" in det else None,
        mask=mask,
        point_count=int(mask.sum()),
    )
