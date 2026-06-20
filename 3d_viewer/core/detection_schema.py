"""Validation and normalization helpers for door/window detection payloads."""
from __future__ import annotations

from collections.abc import Sequence
from math import isfinite


def normalize_detection(det: object, img_w: int = 0, img_h: int = 0) -> dict | None:
    if not isinstance(det, dict):
        return None
    bbox = det.get("bbox")
    if isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if not all(isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if y2 <= y1:
        return None

    if img_w > 0:
        width = (float(img_w) - x1) + x2 if x2 < x1 else x2 - x1
        if width <= 0.0:
            return None
    elif x2 == x1:
        return None

    label = str(det.get("label", "")).lower() or "object"
    normalized = {
        "label": label,
        "bbox": [x1, y1, x2, y2],
    }
    score = det.get("score")
    if score is not None:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = None
        if score_value is not None and isfinite(score_value):
            normalized["score"] = score_value
    if "source" in det:
        normalized["source"] = str(det["source"])
    return normalized


def normalize_detections(detections: object, img_w: int = 0, img_h: int = 0) -> list[dict]:
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
        return []
    normalized: list[dict] = []
    for det in detections:
        item = normalize_detection(det, img_w=img_w, img_h=img_h)
        if item is not None:
            normalized.append(item)
    return normalized
