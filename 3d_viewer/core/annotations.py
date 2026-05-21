"""Manual door/window annotation persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from core.detection_cache import annotation_output_path


def save_manual_annotations(
    workspace: Path,
    image_path: Path,
    width: int,
    height: int,
    detections: Sequence[dict],
) -> Path:
    out_path = annotation_output_path(workspace, image_path.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image": str(image_path),
        "width": int(width),
        "height": int(height),
        "model": "manual",
        "classes": ["door", "window"],
        "source": "pano_annotation_editor",
        "detections": list(detections),
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
