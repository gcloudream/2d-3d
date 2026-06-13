"""Detection JSON cache path helpers."""
from __future__ import annotations

from pathlib import Path


def detection_output_dir(workspace: Path) -> Path:
    return workspace / "out" / "door_window_detections"


def annotation_output_dir(workspace: Path) -> Path:
    return workspace / "out" / "door_window_annotations"


def detection_output_path(workspace: Path, image_name: str) -> Path:
    return detection_output_dir(workspace) / f"{Path(image_name).stem}.json"


def annotation_output_path(workspace: Path, image_name: str) -> Path:
    return annotation_output_dir(workspace) / f"{Path(image_name).stem}.json"


def detection_json_candidates(workspace: Path, image_name: str) -> list[Path]:
    stem = Path(image_name).stem
    return [
        annotation_output_path(workspace, image_name),
        detection_output_path(workspace, image_name),
        workspace / "2d" / "out" / f"{stem}.json",
        workspace / "2d" / "out_example_owlv2" / f"{stem}.json",
    ]


def find_detection_json(workspace: Path, image_name: str) -> Path | None:
    return next((path for path in detection_json_candidates(workspace, image_name) if path.exists()), None)
