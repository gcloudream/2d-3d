"""Run OWLv2 detection as a subprocess for one panorama image."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.detection_cache import detection_output_dir, detection_output_path


DETECTION_MODE_LABELS = {
    "precise": "精准模式",
    "recall": "召回模式",
}

_PRECISE_CLASSES = (
    "door|a door|a glass door|an office door,"
    "window|a window|a glass window|an interior window"
)
_PRECISE_EXCLUDE = "monitor|computer monitor|screen|whiteboard|cabinet|shelf|poster"

_RECALL_CLASSES = (
    "door|a door|a glass door|an office door|a doorway,"
    "window|a window|a glass window|an interior window|a glass partition|"
    "a glass panel|a glass cabinet door"
)


def build_detection_command(
    workspace: Path,
    image_path: Path,
    mode: str = "precise",
) -> list[str]:
    if mode == "precise":
        score_thr = 0.10
        pano_split = 6
        pano_out_size = 1024
        classes = _PRECISE_CLASSES
        exclude = _PRECISE_EXCLUDE
    elif mode == "recall":
        score_thr = 0.06
        pano_split = 6
        pano_out_size = 1024
        classes = _RECALL_CLASSES
        exclude = ""
    else:
        raise ValueError(f"unknown detection mode: {mode}")

    cmd = [
        sys.executable,
        str(workspace / "2d" / "detect_owlvit.py"),
        "--input", str(image_path),
        "--output", str(detection_output_dir(workspace)),
        "--pano-split", str(pano_split),
        "--pano-out-size", str(pano_out_size),
        "--pano-fov", "90",
        "--score-thr", str(score_thr),
        "--min-area", "0.0015",
        "--classes", classes,
    ]
    if exclude:
        cmd.extend(["--exclude", exclude])
    return [
        *cmd,
    ]


def run_detection_for_image(workspace: Path, image_path: Path, mode: str = "precise") -> Path:
    detection_output_dir(workspace).mkdir(parents=True, exist_ok=True)
    cmd = build_detection_command(workspace, image_path, mode=mode)
    try:
        subprocess.run(
            cmd,
            cwd=workspace,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        tail = "\n".join((e.stdout or e.stderr or str(e)).splitlines()[-12:])
        raise RuntimeError(tail) from e
    produced = detection_output_path(workspace, image_path.name, mode="precise")
    if not produced.exists():
        raise RuntimeError(f"detection finished but JSON was not created: {produced}")
    final = detection_output_path(workspace, image_path.name, mode=mode)
    if final != produced:
        produced.replace(final)
    return final
