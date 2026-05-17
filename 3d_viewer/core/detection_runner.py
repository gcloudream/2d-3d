"""Run OWLv2 detection as a subprocess for one panorama image."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.detection_cache import detection_output_dir, detection_output_path


def build_detection_command(
    workspace: Path,
    image_path: Path,
    score_thr: float = 0.12,
    pano_split: int = 4,
    pano_out_size: int = 768,
) -> list[str]:
    return [
        sys.executable,
        str(workspace / "2d" / "detect_owlvit.py"),
        "--input", str(image_path),
        "--output", str(detection_output_dir(workspace)),
        "--pano-split", str(pano_split),
        "--pano-out-size", str(pano_out_size),
        "--pano-fov", "90",
        "--score-thr", str(score_thr),
        "--min-area", "0.0015",
    ]


def run_detection_for_image(workspace: Path, image_path: Path) -> Path:
    detection_output_dir(workspace).mkdir(parents=True, exist_ok=True)
    cmd = build_detection_command(workspace, image_path)
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
    out = detection_output_path(workspace, image_path.name)
    if not out.exists():
        raise RuntimeError(f"detection finished but JSON was not created: {out}")
    return out
