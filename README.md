# 2D / 3D Viewer Tools

This repository contains desktop tools and reference scripts for working with panoramic imagery and point clouds.

## Quick Start

On macOS, launch the main 3D workflow from Finder or Terminal:

```bash
./run_3d_viewer.command
```

The 3D viewer now contains the door/window workflow in one window:

- run OWLv2 detection on the current keyframe;
- show detection boxes on the panorama sphere;
- click a detected door/window region to highlight matching point-cloud points;
- edit the current panorama boxes in the built-in flat panorama editor;
- save manual boxes to `out/door_window_annotations/`, which take priority over model detections.

## Directories

- `3d_viewer/`: PySide6 + ModernGL viewer for showing point clouds and panoramas in a shared 3D space.
- `2d/`: 2D panorama utilities and detection scripts.
- `desktop_viewer/`: earlier desktop viewer implementation kept for reference.
- `算法例子/`: reference projection script used to keep panoramic coordinate formulas aligned.

Large captured datasets, raw point clouds, virtual environments, and generated outputs are intentionally excluded from git.
