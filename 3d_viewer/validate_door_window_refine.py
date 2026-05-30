"""Validate coarse-vs-refined door/window point selection on one keyframe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORKSPACE = HERE.parent

from core.dataset import CameraPose, find_default_dataset, load_dataset
from core.door_window_refine import refine_detection_selection
from core.projection import project_points_to_panorama, rotation_from_angle


def _find_pose_for_image(poses: list[CameraPose], image_name: str) -> CameraPose:
    for pose in poses:
        if pose.image_name == image_name:
            return pose
    raise RuntimeError(f"no pose found for {image_name}")


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate refined door/window point cloud selection")
    ap.add_argument("--detections", required=True, help="path to detection or manual annotation JSON")
    ap.add_argument("--clicked-index", type=int, default=None, help="sampled point index to use as seed")
    ap.add_argument("--max-points", type=int, default=300_000)
    ap.add_argument("--output", default=str(WORKSPACE / "out" / "door_window_refine"))
    args = ap.parse_args()

    det_path = Path(args.detections).expanduser().resolve()
    payload = json.loads(det_path.read_text(encoding="utf-8"))
    image_path = Path(payload["image"])
    detections = list(payload.get("detections", []))
    if not detections:
        raise RuntimeError(f"no detections in {det_path}")

    cfg = find_default_dataset(WORKSPACE)
    if cfg is None:
        raise RuntimeError(f"no dataset found under {WORKSPACE}")
    dataset = load_dataset(cfg, max_points=args.max_points)
    pose = _find_pose_for_image(dataset.poses, image_path.name)

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    R = rotation_from_angle(pose.roll, pose.pitch, pose.yaw)
    uv = project_points_to_panorama(
        dataset.points,
        pose.position,
        R,
        img_w,
        img_h,
        yaw_offset_deg=dataset.pano_yaw_offset_deg,
    )

    clicked_idx = args.clicked_index
    if clicked_idx is None:
        first_bbox = detections[0]["bbox"]
        x1, y1, x2, y2 = [float(v) for v in first_bbox]
        if x2 < x1:
            inside_x = (uv[:, 0] >= x1) | (uv[:, 0] <= x2)
        else:
            inside_x = (uv[:, 0] >= x1) & (uv[:, 0] <= x2)
        inside = inside_x & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        indices = np.flatnonzero(inside)
        if len(indices) == 0:
            raise RuntimeError("first detection contains no sampled points")
        clicked_idx = int(indices[len(indices) // 2])

    selection = refine_detection_selection(
        points=dataset.points,
        uv=uv,
        clicked_idx=clicked_idx,
        detections=detections,
        pano_w=float(img_w),
        cam_pos=pose.position,
    )

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    font = _load_font(28)
    small_font = _load_font(18)

    coarse_idx = np.flatnonzero(selection.coarse_mask)
    refined_idx = np.flatnonzero(selection.refined_mask)
    for idx in coarse_idx[:: max(1, len(coarse_idx) // 6000)]:
        u, v = uv[idx]
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(0, 210, 255))
    for idx in refined_idx[:: max(1, len(refined_idx) // 6000)]:
        u, v = uv[idx]
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=(255, 40, 40))

    draw.rectangle((18, 18, 1260, 142), fill=(0, 0, 0))
    draw.text((34, 30), image_path.name, fill=(255, 255, 255), font=font)
    draw.text(
        (34, 78),
        f"cyan=coarse red=refined clicked={clicked_idx} confidence={selection.confidence} "
        f"coarse={selection.coarse_count} refined={selection.point_count} reason={selection.reason}",
        fill=(255, 255, 255),
        font=small_font,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_refine_check.jpg"
    vis.save(out_path, quality=92)

    print(f"[image] {image_path.name}")
    print(f"[clicked] {clicked_idx}")
    print(f"[detection] {selection.detection_index} {selection.label}")
    print(f"[confidence] {selection.confidence}")
    print(f"[points] coarse={selection.coarse_count:,} refined={selection.point_count:,}")
    print(f"[reason] {selection.reason}")
    print(f"[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
