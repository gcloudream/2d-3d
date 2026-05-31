"""Validate projected point-cloud points against door/window detection bboxes.

Usage with an existing OWLv2 JSON:
    cd /Users/gengchen/Desktop/3dtiqu
    .venv/bin/python 3d_viewer/validate_door_window_match.py \
        --detections 2d/out_example_owlv2/592.692836_IMG.json
"""
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
from core.door_window import match_points_to_detections
from core.projection import project_points_to_panorama, rotation_from_angle


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


def _find_pose_for_image(poses: list[CameraPose], image_name: str) -> CameraPose:
    for pose in poses:
        if pose.image_name == image_name:
            return pose
    raise RuntimeError(f"no pose found for {image_name}")


def _subsample_indices(n: int, count: int) -> np.ndarray:
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, count, dtype=np.int64)


def _draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: list[float],
    pano_w: int,
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
):
    x1, y1, x2, y2 = bbox
    boxes = [(x1, y1, x2, y2)] if x2 >= x1 else [(x1, y1, pano_w - 1, y2), (0, y1, x2, y2)]
    for box in boxes:
        draw.rectangle(box, outline=color, width=5)
    draw.text((x1 + 6, max(0, y1 - 30)), label, fill=color, font=font)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate point projection against detections")
    ap.add_argument("--detections", required=True, help="path to a 2d detection JSON file")
    ap.add_argument("--offset", type=float, default=None, help="yaw offset used for projection")
    ap.add_argument("--max-points", type=int, default=120_000, help="LAS sample size")
    ap.add_argument("--draw-points", type=int, default=8_000, help="number of projected points to draw")
    ap.add_argument("--output", default=str(WORKSPACE / "out" / "door_window_match"))
    args = ap.parse_args()

    det_path = Path(args.detections).expanduser().resolve()
    payload = json.loads(det_path.read_text(encoding="utf-8"))
    image_path = Path(payload["image"])
    detections = payload.get("detections", [])
    if not detections:
        raise RuntimeError(f"no detections in {det_path}")

    cfg = find_default_dataset(WORKSPACE)
    if cfg is None:
        raise RuntimeError(f"no dataset found under {WORKSPACE}")
    dataset = load_dataset(cfg, max_points=args.max_points)
    offset = dataset.pano_yaw_offset_deg if args.offset is None else float(args.offset)
    pose = _find_pose_for_image(dataset.poses, image_path.name)

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    R = rotation_from_angle(pose.roll, pose.pitch, pose.yaw)
    uv = project_points_to_panorama(
        dataset.points, pose.position, R, img_w, img_h, yaw_offset_deg=offset,
    )
    result = match_points_to_detections(uv, detections, pano_w=float(img_w))

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    font = _load_font(32)
    small_font = _load_font(20)

    for i, det in enumerate(detections):
        label = f"{i}:{det.get('label', '?')} {det.get('score', 0):.2f}"
        _draw_bbox(draw, det["bbox"], img_w, (255, 210, 0), label, small_font)

    draw_idx = _subsample_indices(len(uv), args.draw_points)
    for idx in draw_idx:
        u, v = uv[idx]
        if result.hit_mask[idx]:
            color = (255, 40, 40)
            r = 4
        else:
            color = (0, 220, 255)
            r = 2
        draw.ellipse((u - r, v - r, u + r, v + r), fill=color)

    total_hits = int(result.hit_mask.sum())
    draw.rectangle((18, 18, 1280, 140), fill=(0, 0, 0))
    draw.text((34, 30), image_path.name, fill=(255, 255, 255), font=font)
    draw.text(
        (34, 82),
        f"yellow=bbox, cyan=projected points, red=points inside bbox, offset={offset:.1f} deg",
        fill=(255, 255, 255),
        font=small_font,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_door_window_match.jpg"
    vis.save(out_path, quality=92)

    print(f"[image] {image_path.name}")
    print(f"[detections] {len(detections)}")
    print(f"[points] {len(dataset.points):,} projected, {total_hits:,} inside detections")
    print(f"[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
