"""Generate a raw-vs-offset panorama projection check image.

Usage:
    cd /Users/gengchen/Desktop/3dtiqu
    .venv/bin/python 3d_viewer/validate_projection.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORKSPACE = HERE.parent

from core.dataset import find_default_dataset, load_dataset
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


def _pick_sample_points(points: np.ndarray, pose_pos: np.ndarray, count: int) -> np.ndarray:
    rel = points.astype(np.float64) - pose_pos.reshape(1, 3)
    dist = np.linalg.norm(rel, axis=1)
    z_abs = np.abs(rel[:, 2])
    mask = (dist > 2.0) & (dist < 12.0) & (z_abs < 2.0)
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        candidates = np.arange(len(points))
    if len(candidates) <= count:
        return candidates

    order = np.argsort(dist[candidates])
    spread = np.linspace(0, len(order) - 1, count, dtype=np.int64)
    return candidates[order[spread]]


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    color: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
):
    x, y = xy
    r = 14
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=5)
    draw.line((x - r, y, x + r, y), fill=color, width=3)
    draw.line((x, y - r, x, y + r), fill=color, width=3)
    draw.text((x + r + 4, y - r), label, fill=color, font=font)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate panorama projection validation image")
    ap.add_argument("--keyframe", type=int, default=0, help="keyframe index to inspect")
    ap.add_argument("--points", type=int, default=12, help="number of sampled point-cloud points")
    ap.add_argument("--offset", type=float, default=-90.0, help="comparison yaw offset in degrees")
    ap.add_argument("--max-points", type=int, default=80_000, help="LAS sample size for this check")
    ap.add_argument("--output", default=str(WORKSPACE / "out" / "projection_validation"))
    args = ap.parse_args()

    cfg = find_default_dataset(WORKSPACE)
    if cfg is None:
        raise RuntimeError(f"no dataset found under {WORKSPACE}")
    dataset = load_dataset(cfg, max_points=args.max_points)
    if not dataset.poses:
        raise RuntimeError("dataset has no poses")

    pose = dataset.poses[max(0, min(args.keyframe, len(dataset.poses) - 1))]
    image_path = dataset.image_dir / pose.image_name
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    R = rotation_from_angle(pose.roll, pose.pitch, pose.yaw)

    sample_idx = _pick_sample_points(dataset.points, pose.position, args.points)
    sample_points = dataset.points[sample_idx]
    raw_uv = project_points_to_panorama(sample_points, pose.position, R, img_w, img_h, 0.0)
    offset_uv = project_points_to_panorama(sample_points, pose.position, R, img_w, img_h, args.offset)

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    font = _load_font(34)
    small_font = _load_font(24)
    draw.rectangle((18, 18, 880, 138), fill=(0, 0, 0))
    draw.text((34, 30), f"{pose.image_name}", fill=(255, 255, 255), font=font)
    draw.text((34, 82), "red=raw 0 deg, cyan=offset %.1f deg" % args.offset,
              fill=(255, 255, 255), font=small_font)

    for n, (raw, shifted) in enumerate(zip(raw_uv, offset_uv), 1):
        _draw_marker(draw, tuple(raw), (255, 70, 70), f"r{n}", small_font)
        _draw_marker(draw, tuple(shifted), (0, 220, 255), f"o{n}", small_font)
        draw.line((raw[0], raw[1], shifted[0], shifted[1]), fill=(255, 255, 255), width=1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(pose.image_name).stem}_projection_check.jpg"
    vis.save(out_path, quality=92)

    print(f"[dataset] {dataset.data_root.name}")
    print(f"[keyframe] {args.keyframe}: {pose.image_name}")
    print(f"[points] {len(sample_idx)} sampled from {dataset.points.shape[0]:,} loaded points")
    print(f"[out] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
