"""Generate a prototype 3D wall model for the default dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core.dataset import find_default_dataset, load_dataset
from core.wall_model import generate_wall_model
from core.wall_openings import load_wall_openings

WORKSPACE = HERE.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read the scan point cloud and generate wall-line previews plus an OBJ wall model.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE,
        help="Workspace containing the scan dataset and output directory.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=300_000,
        help="Maximum sampled points loaded from the LAS file.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Top-down wall grid resolution in meters per pixel.",
    )
    parser.add_argument(
        "--wall-thickness",
        type=float,
        default=0.14,
        help="Generated wall thickness in meters.",
    )
    parser.add_argument(
        "--min-wall-length",
        type=float,
        default=2.0,
        help="Minimum main wall length in meters before short-wall recovery.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = find_default_dataset(args.workspace)
    if cfg is None:
        raise RuntimeError(f"no dataset found under {args.workspace}")
    dataset = load_dataset(cfg, max_points=args.max_points)
    openings = load_wall_openings(args.workspace, dataset.data_root)
    result = generate_wall_model(
        args.workspace,
        dataset.data_root,
        dataset.points,
        openings=openings,
        resolution_m=args.resolution,
        wall_thickness_m=args.wall_thickness,
        min_wall_length_m=args.min_wall_length,
    )
    print(f"obj: {result.obj_path}")
    print(f"preview: {result.preview_path}")
    print(f"topdown: {result.topdown_preview_path}")
    print(f"metadata: {result.metadata_path}")
    print(f"segments: {result.segment_count}")
    print(
        f"openings: {result.matched_opening_count} matched, "
        f"{result.projected_opening_count} projected, "
        f"{result.unmatched_opening_count} unmatched"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
