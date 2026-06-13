from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.wall_openings import (
    WallOpening,
    append_wall_opening,
    load_wall_openings,
    opening_from_selection,
    save_wall_openings,
    wall_openings_path,
)


class WallOpeningsTest(unittest.TestCase):
    def test_saves_and_loads_openings_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            opening = WallOpening(
                id="window-0001",
                label="window",
                source_image="608.jpg",
                seed_index=12,
                point_count=4,
                confidence="high",
                reason="accepted_vertical_planar_region",
                center=(1.0, 2.0, 1.2),
                normal=(1.0, 0.0, 0.0),
                bbox_min=(0.8, 1.6, 0.9),
                bbox_max=(1.2, 2.4, 1.5),
                width_m=0.8,
                height_m=0.6,
                z_min=0.9,
                z_max=1.5,
                detection_index=3,
                score=0.91,
            )

            save_wall_openings(workspace, data_root, [opening])
            loaded = load_wall_openings(workspace, data_root)

        self.assertEqual(loaded, [opening])

    def test_append_assigns_next_stable_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            opening = WallOpening(
                id="",
                label="door",
                source_image="frame.jpg",
                seed_index=2,
                point_count=3,
                confidence="high",
                reason="accepted_vertical_planar_region",
                center=(0.0, 0.0, 1.0),
                normal=(0.0, 1.0, 0.0),
                bbox_min=(-0.4, -0.1, 0.0),
                bbox_max=(0.4, 0.1, 2.0),
                width_m=0.8,
                height_m=2.0,
                z_min=0.0,
                z_max=2.0,
                detection_index=-1,
                score=None,
            )

            saved = append_wall_opening(workspace, data_root, opening)
            loaded = load_wall_openings(workspace, data_root)

        self.assertEqual(saved.id, "door-0001")
        self.assertEqual(loaded[0].id, "door-0001")

    def test_append_reuses_existing_duplicate_opening(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "scan"
            data_root.mkdir()
            opening = WallOpening(
                id="",
                label="window",
                source_image="608.jpg",
                seed_index=2,
                point_count=100,
                confidence="high",
                reason="fused_frustum_and_planar_geometry",
                center=(1.0, 2.0, 1.0),
                normal=(1.0, 0.0, 0.0),
                bbox_min=(0.9, 1.5, 0.2),
                bbox_max=(1.1, 2.5, 1.8),
                width_m=1.0,
                height_m=1.6,
                z_min=0.2,
                z_max=1.8,
                detection_index=4,
                score=1.0,
            )

            first = append_wall_opening(workspace, data_root, opening)
            second = append_wall_opening(workspace, data_root, opening)
            loaded = load_wall_openings(workspace, data_root)

        self.assertEqual(first.id, "window-0001")
        self.assertEqual(second.id, "window-0001")
        self.assertEqual(len(loaded), 1)

    def test_builds_opening_from_selection_mask(self):
        points = np.asarray([
            [1.0, 0.0, 0.8],
            [1.0, 0.4, 1.0],
            [1.0, 0.8, 1.6],
            [3.0, 3.0, 0.0],
        ])
        mask = np.asarray([True, True, True, False])

        opening = opening_from_selection(
            points,
            mask,
            label="window",
            source_image="608.jpg",
            seed_index=1,
            confidence="high",
            reason="accepted_vertical_planar_region",
            plane_point=np.asarray([1.0, 0.4, 1.0]),
            plane_normal=np.asarray([1.0, 0.0, 0.0]),
            width_m=0.8,
            height_m=0.8,
            detection_index=2,
            score=0.8,
        )

        self.assertEqual(opening.label, "window")
        self.assertEqual(opening.point_count, 3)
        self.assertEqual(opening.center, (1.0, 0.4, 1.13333))
        self.assertEqual(opening.bbox_min, (1.0, 0.0, 0.8))
        self.assertEqual(opening.bbox_max, (1.0, 0.8, 1.6))
        self.assertEqual(opening.z_min, 0.8)
        self.assertEqual(opening.z_max, 1.6)

    def test_opening_from_selection_rejects_unrecordable_metadata(self):
        points = np.asarray([
            [1.0, 0.0, 0.8],
            [1.0, 0.4, 1.0],
            [1.0, 0.8, 1.6],
        ])
        mask = np.asarray([True, True, True])
        common = {
            "source_image": "608.jpg",
            "seed_index": 1,
            "confidence": "medium",
            "plane_point": np.asarray([1.0, 0.4, 1.0]),
            "plane_normal": np.asarray([1.0, 0.0, 0.0]),
            "width_m": 0.8,
            "height_m": 0.8,
            "detection_index": 2,
            "score": 0.8,
        }

        bad_cases = [
            {"label": "object", "reason": "accepted_vertical_planar_region"},
            {"label": "window", "reason": "frustum_only_rejected_not_vertical_plane"},
            {"label": "window", "reason": "frustum_only_too_few_region_points"},
        ]
        for bad in bad_cases:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    opening_from_selection(points, mask, **common, **bad)

    def test_wall_openings_path_uses_dataset_name(self):
        path = wall_openings_path(Path("/workspace"), Path("/workspace/scan-a"))

        self.assertEqual(path, Path("/workspace/out/wall_openings/scan-a_openings.json"))


if __name__ == "__main__":
    unittest.main()
