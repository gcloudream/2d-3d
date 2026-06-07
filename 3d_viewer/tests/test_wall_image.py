from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.wall_image import generate_wall_density_image, preserve_wall_density_grid


class WallImageTest(unittest.TestCase):
    def test_generates_topdown_density_image_and_metadata(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data_root = workspace / "room_a"
            data_root.mkdir()

            result = generate_wall_density_image(
                workspace,
                data_root,
                points,
                resolution_m=0.5,
                height_percentiles=(0.0, 100.0),
            )

            self.assertEqual(result.image_path, workspace / "out" / "wall_images" / "room_a_wall_density.png")
            self.assertEqual(result.preserved_image_path, workspace / "out" / "wall_images" / "room_a_wall_preserved.png")
            self.assertEqual(result.metadata_path, workspace / "out" / "wall_images" / "room_a_wall_density.json")
            self.assertTrue(result.image_path.exists())
            self.assertTrue(result.preserved_image_path.exists())
            self.assertTrue(result.metadata_path.exists())

            with Image.open(result.image_path) as image:
                self.assertEqual(image.size, (3, 3))
            with Image.open(result.preserved_image_path) as image:
                self.assertEqual(image.size, (3, 3))

            payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "topdown_density_and_preserved_wall_density")
            self.assertEqual(payload["selected_points"], 5)
            self.assertEqual(payload["total_points"], 5)
            self.assertEqual(payload["resolution_m"], 0.5)
            self.assertIn("wall_pixel_count", payload)

    def test_preserves_weak_wall_runs_connected_to_strong_wall_seed(self):
        grid = np.zeros((20, 20), dtype=np.uint32)
        grid[2:18, 4] = 3
        grid[6:10, 4] = 8
        grid[12, 5:17] = 3
        grid[12, 9:13] = 8
        grid[2:5, 14:17] = 8
        vertical_span = np.zeros_like(grid, dtype=np.float32)
        vertical_span[2:18, 4] = 1.5
        vertical_span[12, 5:17] = 1.5
        vertical_span[2:5, 14:17] = 0.4
        height_layers = np.zeros_like(grid, dtype=np.uint8)
        height_layers[2:18, 4] = 4
        height_layers[12, 5:17] = 4
        height_layers[2:5, 14:17] = 1

        mask = preserve_wall_density_grid(
            grid,
            vertical_span,
            height_layers,
            resolution_m=0.1,
            min_wall_run_m=0.8,
            max_wall_gap_m=0.2,
            weak_density_percentile=20.0,
            strong_density_percentile=80.0,
        )

        self.assertTrue(mask[10, 4])
        self.assertTrue(mask[12, 10])
        self.assertFalse(mask[3, 15])

    def test_rejects_empty_cloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "empty"):
                generate_wall_density_image(
                    Path(tmp),
                    Path(tmp) / "room",
                    np.empty((0, 3), dtype=np.float64),
                )


if __name__ == "__main__":
    unittest.main()
