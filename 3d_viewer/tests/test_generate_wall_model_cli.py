from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_wall_model import build_arg_parser


class GenerateWallModelCliTest(unittest.TestCase):
    def test_parser_accepts_workspace_and_sampling_options(self):
        parser = build_arg_parser()

        args = parser.parse_args([
            "--workspace",
            "/tmp/scan",
            "--max-points",
            "12345",
            "--resolution",
            "0.1",
        ])

        self.assertEqual(args.workspace, Path("/tmp/scan"))
        self.assertEqual(args.max_points, 12345)
        self.assertEqual(args.resolution, 0.1)


if __name__ == "__main__":
    unittest.main()
