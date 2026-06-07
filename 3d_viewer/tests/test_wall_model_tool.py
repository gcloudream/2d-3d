from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from wall_model_tool import _ObjModelGLWindow, build_arg_parser, load_obj_triangles


class WallModelToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_parser_accepts_workspace_sampling_and_auto_generate(self):
        parser = build_arg_parser()

        args = parser.parse_args([
            "--workspace",
            "/tmp/scan",
            "--max-points",
            "12345",
            "--auto-generate",
        ])

        self.assertEqual(args.workspace, Path("/tmp/scan"))
        self.assertEqual(args.max_points, 12345)
        self.assertTrue(args.auto_generate)

    def test_load_obj_triangles_triangulates_quad_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj_path = Path(tmp) / "wall.obj"
            obj_path.write_text(
                "\n".join([
                    "v 0 0 0",
                    "v 1 0 0",
                    "v 1 1 0",
                    "v 0 1 0",
                    "f 1 2 3 4",
                ])
                + "\n",
                encoding="utf-8",
            )

            triangles = load_obj_triangles(obj_path)

        self.assertEqual(triangles.shape, (6, 3))
        self.assertEqual(triangles[0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(triangles[5].tolist(), [0.0, 1.0, 0.0])

    def test_obj_preview_shows_mesh_edges_by_default(self):
        view = _ObjModelGLWindow()
        self.addCleanup(view.close)

        self.assertTrue(view.show_mesh_edges)


if __name__ == "__main__":
    unittest.main()
