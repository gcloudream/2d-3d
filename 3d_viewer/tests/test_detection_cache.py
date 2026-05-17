from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.detection_cache import (
    detection_json_candidates,
    detection_output_path,
    find_detection_json,
)


class DetectionCacheTest(unittest.TestCase):
    def test_prefers_new_output_cache_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "out" / "door_window_detections").mkdir(parents=True)
            (workspace / "2d" / "out").mkdir(parents=True)
            first = workspace / "out" / "door_window_detections" / "foo.json"
            second = workspace / "2d" / "out" / "foo.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")

            candidates = detection_json_candidates(workspace, "foo.jpg")
            found = find_detection_json(workspace, "foo.jpg")

            self.assertEqual(candidates[0], first)
            self.assertEqual(found, first)

    def test_builds_output_path_from_image_stem(self):
        workspace = Path("/tmp/work")
        out = detection_output_path(workspace, "608.675997_IMG.jpg")
        self.assertEqual(
            out,
            workspace / "out" / "door_window_detections" / "608.675997_IMG.json",
        )

    def test_recall_mode_uses_separate_cache_file(self):
        workspace = Path("/tmp/work")
        out = detection_output_path(workspace, "608.675997_IMG.jpg", mode="recall")
        self.assertEqual(
            out,
            workspace / "out" / "door_window_detections" / "608.675997_IMG_recall.json",
        )


if __name__ == "__main__":
    unittest.main()
