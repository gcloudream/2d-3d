from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.detection_cache import (
    annotation_output_path,
    detection_json_candidates,
    detection_output_path,
    find_detection_json,
)


class DetectionCacheTest(unittest.TestCase):
    def test_prefers_new_output_cache_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "out" / "door_window_annotations").mkdir(parents=True)
            (workspace / "out" / "door_window_detections").mkdir(parents=True)
            (workspace / "2d" / "out").mkdir(parents=True)
            ann = workspace / "out" / "door_window_annotations" / "foo.json"
            first = workspace / "out" / "door_window_detections" / "foo.json"
            second = workspace / "2d" / "out" / "foo.json"
            ann.write_text("{}", encoding="utf-8")
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")

            candidates = detection_json_candidates(workspace, "foo.jpg")
            found = find_detection_json(workspace, "foo.jpg")

            self.assertEqual(candidates[0], ann)
            self.assertEqual(found, ann)

    def test_builds_output_path_from_image_stem(self):
        workspace = Path("/tmp/work")
        out = detection_output_path(workspace, "608.675997_IMG.jpg")
        self.assertEqual(
            out,
            workspace / "out" / "door_window_detections" / "608.675997_IMG.json",
        )

    def test_candidates_do_not_include_recall_mode_cache(self):
        workspace = Path("/tmp/work")
        candidates = detection_json_candidates(workspace, "608.675997_IMG.jpg")
        self.assertNotIn(
            workspace / "out" / "door_window_detections" / "608.675997_IMG_recall.json",
            candidates,
        )

    def test_annotation_cache_takes_priority_over_detection_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "out" / "door_window_annotations").mkdir(parents=True)
            (workspace / "out" / "door_window_detections").mkdir(parents=True)
            ann = annotation_output_path(workspace, "foo.jpg")
            det = detection_output_path(workspace, "foo.jpg")
            ann.write_text("{}", encoding="utf-8")
            det.write_text("{}", encoding="utf-8")

            found = find_detection_json(workspace, "foo.jpg")
            self.assertEqual(found, ann)


if __name__ == "__main__":
    unittest.main()
