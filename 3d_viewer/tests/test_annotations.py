from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.annotations import save_manual_annotations


class ManualAnnotationsTest(unittest.TestCase):
    def test_saves_manual_annotation_json_for_image_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            image_path = workspace / "dataset" / "CAM" / "608.675997_IMG.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"not a real image")
            detections = [
                {"label": "window", "score": 1.0, "source": "manual", "bbox": [10, 20, 30, 40]},
                {"label": "door", "score": 1.0, "source": "manual", "bbox": [50, 60, 70, 80]},
            ]

            out = save_manual_annotations(
                workspace=workspace,
                image_path=image_path,
                width=5760,
                height=2880,
                detections=detections,
            )

            self.assertEqual(out, workspace / "out" / "door_window_annotations" / "608.675997_IMG.json")
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["image"], str(image_path))
            self.assertEqual(payload["width"], 5760)
            self.assertEqual(payload["height"], 2880)
            self.assertEqual(payload["model"], "manual")
            self.assertEqual(payload["source"], "pano_annotation_editor")
            self.assertEqual(payload["detections"], detections)


if __name__ == "__main__":
    unittest.main()
