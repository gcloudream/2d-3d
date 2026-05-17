from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.detection_runner import build_detection_command


class DetectionRunnerTest(unittest.TestCase):
    def test_precise_mode_uses_stricter_threshold_and_excludes(self):
        cmd = build_detection_command(
            Path("/workspace"),
            Path("/workspace/data/foo.jpg"),
            mode="precise",
        )

        self.assertIn("--score-thr", cmd)
        self.assertEqual(cmd[cmd.index("--score-thr") + 1], "0.1")
        self.assertEqual(cmd[cmd.index("--pano-split") + 1], "6")
        self.assertEqual(cmd[cmd.index("--pano-out-size") + 1], "1024")
        self.assertIn("--exclude", cmd)
        self.assertIn("monitor", cmd[cmd.index("--exclude") + 1])
        self.assertNotIn("glass partition", cmd[cmd.index("--classes") + 1])

    def test_recall_mode_uses_lower_threshold_and_richer_prompts(self):
        cmd = build_detection_command(
            Path("/workspace"),
            Path("/workspace/data/foo.jpg"),
            mode="recall",
        )

        self.assertEqual(cmd[cmd.index("--score-thr") + 1], "0.06")
        self.assertIn("glass partition", cmd[cmd.index("--classes") + 1])
        self.assertIn("glass cabinet door", cmd[cmd.index("--classes") + 1])


if __name__ == "__main__":
    unittest.main()
