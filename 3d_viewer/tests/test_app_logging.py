from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.app_logging import (
    operation_events_path,
    operation_log_path,
    log_operation,
    reset_operation_logging,
)


class AppLoggingTest(unittest.TestCase):
    def tearDown(self):
        reset_operation_logging()

    @staticmethod
    def _flush_logging():
        for handler in logging.getLogger("3d_viewer").handlers:
            handler.flush()

    def test_log_operation_writes_text_and_structured_event_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            event_path = log_operation(
                workspace,
                "unit_test_event",
                component="tests",
                payload={"count": np.int64(3), "center": np.asarray([1.0, 2.0, 3.0])},
            )

            self.assertEqual(event_path, operation_events_path(workspace))
            payload = json.loads(event_path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(payload["event"], "unit_test_event")
            self.assertEqual(payload["component"], "tests")
            self.assertEqual(payload["payload"]["count"], 3)
            self.assertEqual(payload["payload"]["center"], [1.0, 2.0, 3.0])
            self._flush_logging()
            text = operation_log_path(workspace).read_text(encoding="utf-8")
            self.assertIn("unit_test_event", text)

    def test_log_operation_does_not_write_text_log_to_previous_workspace(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            workspace_a = Path(first)
            workspace_b = Path(second)

            log_operation(workspace_a, "first_event", component="tests")
            log_operation(workspace_b, "second_event", component="tests")
            self._flush_logging()

            first_text = operation_log_path(workspace_a).read_text(encoding="utf-8")
            second_text = operation_log_path(workspace_b).read_text(encoding="utf-8")
            self.assertIn("first_event", first_text)
            self.assertNotIn("second_event", first_text)
            self.assertIn("second_event", second_text)


if __name__ == "__main__":
    unittest.main()
