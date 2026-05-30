from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.scene_sync import (
    configure_observer_scene,
    set_scene_pair_highlight_mask,
    set_scene_pair_point_size,
)


class FakeScene:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def set_show_pano(self, on: bool):
        self.calls.append(("set_show_pano", bool(on)))

    def set_show_pc(self, on: bool):
        self.calls.append(("set_show_pc", bool(on)))

    def set_show_bboxes(self, on: bool):
        self.calls.append(("set_show_bboxes", bool(on)))

    def set_pick_mode(self, on: bool):
        self.calls.append(("set_pick_mode", bool(on)))

    def set_global_view_mode(self, on: bool):
        self.calls.append(("set_global_view_mode", bool(on)))

    def set_point_size(self, size: float):
        self.calls.append(("set_point_size", float(size)))

    def set_highlight_mask(self, mask):
        self.calls.append(("set_highlight_mask", mask))


class SceneSyncTest(unittest.TestCase):
    def test_configure_observer_scene_disables_interaction_layers(self):
        scene = FakeScene()

        configure_observer_scene(scene)

        self.assertEqual(scene.calls, [
            ("set_show_pano", False),
            ("set_show_pc", True),
            ("set_show_bboxes", False),
            ("set_pick_mode", False),
            ("set_global_view_mode", True),
        ])

    def test_set_scene_pair_point_size_updates_both_scenes(self):
        primary = FakeScene()
        observer = FakeScene()

        set_scene_pair_point_size(primary, observer, 4.5)

        self.assertEqual(primary.calls, [("set_point_size", 4.5)])
        self.assertEqual(observer.calls, [("set_point_size", 4.5)])

    def test_set_scene_pair_highlight_mask_updates_both_scenes(self):
        primary = FakeScene()
        observer = FakeScene()
        mask = np.array([True, False, True])

        set_scene_pair_highlight_mask(primary, observer, mask)

        self.assertIs(primary.calls[0][1], mask)
        self.assertIs(observer.calls[0][1], mask)


if __name__ == "__main__":
    unittest.main()
