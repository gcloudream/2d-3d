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
    set_scene_pair_highlight_style,
    set_scene_pair_point_size,
    set_scene_pair_selected_depth_test,
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

    def set_selected_depth_test(self, on: bool):
        self.calls.append(("set_selected_depth_test", bool(on)))

    def set_point_size(self, size: float):
        self.calls.append(("set_point_size", float(size)))

    def set_highlight_mask(self, mask):
        self.calls.append(("set_highlight_mask", mask))

    def set_highlight_style(self, ring_color, fill_color):
        self.calls.append(("set_highlight_style", tuple(ring_color), tuple(fill_color)))


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
            ("set_selected_depth_test", True),
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

    def test_set_scene_pair_selected_depth_test_updates_both_scenes(self):
        primary = FakeScene()
        observer = FakeScene()

        set_scene_pair_selected_depth_test(primary, observer, False)

        self.assertEqual(primary.calls, [("set_selected_depth_test", False)])
        self.assertEqual(observer.calls, [("set_selected_depth_test", False)])

    def test_set_scene_pair_highlight_style_updates_both_scenes(self):
        primary = FakeScene()
        observer = FakeScene()

        set_scene_pair_highlight_style(primary, observer, (0.0, 0.85, 1.0), (0.0, 0.2, 1.0))

        expected = [("set_highlight_style", (0.0, 0.85, 1.0), (0.0, 0.2, 1.0))]
        self.assertEqual(primary.calls, expected)
        self.assertEqual(observer.calls, expected)


if __name__ == "__main__":
    unittest.main()
