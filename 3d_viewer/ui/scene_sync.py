"""Small helpers for keeping the two SceneView panes in sync."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class SceneLike(Protocol):
    def set_show_pano(self, on: bool): ...
    def set_show_pc(self, on: bool): ...
    def set_show_bboxes(self, on: bool): ...
    def set_pick_mode(self, on: bool): ...
    def set_point_size(self, size: float): ...
    def set_highlight_mask(self, mask: np.ndarray | None): ...
    def set_selected_depth_test(self, on: bool): ...


def configure_observer_scene(scene: SceneLike):
    scene.set_show_pano(False)
    scene.set_show_pc(True)
    scene.set_show_bboxes(False)
    scene.set_pick_mode(False)
    if hasattr(scene, "set_global_view_mode"):
        scene.set_global_view_mode(True)
    if hasattr(scene, "set_selected_depth_test"):
        scene.set_selected_depth_test(True)


def set_scene_pair_point_size(primary: SceneLike, observer: SceneLike, size: float):
    value = float(size)
    primary.set_point_size(value)
    observer.set_point_size(value)


def set_scene_pair_highlight_mask(
    primary: SceneLike,
    observer: SceneLike,
    mask: np.ndarray | None,
):
    primary.set_highlight_mask(mask)
    observer.set_highlight_mask(mask)
