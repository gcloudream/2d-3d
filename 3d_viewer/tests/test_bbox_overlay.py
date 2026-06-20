from __future__ import annotations

import sys
import unittest
from pathlib import Path

import moderngl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.bbox_overlay import bbox_edge_samples, split_wrapped_bbox, uv_to_pano_local_dirs
from render.bbox_overlay import BboxOverlay, _color_for_detection


class _FakeBuffer:
    def __init__(self, data: bytes):
        self.data = data
        self.released = False

    def release(self):
        self.released = True


class _FakeVertexArray:
    def __init__(self, buffers):
        self.buffers = buffers
        self.released = False
        self.render_modes: list[int] = []

    def release(self):
        self.released = True

    def render(self, mode):
        self.render_modes.append(mode)


class _FakeUniform:
    def __init__(self):
        self.value = None
        self.writes: list[bytes] = []

    def write(self, data: bytes):
        self.writes.append(data)


class _FakeProgram:
    def __init__(self):
        self.uniforms: dict[str, _FakeUniform] = {}

    def __getitem__(self, key: str):
        return self.uniforms.setdefault(key, _FakeUniform())


class _FakeContext:
    def __init__(self):
        self.buffers: list[_FakeBuffer] = []
        self.vertex_arrays: list[_FakeVertexArray] = []
        self.enabled: list[int] = []
        self.disabled: list[int] = []

    def program(self, **_kwargs):
        return _FakeProgram()

    def buffer(self, data: bytes):
        buf = _FakeBuffer(data)
        self.buffers.append(buf)
        return buf

    def vertex_array(self, _prog, buffers):
        vao = _FakeVertexArray(buffers)
        self.vertex_arrays.append(vao)
        return vao

    def enable(self, flag):
        self.enabled.append(flag)

    def disable(self, flag):
        self.disabled.append(flag)


class BboxOverlayTest(unittest.TestCase):
    def test_regular_bbox_stays_single_box(self):
        boxes = split_wrapped_bbox([10.0, 20.0, 30.0, 40.0], pano_w=100.0)

        self.assertEqual(boxes, [[10.0, 20.0, 30.0, 40.0]])

    def test_wrapped_bbox_splits_at_panorama_seam(self):
        boxes = split_wrapped_bbox([90.0, 20.0, 10.0, 40.0], pano_w=100.0)

        self.assertEqual(boxes, [
            [90.0, 20.0, 100.0, 40.0],
            [0.0, 20.0, 10.0, 40.0],
        ])

    def test_edge_samples_form_line_segments(self):
        segments = bbox_edge_samples([10.0, 20.0, 30.0, 40.0], pano_w=100.0, samples_per_edge=3)

        self.assertEqual(segments.shape, (16, 2))
        np.testing.assert_allclose(segments[0], [10.0, 20.0])
        np.testing.assert_allclose(segments[1], [20.0, 20.0])
        np.testing.assert_allclose(segments[-2], [10.0, 30.0])
        np.testing.assert_allclose(segments[-1], [10.0, 40.0])

    def test_uv_to_pano_local_dirs_maps_center_to_front(self):
        dirs = uv_to_pano_local_dirs(
            np.array([[2880.0, 1440.0]], dtype=np.float32),
            img_w=5760.0,
            img_h=2880.0,
            yaw_offset_deg=0.0,
        )

        np.testing.assert_allclose(dirs[0], [0.0, 1.0, 0.0], atol=1e-6)

    def test_uv_to_pano_local_dirs_removes_display_yaw_offset(self):
        raw = uv_to_pano_local_dirs(
            np.array([[2880.0, 1440.0]], dtype=np.float32),
            img_w=5760.0,
            img_h=2880.0,
            yaw_offset_deg=0.0,
        )
        shifted = uv_to_pano_local_dirs(
            np.array([[1440.0, 1440.0]], dtype=np.float32),
            img_w=5760.0,
            img_h=2880.0,
            yaw_offset_deg=-90.0,
        )

        np.testing.assert_allclose(shifted, raw, atol=1e-6)

    def test_selected_detection_color_is_distinct_from_point_highlight(self):
        self.assertEqual(_color_for_detection({"label": "window"}, selected=True), (0.0, 1.0, 0.45))

    def test_render_overlay_reuses_position_buffer_when_only_selected_box_changes(self):
        ctx = _FakeContext()
        overlay = BboxOverlay(ctx)
        detections = [
            {"label": "window", "bbox": [10.0, 20.0, 30.0, 40.0]},
            {"label": "door", "bbox": [50.0, 20.0, 70.0, 40.0]},
        ]
        R = np.eye(3, dtype=np.float32)

        overlay.set_detections(detections, 100, 50, R, 0.0, selected_index=0)
        first_pos = overlay.vbo_pos
        first_col = overlay.vbo_col
        first_vao = overlay.vao
        self.assertEqual(len(ctx.buffers), 2)

        overlay.set_detections(detections, 100, 50, R, 0.0, selected_index=1)

        self.assertIs(overlay.vbo_pos, first_pos)
        self.assertIsNot(overlay.vbo_col, first_col)
        self.assertIsNot(overlay.vao, first_vao)
        self.assertFalse(first_pos.released)
        self.assertTrue(first_col.released)
        self.assertTrue(first_vao.released)
        self.assertEqual(len(ctx.buffers), 3)

    def test_overlay_renders_bbox_edges_as_points_to_avoid_stray_lines(self):
        ctx = _FakeContext()
        overlay = BboxOverlay(ctx)
        overlay.set_detections(
            [{"label": "window", "bbox": [10.0, 20.0, 30.0, 40.0]}],
            100,
            50,
            np.eye(3, dtype=np.float32),
            0.0,
            selected_index=0,
        )

        overlay.render(np.eye(4, dtype=np.float32), np.zeros(3, dtype=np.float32))

        self.assertEqual(overlay.vao.render_modes, [moderngl.POINTS])
        self.assertNotIn(moderngl.LINES, overlay.vao.render_modes)


if __name__ == "__main__":
    unittest.main()
