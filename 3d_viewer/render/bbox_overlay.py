"""Draw door/window detection boxes on the panorama sphere."""
from __future__ import annotations

import moderngl
import numpy as np

from core.bbox_overlay import bbox_edge_samples, uv_to_pano_local_dirs


_VERT_SRC = """
#version 330
uniform mat4 mvp;
uniform vec3 cam_pos;
uniform float point_size;
in vec3 in_dir;
in vec3 in_color;
out vec3 v_color;
void main() {
    vec3 world_pos = cam_pos + in_dir * 49.7;
    v_color = in_color;
    vec4 clip = mvp * vec4(world_pos, 1.0);
    gl_Position = vec4(clip.xy, clip.w * 0.9999, clip.w);
    gl_PointSize = point_size;
}
"""

_FRAG_SRC = """
#version 330
in vec3 v_color;
out vec4 frag;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if (length(c) > 0.5) discard;
    frag = vec4(v_color, 1.0);
}
"""


def _color_for_detection(det: dict, selected: bool) -> tuple[float, float, float]:
    if selected:
        return (0.0, 1.0, 0.45)
    label = str(det.get("label", "")).lower()
    if label == "door":
        return (1.0, 0.45, 0.10)
    if label == "window":
        return (0.0, 0.85, 1.0)
    return (1.0, 0.9, 0.0)


class BboxOverlay:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)
        self.vbo_pos: moderngl.Buffer | None = None
        self.vbo_col: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None
        self.n = 0
        self._geometry_key = None
        self._color_key = None
        self._vertex_counts: list[int] = []

    def clear(self):
        if self.vbo_pos is not None:
            self.vbo_pos.release()
            self.vbo_pos = None
        if self.vbo_col is not None:
            self.vbo_col.release()
            self.vbo_col = None
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        self.n = 0
        self._geometry_key = None
        self._color_key = None
        self._vertex_counts = []

    def _release_color_vao(self):
        if self.vbo_col is not None:
            self.vbo_col.release()
            self.vbo_col = None
        if self.vao is not None:
            self.vao.release()
            self.vao = None
        self._color_key = None

    def set_detections(
        self,
        detections: list[dict],
        img_w: int,
        img_h: int,
        R_pano: np.ndarray,
        yaw_offset_deg: float,
        selected_index: int = -1,
    ):
        if not detections or img_w <= 0 or img_h <= 0:
            self.clear()
            return

        bboxes = tuple(tuple(float(v) for v in det["bbox"]) for det in detections)
        R_key = np.asarray(R_pano, dtype=np.float32).reshape(9).tobytes()
        geometry_key = (bboxes, int(img_w), int(img_h), float(yaw_offset_deg), R_key)
        labels = tuple(str(det.get("label", "")).lower() for det in detections)
        color_key = (labels, int(selected_index), geometry_key)

        if self.vbo_pos is None or geometry_key != self._geometry_key:
            if self.vbo_pos is not None:
                self.vbo_pos.release()
                self.vbo_pos = None
            self._release_color_vao()
            all_dirs: list[np.ndarray] = []
            vertex_counts: list[int] = []
            R_inv = np.asarray(R_pano, dtype=np.float32).T
            for det in detections:
                uv = bbox_edge_samples(det["bbox"], pano_w=float(img_w), samples_per_edge=48)
                dirs_local = uv_to_pano_local_dirs(uv, float(img_w), float(img_h), yaw_offset_deg)
                dirs_world = dirs_local @ R_inv.T
                all_dirs.append(dirs_world.astype(np.float32, copy=False))
                vertex_counts.append(len(dirs_world))
            pos = np.ascontiguousarray(np.concatenate(all_dirs, axis=0))
            self.vbo_pos = self.ctx.buffer(pos.tobytes())
            self.n = len(pos)
            self._vertex_counts = vertex_counts
            self._geometry_key = geometry_key

        if self.vbo_col is not None and self.vao is not None and color_key == self._color_key:
            return

        self._release_color_vao()
        all_cols: list[np.ndarray] = []
        for i, det in enumerate(detections):
            count = self._vertex_counts[i] if i < len(self._vertex_counts) else 0
            color = np.asarray(
                _color_for_detection(det, i == int(selected_index)),
                dtype=np.float32,
            )
            all_cols.append(np.repeat(color.reshape(1, 3), count, axis=0))
        col = np.ascontiguousarray(np.concatenate(all_cols, axis=0))
        self.vbo_col = self.ctx.buffer(col.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [
            (self.vbo_pos, "3f", "in_dir"),
            (self.vbo_col, "3f", "in_color"),
        ])
        self._color_key = color_key

    def render(self, mvp: np.ndarray, cam_pos: np.ndarray):
        if self.vao is None or self.n == 0:
            return
        self.prog["mvp"].write(mvp.T.tobytes())
        self.prog["cam_pos"].value = tuple(float(v) for v in cam_pos)
        self.prog["point_size"].value = 4.0
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.vao.render(mode=moderngl.POINTS)
        self.ctx.enable(moderngl.DEPTH_TEST)
