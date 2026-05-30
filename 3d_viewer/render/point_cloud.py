"""点云渲染：VBO + shader，支持选中高亮。"""
from __future__ import annotations

import moderngl
import numpy as np


_VERT_SRC = """
#version 330
uniform mat4 mvp;
uniform float point_size;
uniform int highlight_idx;
in vec3 in_pos;
in vec3 in_color;
flat out int v_id;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_color = in_color;
    v_id = gl_VertexID;
    gl_PointSize = (gl_VertexID == highlight_idx) ? point_size * 4.0 : point_size;
}
"""

_FRAG_SRC = """
#version 330
flat in int v_id;
in vec3 v_color;
uniform int highlight_idx;
out vec4 frag;
void main() {
    if (v_id == highlight_idx) {
        // 黄色描边的高亮点
        vec2 c = gl_PointCoord - vec2(0.5);
        float d = length(c);
        if (d > 0.5) discard;
        if (d > 0.4) frag = vec4(1.0, 0.95, 0.0, 1.0);
        else         frag = vec4(1.0, 1.0, 1.0, 1.0);
    } else {
        frag = vec4(v_color, 1.0);
    }
}
"""

_SELECTED_VERT_SRC = """
#version 330
uniform mat4 mvp;
uniform float point_size;
in vec3 in_pos;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    gl_PointSize = point_size;
}
"""

_SELECTED_FRAG_SRC = """
#version 330
out vec4 frag;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    float d = length(c);
    if (d > 0.5) discard;
    if (d > 0.38) frag = vec4(1.0, 1.0, 0.0, 1.0);
    else          frag = vec4(1.0, 0.0, 0.0, 1.0);
}
"""


class PointCloud:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)
        self.selected_prog = ctx.program(
            vertex_shader=_SELECTED_VERT_SRC,
            fragment_shader=_SELECTED_FRAG_SRC,
        )
        self.vbo_pos: moderngl.Buffer | None = None
        self.vbo_col: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None
        self.selected_vbo: moderngl.Buffer | None = None
        self.selected_vao: moderngl.VertexArray | None = None
        self.selected_n = 0
        self.n = 0
        self.points: np.ndarray | None = None      # 仍保留供 KDTree 用
        self.colors: np.ndarray | None = None
        self.point_size = 2.0
        self.highlight = -1
        self.selected_depth_test = False

    def set_selected_depth_test(self, on: bool):
        self.selected_depth_test = bool(on)

    def upload(self, points: np.ndarray, colors: np.ndarray):
        for buf in (self.vbo_pos, self.vbo_col):
            if buf is not None:
                buf.release()
        if self.vao is not None:
            self.vao.release()
        if self.selected_vao is not None:
            self.selected_vao.release()
        if self.selected_vbo is not None:
            self.selected_vbo.release()
        pos = points.astype(np.float32, copy=False)
        col = (colors.astype(np.float32) / 255.0).astype(np.float32)
        self.vbo_pos = self.ctx.buffer(pos.tobytes())
        self.vbo_col = self.ctx.buffer(col.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [
            (self.vbo_pos, "3f", "in_pos"),
            (self.vbo_col, "3f", "in_color"),
        ])
        self.n = len(points)
        self.points = pos
        self.colors = colors
        self.highlight = -1
        self.selected_vbo = None
        self.selected_vao = None
        self.selected_n = 0

    def set_highlight_mask(self, mask: np.ndarray | None):
        if self.n == 0 or self.points is None:
            return
        if self.selected_vao is not None:
            self.selected_vao.release()
            self.selected_vao = None
        if self.selected_vbo is not None:
            self.selected_vbo.release()
            self.selected_vbo = None
        if mask is None:
            self.selected_n = 0
            return
        data = np.asarray(mask, dtype=bool)
        if len(data) != self.n:
            raise ValueError(f"highlight mask length {len(data)} != point count {self.n}")
        selected = np.ascontiguousarray(self.points[data].astype(np.float32, copy=False))
        self.selected_n = len(selected)
        if self.selected_n == 0:
            return
        self.selected_vbo = self.ctx.buffer(selected.tobytes())
        self.selected_vao = self.ctx.simple_vertex_array(
            self.selected_prog, self.selected_vbo, "in_pos",
        )

    def render(self, mvp: np.ndarray):
        if self.vao is None or self.n == 0:
            return
        self.prog["mvp"].write(mvp.T.tobytes())
        self.prog["point_size"].value = float(self.point_size)
        self.prog["highlight_idx"].value = int(self.highlight)
        # ModernGL 在 macOS 上要显式开 PROGRAM_POINT_SIZE
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.vao.render(mode=moderngl.POINTS)

        if self.selected_vao is not None and self.selected_n > 0:
            self.selected_prog["mvp"].write(mvp.T.tobytes())
            self.selected_prog["point_size"].value = max(8.0, float(self.point_size) * 5.0)
            if not self.selected_depth_test:
                self.ctx.disable(moderngl.DEPTH_TEST)
            self.selected_vao.render(mode=moderngl.POINTS)
            if not self.selected_depth_test:
                self.ctx.enable(moderngl.DEPTH_TEST)
