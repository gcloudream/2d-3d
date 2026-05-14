"""全景球：内表面贴 equirectangular 全景图，永远以相机为球心。

全景图作为背景先渲染，点云随后覆盖在它之上。
"""
from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from core.projection import rotation_from_angle


_VERT_SRC = """
#version 330
uniform mat4 mvp;
uniform vec3 cam_pos;
in vec3 in_pos;
out vec3 v_dir;
void main() {
    vec3 world_pos = cam_pos + in_pos;
    v_dir = in_pos;
    vec4 clip = mvp * vec4(world_pos, 1.0);
    gl_Position = vec4(clip.xy, clip.w * 0.99999, clip.w);
}
"""

# 把方向 v_dir 旋转到 pano 局部坐标系，再用 projectToPanoramic.coordinate_to_pixel 的公式取像素。
# 这里 R_pano (uniform) 是 world->pano_local 的 3x3。
_FRAG_SRC = """
#version 330
uniform sampler2D pano;
uniform mat3 R_pano;        // world -> pano_local
uniform float img_w;
uniform float img_h;
in vec3 v_dir;
out vec4 frag;

#define PI 3.141592653589793

void main() {
    vec3 p = R_pano * v_dir;
    // 复刻 coordinate_to_pixel: 先算水平角 tm = atan(-x/y)
    float u_pix;
    float tm_h;
    if (p.y == 0.0) tm_h = 0.0; else tm_h = atan(-p.x / p.y);

    bool left = (p.x < 0.0);
    bool front = (p.y > 0.0);
    if (left && front)        u_pix = img_h - img_h * tm_h / PI;
    else if (left && !front)  u_pix = img_h * (-tm_h) / PI;
    else if (!left && front)  u_pix = img_h - img_h * tm_h / PI;
    else                      u_pix = img_w - img_h * tm_h / PI;
    u_pix = mod(u_pix, img_w);

    float horiz = sqrt(p.x * p.x + p.y * p.y);
    float tm_v;
    if (p.z == 0.0) tm_v = PI * 0.5; else tm_v = atan(horiz / p.z);
    if (tm_v < 0.0) tm_v += PI;
    float v_pix = mod(img_h * tm_v / PI, img_h);

    vec2 uv = vec2(u_pix / img_w, v_pix / img_h);
    frag = vec4(texture(pano, uv).rgb, 1.0);
}
"""


def _make_inward_sphere(radius: float = 50.0, lat_seg: int = 48, lon_seg: int = 96) -> np.ndarray:
    """生成内表面（法线朝心）的球面三角形顶点。"""
    verts: list[tuple[float, float, float]] = []
    for i in range(lat_seg):
        v0 = i / lat_seg
        v1 = (i + 1) / lat_seg
        for j in range(lon_seg):
            u0 = j / lon_seg
            u1 = (j + 1) / lon_seg

            def _xyz(u, v):
                theta = u * 2 * np.pi
                phi = v * np.pi
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                return (x, y, z)

            p00 = _xyz(u0, v0)
            p01 = _xyz(u0, v1)
            p10 = _xyz(u1, v0)
            p11 = _xyz(u1, v1)
            verts.append(p00); verts.append(p10); verts.append(p11)
            verts.append(p00); verts.append(p11); verts.append(p01)
    return np.asarray(verts, dtype=np.float32)


class PanoSphere:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)
        verts = _make_inward_sphere(radius=50.0)
        self.vbo = ctx.buffer(verts.tobytes())
        self.vao = ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")
        self.tex: moderngl.Texture | None = None
        self.img_w = 1.0
        self.img_h = 1.0
        self.R_pano = np.eye(3, dtype=np.float32)
        self._loaded_path: Path | None = None

    def load_image(self, path: Path):
        if self._loaded_path == path and self.tex is not None:
            return
        img = Image.open(path).convert("RGBA")
        if img.width > 4096:
            target = (4096, max(1, round(img.height * 4096 / img.width)))
            img = img.resize(target, Image.Resampling.LANCZOS)
        # Equirectangular 顶部对应 phi=0（北极），与公式一致。
        # PIL 第一行就是图像顶行；ModernGL 采样 v=0 读上传数据第一行。
        # 因此这里不做 flip，否则会和 coordinate_to_pixel 的 v_pix 方向相反。
        data = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        h, w, _ = data.shape
        if self.tex is not None:
            self.tex.release()
        self.tex = self.ctx.texture((w, h), 4, data.tobytes())
        # shader 已经对 u_pix 做 mod；这里用 clamp 避免 macOS/Qt 上 NPOT+repeat 被判成不可采样。
        self.tex.repeat_x = False
        self.tex.repeat_y = False
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.img_w = float(w)
        self.img_h = float(h)
        self._loaded_path = path

    def set_pose(self, roll: float, pitch: float, yaw: float):
        self.R_pano = rotation_from_angle(roll, pitch, yaw).astype(np.float32)

    def render(self, mvp: np.ndarray, cam_pos: np.ndarray):
        if self.tex is None:
            return
        self.tex.use(0)
        self.prog["mvp"].write(mvp.T.tobytes())
        self.prog["cam_pos"].value = tuple(float(v) for v in cam_pos)
        self.prog["R_pano"].write(self.R_pano.T.tobytes())
        self.prog["img_w"].value = self.img_w
        self.prog["img_h"].value = self.img_h
        self.prog["pano"].value = 0
        # 全景作为背景先画，点云随后用 depth test 覆盖在它之上。
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.vao.render()
        self.ctx.enable(moderngl.DEPTH_TEST)
