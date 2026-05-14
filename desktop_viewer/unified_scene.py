from __future__ import annotations

import ctypes
from pathlib import Path

import cv2
import numpy as np
from OpenGL import GL
from PySide6.QtCore import Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from data import CameraPose
from projection import project_points_to_pano, rasterize_points_to_overlay


VERTEX_SHADER = """
#version 120
attribute vec2 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;

void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 120
varying vec2 v_uv;
uniform sampler2D u_image;

void main() {
    gl_FragColor = texture2D(u_image, v_uv);
}
"""


def _compile_shader(source: str, shader_type: int) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        raise RuntimeError(log)
    return shader


def _link_program() -> int:
    vertex_shader = _compile_shader(VERTEX_SHADER, GL.GL_VERTEX_SHADER)
    fragment_shader = _compile_shader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
    program = GL.glCreateProgram()
    if program == 0:
        raise RuntimeError("OpenGL did not create a shader program.")
    GL.glAttachShader(program, vertex_shader)
    GL.glAttachShader(program, fragment_shader)
    GL.glBindAttribLocation(program, 0, "a_pos")
    GL.glBindAttribLocation(program, 1, "a_uv")
    GL.glLinkProgram(program)
    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        raise RuntimeError(log)
    return program


def _viewer_rotation(yaw: float, pitch: float) -> np.ndarray:
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)

    yaw_matrix = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pitch_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, sp],
            [0.0, -sp, cp],
        ],
        dtype=np.float32,
    )
    return pitch_matrix @ yaw_matrix


def _render_equirectangular_view(
    source: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    width: int,
    height: int,
) -> np.ndarray:
    if source is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    src_h, src_w = source.shape[:2]
    aspect = width / max(1, height)
    tan_half = np.tan(np.radians(fov) / 2.0)

    xs = ((np.arange(width, dtype=np.float32) + 0.5) / width) * 2.0 - 1.0
    ys = 1.0 - ((np.arange(height, dtype=np.float32) + 0.5) / height) * 2.0
    grid_x, grid_y = np.meshgrid(xs, ys)

    ray_view = np.stack(
        (
            grid_x * aspect * tan_half,
            np.ones_like(grid_x),
            grid_y * tan_half,
        ),
        axis=-1,
    )

    inv_view_rot = np.linalg.inv(_viewer_rotation(yaw, pitch)).astype(np.float32)
    ray_local = ray_view @ inv_view_rot.T
    ray_norm = np.linalg.norm(ray_local, axis=-1, keepdims=True)
    ray_local = ray_local / np.maximum(ray_norm, 1e-8)

    map_x = (0.5 + np.arctan2(ray_local[..., 0], ray_local[..., 1]) / (2.0 * np.pi)) * src_w
    horizontal = np.sqrt(ray_local[..., 0] ** 2 + ray_local[..., 1] ** 2)
    map_y = (np.arctan2(horizontal, ray_local[..., 2]) / np.pi) * src_h

    map_x = np.mod(map_x, src_w).astype(np.float32)
    map_y = np.clip(map_y, 0.0, src_h - 1.0).astype(np.float32)
    frame = cv2.remap(source, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return np.ascontiguousarray(frame)


class UnifiedSceneView(QOpenGLWidget):
    """A single panoramic space with point-cloud overlay."""

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.points_world = np.empty((0, 3), dtype=np.float32)
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self.pose: CameraPose | None = None

        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = 82.0
        self.point_size = 2
        self.show_panorama = True
        self.show_points = True
        self._last_mouse_pos = None

        self.program = 0
        self.u_image_location = -1
        self.quad_vbo = 0
        self.background_texture = 0
        self.overlay_texture = 0
        self.initialized = False
        self.init_error: str | None = None

        self._background_source_rgb: np.ndarray | None = None
        self._overlay_source_rgba: np.ndarray | None = None
        self._background_frame_rgb: np.ndarray | None = None
        self._overlay_frame_rgba: np.ndarray | None = None
        self._overlay_source_dirty = True
        self._frame_dirty = True
        self._texture_dirty = True
        self._viewport_size = (0, 0)

    def set_world_points(self, points: np.ndarray, colors: np.ndarray) -> None:
        self.points_world = points.astype(np.float32, copy=False)
        self.colors = colors.astype(np.uint8, copy=False)
        self._overlay_source_dirty = True
        self._frame_dirty = True
        self._texture_dirty = True
        self.update()

    def set_pose(self, pose: CameraPose, image_path: Path) -> None:
        self.pose = pose
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取全景图: {image_path}")
        self._background_source_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.yaw = 0.0
        self.pitch = 0.0
        self._overlay_source_dirty = True
        self._frame_dirty = True
        self._texture_dirty = True
        self.update()

    def set_point_size(self, point_size: int) -> None:
        self.point_size = max(1, int(point_size))
        self._overlay_source_dirty = True
        self._frame_dirty = True
        self._texture_dirty = True
        self.update()

    def set_show_panorama(self, show: bool) -> None:
        self.show_panorama = show
        self._frame_dirty = True
        self.update()

    def set_show_points(self, show: bool) -> None:
        self.show_points = show
        self._overlay_source_dirty = True
        self._frame_dirty = True
        self.update()

    def reset_view(self) -> None:
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = 82.0
        self._frame_dirty = True
        self.update()

    def initializeGL(self) -> None:  # type: ignore[override]
        try:
            GL.glClearColor(0.055, 0.062, 0.074, 1.0)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

            self.program = _link_program()
            self.u_image_location = GL.glGetUniformLocation(self.program, "u_image")

            quad = np.array(
                [
                    -1.0, -1.0, 0.0, 1.0,
                    1.0, -1.0, 1.0, 1.0,
                    -1.0, 1.0, 0.0, 0.0,
                    1.0, 1.0, 1.0, 0.0,
                ],
                dtype=np.float32,
            )
            self.quad_vbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.quad_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

            self.background_texture = GL.glGenTextures(1)
            self.overlay_texture = GL.glGenTextures(1)
            for texture, fmt in (
                (self.background_texture, GL.GL_RGB),
                (self.overlay_texture, GL.GL_RGBA),
            ):
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            self.initialized = True
        except Exception as exc:
            self.initialized = False
            self.init_error = str(exc)
            print(f"OpenGL 初始化失败: {self.init_error}")

    def paintGL(self) -> None:  # type: ignore[override]
        GL.glClearColor(0.055, 0.062, 0.074, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self.initialized or self.program == 0:
            return

        dpr = max(1.0, float(self.devicePixelRatioF()))
        width = max(1, int(round(self.width() * dpr)))
        height = max(1, int(round(self.height() * dpr)))
        GL.glViewport(0, 0, width, height)

        self._update_sources_if_needed()
        if self._texture_dirty:
            self._upload_texture(self.background_texture, self._background_frame_rgb, GL.GL_RGB)
            self._upload_texture(self.overlay_texture, self._overlay_frame_rgba, GL.GL_RGBA)
            self._texture_dirty = False

        self._draw_texture(self.background_texture)
        if self.show_points:
            self._draw_texture(self.overlay_texture, blend=True)

    def resizeGL(self, width: int, height: int) -> None:  # type: ignore[override]
        self._frame_dirty = True

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._last_mouse_pos = event.position()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._last_mouse_pos is None or not (event.buttons() & Qt.LeftButton):
            return
        pos = event.position()
        delta = pos - self._last_mouse_pos
        self._last_mouse_pos = pos
        self.yaw -= float(delta.x()) * 0.006
        self.pitch += float(delta.y()) * 0.006
        limit = np.radians(88.0)
        self.pitch = float(np.clip(self.pitch, -limit, limit))
        self._frame_dirty = True
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._last_mouse_pos = None

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        steps = event.angleDelta().y() / 120.0
        self.fov = float(np.clip(self.fov - steps * 3.0, 35.0, 115.0))
        self._frame_dirty = True
        self.update()

    def _update_sources_if_needed(self) -> None:
        if self._overlay_source_dirty:
            self._overlay_source_dirty = False
            self._overlay_source_rgba = self._build_overlay_source()
        if self._frame_dirty:
            self._frame_dirty = False
            self._update_frames()
            self._texture_dirty = True

    def _build_overlay_source(self) -> np.ndarray | None:
        if self._background_source_rgb is None:
            return None

        height, width = self._background_source_rgb.shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        if not self.show_points or self.pose is None or len(self.points_world) == 0:
            return overlay

        uv, valid = project_points_to_pano(
            self.points_world,
            self.pose.position,
            self.pose.roll,
            self.pose.pitch,
            self.pose.yaw,
            width,
            height,
        )
        if not np.any(valid):
            return overlay

        points = self.points_world[valid]
        colors = self.colors[valid]
        distances = np.linalg.norm(points - self.pose.position.reshape(1, 3), axis=1).astype(np.float32)
        uv = uv[valid]
        return rasterize_points_to_overlay(
            uv,
            distances,
            colors,
            width,
            height,
            self.point_size,
        )

    def _update_frames(self) -> None:
        dpr = max(1.0, float(self.devicePixelRatioF()))
        width = max(1, int(round(self.width() * dpr)))
        height = max(1, int(round(self.height() * dpr)))
        if self._viewport_size != (width, height):
            self._viewport_size = (width, height)

        if self.show_panorama and self._background_source_rgb is not None:
            self._background_frame_rgb = _render_equirectangular_view(
                self._background_source_rgb,
                self.yaw,
                self.pitch,
                self.fov,
                width,
                height,
            )
        else:
            self._background_frame_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        if self._overlay_source_rgba is None:
            self._overlay_frame_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        else:
            if not self.show_points:
                self._overlay_frame_rgba = np.zeros((height, width, 4), dtype=np.uint8)
                return
            self._overlay_frame_rgba = _render_equirectangular_view(
                self._overlay_source_rgba,
                self.yaw,
                self.pitch,
                self.fov,
                width,
                height,
            )

    def _upload_texture(self, texture: int, frame: np.ndarray | None, fmt: int) -> None:
        if texture == 0 or frame is None:
            return
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            fmt,
            width,
            height,
            0,
            fmt,
            GL.GL_UNSIGNED_BYTE,
            frame,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _draw_texture(self, texture: int, blend: bool = False) -> None:
        if texture == 0:
            return
        if blend:
            GL.glEnable(GL.GL_BLEND)
        else:
            GL.glDisable(GL.GL_BLEND)

        GL.glUseProgram(self.program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        if self.u_image_location >= 0:
            GL.glUniform1i(self.u_image_location, 0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.quad_vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, ctypes.c_void_p(8))
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glDisableVertexAttribArray(0)
        GL.glDisableVertexAttribArray(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
