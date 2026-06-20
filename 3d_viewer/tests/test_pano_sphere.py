from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from render.pano_sphere import PanoSphere


class FakeTexture:
    def __init__(self):
        self.repeat_x = True
        self.repeat_y = True
        self.filter = None
        self.mipmaps_built = False
        self.released = False

    def release(self):
        self.released = True

    def build_mipmaps(self):
        self.mipmaps_built = True


class FakeContext:
    def __init__(self):
        self.created_texture: FakeTexture | None = None
        self.texture_size: tuple[int, int] | None = None

    def program(self, **_kwargs):
        return object()

    def buffer(self, _data):
        return object()

    def simple_vertex_array(self, *_args):
        return object()

    def texture(self, size, _components, _data):
        self.texture_size = tuple(size)
        self.created_texture = FakeTexture()
        return self.created_texture


class PanoSphereTest(unittest.TestCase):
    def test_load_image_builds_complete_mipmap_texture_for_macos_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "pano.png"
            data = np.full((8, 16, 3), 128, dtype=np.uint8)
            Image.fromarray(data, mode="RGB").save(image_path)
            ctx = FakeContext()
            pano = PanoSphere(ctx)

            pano.load_image(image_path)

            tex = ctx.created_texture
            self.assertIsNotNone(tex)
            self.assertFalse(tex.repeat_x)
            self.assertFalse(tex.repeat_y)
            self.assertTrue(tex.mipmaps_built)
            self.assertEqual(tex.filter, (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR))

    def test_load_image_caps_large_panoramas_to_existing_viewer_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "large_pano.png"
            data = np.full((2500, 5000, 3), 128, dtype=np.uint8)
            Image.fromarray(data, mode="RGB").save(image_path)
            ctx = FakeContext()
            pano = PanoSphere(ctx)

            pano.load_image(image_path)

            self.assertEqual(ctx.texture_size, (4096, 2048))
            self.assertEqual(pano.img_w, 4096.0)
            self.assertEqual(pano.img_h, 2048.0)


if __name__ == "__main__":
    unittest.main()
