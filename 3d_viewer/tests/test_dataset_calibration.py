from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.dataset import CameraPose, apply_pano_calibration, dataset_config_from_root, load_pano_calibration


class DatasetCalibrationTest(unittest.TestCase):
    def test_builds_dataset_config_from_selected_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scan-a"
            (root / "CAM").mkdir(parents=True)
            (root / "LAS_Rgb").mkdir()
            camera_file = root / "CAM" / "camera_pos.cam"
            las_file = root / "LAS_Rgb" / "scan-a_rgb_0.las"
            camera_file.write_text("", encoding="utf-8")
            las_file.write_bytes(b"")

            cfg = dataset_config_from_root(root)

        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.data_root, root)
        self.assertEqual(cfg.camera_file, camera_file)
        self.assertEqual(cfg.image_dir, root / "CAM")
        self.assertEqual(cfg.pointcloud_file, las_file)

    def test_loads_panorama_camera_calibration_from_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calib_dir = root / "CALIBRATION_CAMERA"
            calib_dir.mkdir()
            (calib_dir / "CAMERA_PANO_123.yaml").write_text(
                "\n".join([
                    "camera_model: panorama",
                    "width: 5760",
                    "height: 2880",
                    "extrinsicTrans: [0, -0.06, 0.15]",
                    "extrinsicAngle: [0.03, 0.07, 1.62]",
                    "timeBias: 0",
                ]),
                encoding="utf-8",
            )

            calibration = load_pano_calibration(root)

            self.assertIsNotNone(calibration)
            np.testing.assert_allclose(calibration.translation, [0.0, -0.06, 0.15])
            np.testing.assert_allclose(calibration.angle, [0.03, 0.07, 1.62])
            self.assertAlmostEqual(calibration.default_yaw_offset_deg, -92.819162, places=5)

    def test_applies_panorama_translation_in_body_frame_to_pose_position(self):
        pose = CameraPose(
            image_name="frame.jpg",
            x=10.0,
            y=20.0,
            z=30.0,
            roll=0.0,
            pitch=0.0,
            yaw=np.pi / 2.0,
            timestamp=1.0,
        )

        calibrated = apply_pano_calibration(
            [pose],
            translation=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

        self.assertEqual(calibrated[0].image_name, "frame.jpg")
        np.testing.assert_allclose(calibrated[0].position, [12.0, 19.0, 33.0], atol=1e-6)
        np.testing.assert_allclose(calibrated[0].body_position, [10.0, 20.0, 30.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
