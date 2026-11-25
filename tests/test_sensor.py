"""Automated unit tests for the integrated 6-axis MPU-6050 interface."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpu6050_sim import MPU6050, MPU6050Config, STANDARD_GRAVITY

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class TestMPU6050Sensor(unittest.TestCase):
    def test_from_config_and_reproducibility(self) -> None:
        s1 = MPU6050.from_config(DEFAULT_CONFIG)
        s2 = MPU6050.from_config(DEFAULT_CONFIG)

        acc_in = np.tile([0.0, 0.0, STANDARD_GRAVITY], (500, 1))
        gyr_in = np.zeros((500, 3))

        r1 = s1.simulate(acc_in, gyr_in)
        r2 = s2.simulate(acc_in, gyr_in)

        np.testing.assert_array_equal(r1.accelerometer, r2.accelerometer)
        np.testing.assert_array_equal(r1.gyroscope, r2.gyroscope)
        np.testing.assert_array_equal(r1.raw_accelerometer, r2.raw_accelerometer)
        np.testing.assert_array_equal(r1.raw_gyroscope, r2.raw_gyroscope)

    def test_reset_restores_exact_sequence(self) -> None:
        sensor = MPU6050.from_config(DEFAULT_CONFIG)
        m1 = [sensor.step([0.0, 0.0, STANDARD_GRAVITY], [0.1, -0.2, 0.3]) for _ in range(25)]
        sensor.reset()
        m2 = [sensor.step([0.0, 0.0, STANDARD_GRAVITY], [0.1, -0.2, 0.3]) for _ in range(25)]
        for a, b in zip(m1, m2):
            self.assertAlmostEqual(a.timestamp_s, b.timestamp_s)
            np.testing.assert_array_equal(a.accelerometer, b.accelerometer)
            np.testing.assert_array_equal(a.gyroscope, b.gyroscope)

    def test_independent_rng_streams_between_accel_and_gyro(self) -> None:
        s1 = MPU6050.from_config(DEFAULT_CONFIG)
        s2 = MPU6050.from_config(DEFAULT_CONFIG)
        s2.update_parameters(gyroscope={"noise_density_radps_per_sqrthz": [0.01, 0.01, 0.01]})

        acc_in = np.tile([0.0, 0.0, STANDARD_GRAVITY], (200, 1))
        gyr_in = np.zeros((200, 3))
        r1 = s1.simulate(acc_in, gyr_in)
        r2 = s2.simulate(acc_in, gyr_in)

        # Changing gyroscope noise must NOT affect accelerometer outputs
        np.testing.assert_array_equal(r1.accelerometer, r2.accelerometer)
        self.assertFalse(np.allclose(r1.gyroscope, r2.gyroscope))

    def test_csv_export_and_yaml_roundtrip(self) -> None:
        sensor = MPU6050.from_config(DEFAULT_CONFIG)
        t = sensor.time_vector(duration_s=0.1)  # 20 samples at 200 Hz
        acc_in = np.tile([0.0, 0.0, STANDARD_GRAVITY], (t.size, 1))
        gyr_in = np.zeros((t.size, 3))
        result = sensor.simulate(acc_in, gyr_in)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = result.to_csv(Path(tmpdir) / "sim.csv")
            with csv_path.open("r", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(len(rows), t.size + 1)
            self.assertEqual(rows[0][0], "time_s")
            self.assertEqual(len(rows[0]), 19)

            yaml_path = Path(tmpdir) / "roundtrip.yaml"
            sensor.config.to_yaml(yaml_path)
            reloaded = MPU6050Config.from_yaml(yaml_path)
            self.assertEqual(reloaded, sensor.config)

    def test_mismatched_input_lengths_raise(self) -> None:
        sensor = MPU6050.from_config(DEFAULT_CONFIG)
        with self.assertRaises(ValueError):
            sensor.simulate(np.zeros((10, 3)), np.zeros((12, 3)))


if __name__ == "__main__":
    unittest.main()
