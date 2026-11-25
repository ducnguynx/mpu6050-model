"""Automated unit tests for the MPU-6050 3-axis gyroscope model."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpu6050_sim import (
    GYRO_SENSITIVITY_LSB_PER_DPS,
    Gyroscope,
    GyroscopeConfig,
)


class TestGyroscope(unittest.TestCase):
    def test_deterministic_scale_misalignment_and_bias(self) -> None:
        cfg = GyroscopeConfig(
            full_scale_range_dps=500,
            scale_factor_error=(0.015, -0.02, 0.01),
            misalignment=(
                (0.0, -0.01, 0.005),
                (0.012, 0.0, -0.008),
                (-0.005, 0.008, 0.0),
            ),
            static_bias_radps=(0.02, -0.015, 0.01),
            noise_density_radps_per_sqrthz=(0.0, 0.0, 0.0),
            bias_instability_radps=(0.0, 0.0, 0.0),
            random_walk_radps_per_sqrts=(0.0, 0.0, 0.0),
            enable_saturation=False,
            enable_quantization=False,
        )
        gyro = Gyroscope(cfg, sample_rate_hz=200.0, rng=np.random.default_rng(0))
        w = np.array([0.5, -0.3, 1.2])
        expected = (
            (np.eye(3) + np.diag(cfg.scale_factor_error) + np.asarray(cfg.misalignment)) @ w
            + np.asarray(cfg.static_bias_radps)
        )
        meas = gyro.step(w)
        np.testing.assert_allclose(meas.output, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(meas.ideal_input, w)

    def test_saturation_clamping_to_fsr(self) -> None:
        cfg = GyroscopeConfig(
            full_scale_range_dps=250,
            scale_factor_error=(0.0, 0.0, 0.0),
            static_bias_radps=(0.0, 0.0, 0.0),
            noise_density_radps_per_sqrthz=(0.0, 0.0, 0.0),
            bias_instability_radps=(0.0, 0.0, 0.0),
            random_walk_radps_per_sqrts=(0.0, 0.0, 0.0),
            enable_saturation=True,
            enable_quantization=False,
        )
        gyro = Gyroscope(cfg, sample_rate_hz=200.0, rng=np.random.default_rng(1))
        fsr = math.radians(250.0)
        meas = gyro.step([10.0, -10.0, 0.5])
        np.testing.assert_allclose(meas.output, [fsr, -fsr, 0.5], rtol=1e-12, atol=1e-12)

    def test_16bit_adc_quantization_and_counts(self) -> None:
        for fsr_dps, lsb_per_dps in GYRO_SENSITIVITY_LSB_PER_DPS.items():
            cfg = GyroscopeConfig(
                full_scale_range_dps=fsr_dps,
                scale_factor_error=(0.0, 0.0, 0.0),
                static_bias_radps=(0.0, 0.0, 0.0),
                noise_density_radps_per_sqrthz=(0.0, 0.0, 0.0),
                bias_instability_radps=(0.0, 0.0, 0.0),
                random_walk_radps_per_sqrts=(0.0, 0.0, 0.0),
                enable_saturation=True,
                enable_quantization=True,
            )
            gyro = Gyroscope(cfg, sample_rate_hz=200.0, rng=np.random.default_rng(2))
            rate_10dps = math.radians(10.0)
            meas = gyro.step([rate_10dps, 0.0, 0.0])
            self.assertEqual(int(meas.raw_counts[0]), int(round(10.0 * lsb_per_dps)))
            self.assertLessEqual(
                abs(meas.output[0] - rate_10dps), 0.5 * cfg.quantization_step_radps + 1e-12
            )

    def test_datasheet_rms_noise_at_200hz(self) -> None:
        # At 200 Hz (Nyquist BW = 100 Hz), default N_g gives exactly 0.05 deg/s-rms
        cfg = GyroscopeConfig(
            static_bias_radps=(0.0, 0.0, 0.0),
            scale_factor_error=(0.0, 0.0, 0.0),
            bias_instability_radps=(0.0, 0.0, 0.0),
            random_walk_radps_per_sqrts=(0.0, 0.0, 0.0),
            enable_quantization=False,
        )
        gyro = Gyroscope(cfg, sample_rate_hz=200.0, rng=np.random.default_rng(7))
        out, _ = gyro.simulate(np.zeros((100_000, 3)))
        rms_dps = np.std(out, axis=0) * (180.0 / math.pi)
        np.testing.assert_allclose(rms_dps, 0.05, rtol=0.02)

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GyroscopeConfig(full_scale_range_dps=300)
        with self.assertRaises(ValueError):
            GyroscopeConfig(noise_density_radps_per_sqrthz=(-1e-5, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
