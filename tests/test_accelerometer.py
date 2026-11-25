"""Automated unit tests for the MPU-6050 3-axis accelerometer model."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpu6050_sim import (
    ACCEL_SENSITIVITY_LSB_PER_G,
    STANDARD_GRAVITY,
    Accelerometer,
    AccelerometerConfig,
)


class TestAccelerometer(unittest.TestCase):
    def test_deterministic_scale_misalignment_and_bias(self) -> None:
        cfg = AccelerometerConfig(
            full_scale_range_g=4,
            scale_factor_error=(0.02, -0.01, 0.03),
            misalignment=(
                (0.0, 0.01, -0.02),
                (-0.01, 0.0, 0.015),
                (0.02, -0.015, 0.0),
            ),
            static_bias_mps2=(0.1, -0.2, 0.3),
            noise_density_mps2_per_sqrthz=(0.0, 0.0, 0.0),
            bias_instability_mps2=(0.0, 0.0, 0.0),
            random_walk_mps2_per_sqrts=(0.0, 0.0, 0.0),
            enable_saturation=False,
            enable_quantization=False,
        )
        accel = Accelerometer(cfg, sample_rate_hz=100.0, rng=np.random.default_rng(0))
        u = np.array([1.0, -2.0, STANDARD_GRAVITY])
        expected = (
            (np.eye(3) + np.diag(cfg.scale_factor_error) + np.asarray(cfg.misalignment)) @ u
            + np.asarray(cfg.static_bias_mps2)
        )
        meas = accel.step(u)
        np.testing.assert_allclose(meas.output, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(meas.ideal_input, u)

    def test_saturation_clamping_to_fsr(self) -> None:
        cfg = AccelerometerConfig(
            full_scale_range_g=2,
            scale_factor_error=(0.0, 0.0, 0.0),
            static_bias_mps2=(0.0, 0.0, 0.0),
            noise_density_mps2_per_sqrthz=(0.0, 0.0, 0.0),
            bias_instability_mps2=(0.0, 0.0, 0.0),
            random_walk_mps2_per_sqrts=(0.0, 0.0, 0.0),
            enable_saturation=True,
            enable_quantization=False,
        )
        accel = Accelerometer(cfg, sample_rate_hz=100.0, rng=np.random.default_rng(1))
        fsr = 2.0 * STANDARD_GRAVITY
        meas = accel.step([50.0, -50.0, 5.0])
        np.testing.assert_allclose(meas.output, [fsr, -fsr, 5.0], rtol=1e-12, atol=1e-12)

    def test_16bit_adc_quantization_and_counts(self) -> None:
        for fsr_g, lsb_per_g in ACCEL_SENSITIVITY_LSB_PER_G.items():
            cfg = AccelerometerConfig(
                full_scale_range_g=fsr_g,
                scale_factor_error=(0.0, 0.0, 0.0),
                static_bias_mps2=(0.0, 0.0, 0.0),
                noise_density_mps2_per_sqrthz=(0.0, 0.0, 0.0),
                bias_instability_mps2=(0.0, 0.0, 0.0),
                random_walk_mps2_per_sqrts=(0.0, 0.0, 0.0),
                enable_saturation=True,
                enable_quantization=True,
            )
            accel = Accelerometer(cfg, sample_rate_hz=100.0, rng=np.random.default_rng(2))
            meas = accel.step([0.0, 0.0, STANDARD_GRAVITY])  # Exactly +1 g on Z
            self.assertEqual(int(meas.raw_counts[2]), int(round(lsb_per_g)))
            self.assertLessEqual(
                abs(meas.output[2] - STANDARD_GRAVITY), 0.5 * cfg.quantization_step_mps2 + 1e-12
            )

    def test_step_and_batch_simulate_noise_statistics(self) -> None:
        cfg = AccelerometerConfig(
            static_bias_mps2=(0.0, 0.0, 0.0),
            scale_factor_error=(0.0, 0.0, 0.0),
            bias_instability_mps2=(0.0, 0.0, 0.0),
            random_walk_mps2_per_sqrts=(0.0, 0.0, 0.0),
            enable_quantization=False,
        )
        fs = 200.0
        accel = Accelerometer(cfg, sample_rate_hz=fs, rng=np.random.default_rng(99))
        out, _ = accel.simulate(np.zeros((50_000, 3)))
        expected_std = cfg.noise_density_mps2_per_sqrthz[0] * np.sqrt(fs)
        empirical_std = np.std(out, axis=0)
        np.testing.assert_allclose(empirical_std, expected_std, rtol=0.03)

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AccelerometerConfig(full_scale_range_g=3)
        with self.assertRaises(ValueError):
            AccelerometerConfig(misalignment=((0.01, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        with self.assertRaises(ValueError):
            AccelerometerConfig(bias_correlation_time_s=(0.0, 10.0, 10.0))


if __name__ == "__main__":
    unittest.main()
