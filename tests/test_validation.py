"""Automated validation tests against Farrell et al. (2022) & El-Sheimy et al. (2008)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpu6050_sim import (
    BIAS_INSTABILITY_AV_FACTOR,
    GAUSS_MARKOV_PEAK_FACTOR,
    GAUSS_MARKOV_PEAK_LAMBDA,
    MPU6050,
    ThreeAxisStochasticError,
    overlapping_allan_variance,
    theoretical_allan_variance,
)

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class TestPaperValidation(unittest.TestCase):
    def test_farrell_2022_gauss_markov_peak_constant(self) -> None:
        # Farrell et al. (2022): sigma_zG(1.89 * T_B) = 0.4365 * sqrt(S_B * T_B)
        self.assertAlmostEqual(GAUSS_MARKOV_PEAK_LAMBDA, 1.89, places=6)
        self.assertAlmostEqual(GAUSS_MARKOV_PEAK_FACTOR, 0.43654, places=4)
        self.assertAlmostEqual(BIAS_INSTABILITY_AV_FACTOR, 0.66428, places=4)

        # Verify theoretical_allan_variance peak matches 0.66428 * B at tau = 1.89 * T_B
        t_b = 50.0
        b_coeff = 2.0e-4
        tau_peak = np.array([1.89 * t_b])
        theo = theoretical_allan_variance(
            tau_peak,
            noise_density=0.0,
            bias_instability=b_coeff,
            bias_correlation_time_s=t_b,
            random_walk=0.0,
        )
        self.assertAlmostEqual(
            math.sqrt(float(theo["gauss_markov"][0])),
            BIAS_INSTABILITY_AV_FACTOR * b_coeff,
            places=10,
        )

    def test_exact_discrete_gauss_markov_variance_preservation(self) -> None:
        # Check stationary variance of discrete GM recurrence equals continuous S_B * T_B / 2
        gen = ThreeAxisStochasticError(
            sample_rate_hz=50.0,
            noise_density=[0.0, 0.0, 0.0],
            bias_instability=[1e-3, 1e-3, 1e-3],
            bias_correlation_time_s=[2.0, 2.0, 2.0],
            random_walk=[0.0, 0.0, 0.0],
            stationary_bias_init=True,
        )
        rng = np.random.default_rng(1234)
        gen.reset(rng)
        seq, _ = gen.generate(100_000, rng)  # 2000 s = 1000 correlation times
        emp_std = float(np.mean(np.std(seq, axis=0)))
        theo_std = float(gen.params.gm_steady_state_std[0])
        self.assertAlmostEqual(emp_std, theo_std, delta=0.06 * theo_std)

    def test_white_noise_allan_deviation_slope_and_level(self) -> None:
        # El-Sheimy (2008) & Farrell (2022): sigma_A(tau) = N / sqrt(tau)
        fs = 100.0
        n_coeff = 1.5e-3
        sensor = MPU6050.from_config(DEFAULT_CONFIG)
        sensor.update_parameters(
            simulation={"sample_rate_hz": fs, "random_seed": 2026, "stationary_bias_init": False},
            gyroscope={
                "static_bias_radps": [0.0, 0.0, 0.0],
                "noise_density_radps_per_sqrthz": [n_coeff, n_coeff, n_coeff],
                "bias_instability_radps": [0.0, 0.0, 0.0],
                "random_walk_radps_per_sqrts": [0.0, 0.0, 0.0],
                "enable_quantization": False,
            },
        )
        res = sensor.simulate(np.zeros((100_000, 3)), np.zeros((100_000, 3)))
        tau, avar = overlapping_allan_variance(res.gyroscope[:, 0], fs, max_clusters=40)
        adev = np.sqrt(avar)
        mask = tau <= 10.0  # Well-averaged region (>= 100 clusters)
        theo = n_coeff / np.sqrt(tau[mask])
        mape = float(np.mean(np.abs(adev[mask] - theo) / theo))
        self.assertLess(mape, 0.05)

        # Log-log slope must be close to -0.5
        slope, _ = np.polyfit(np.log10(tau[mask]), np.log10(adev[mask]), 1)
        self.assertAlmostEqual(float(slope), -0.5, delta=0.04)

    def test_random_walk_allan_deviation_slope_and_level(self) -> None:
        # El-Sheimy (2008) & Farrell (2022): sigma_A(tau) = K * sqrt(tau / 3)
        fs = 50.0
        k_coeff = 2.0e-4
        sensor = MPU6050.from_config(DEFAULT_CONFIG)
        sensor.update_parameters(
            simulation={"sample_rate_hz": fs, "random_seed": 31415, "stationary_bias_init": False},
            gyroscope={
                "static_bias_radps": [0.0, 0.0, 0.0],
                "noise_density_radps_per_sqrthz": [0.0, 0.0, 0.0],
                "bias_instability_radps": [0.0, 0.0, 0.0],
                "random_walk_radps_per_sqrts": [k_coeff, k_coeff, k_coeff],
                "enable_quantization": False,
            },
        )
        res = sensor.simulate(np.zeros((100_000, 3)), np.zeros((100_000, 3)))
        # Average Allan variance across 3 independent axes for tight estimation
        tau, av0 = overlapping_allan_variance(res.gyroscope[:, 0], fs, max_clusters=40)
        _, av1 = overlapping_allan_variance(res.gyroscope[:, 1], fs, max_clusters=40)
        _, av2 = overlapping_allan_variance(res.gyroscope[:, 2], fs, max_clusters=40)
        adev = np.sqrt((av0 + av1 + av2) / 3.0)

        mask = (tau >= 0.2) & (tau <= 20.0)
        theo = k_coeff * np.sqrt(tau[mask] / 3.0)
        mape = float(np.mean(np.abs(adev[mask] - theo) / theo))
        self.assertLess(mape, 0.08)

        # Log-log slope must be close to +0.5
        slope, _ = np.polyfit(np.log10(tau[mask]), np.log10(adev[mask]), 1)
        self.assertAlmostEqual(float(slope), 0.5, delta=0.05)

    def test_combined_mpu6050_allan_deviation_matches_theory(self) -> None:
        # Full default MPU-6050 config (white + GM + RW + 16-bit ADC quantization)
        fs = 50.0
        n_samples = 180_000  # 3600 s
        sensor = MPU6050.from_config(DEFAULT_CONFIG)
        sensor.update_parameters(simulation={"sample_rate_hz": fs, "random_seed": 2022})
        res = sensor.simulate(np.zeros((n_samples, 3)), np.zeros((n_samples, 3)))

        tau_a, av_a = overlapping_allan_variance(res.accelerometer[:, 0], fs, max_clusters=50)
        acc_cfg = sensor.config.accelerometer
        theo_a = np.sqrt(
            theoretical_allan_variance(
                tau_a,
                noise_density=acc_cfg.noise_density_mps2_per_sqrthz[0],
                bias_instability=acc_cfg.bias_instability_mps2[0],
                bias_correlation_time_s=acc_cfg.bias_correlation_time_s[0],
                random_walk=acc_cfg.random_walk_mps2_per_sqrts[0],
                quantization_step=acc_cfg.quantization_step_mps2,
                sample_rate_hz=fs,
            )["total"]
        )
        mask_a = tau_a <= 50.0
        mape_a = float(np.mean(np.abs(np.sqrt(av_a[mask_a]) - theo_a[mask_a]) / theo_a[mask_a]))
        self.assertLess(mape_a, 0.08)

        tau_g, av_g = overlapping_allan_variance(res.gyroscope[:, 0], fs, max_clusters=50)
        gyr_cfg = sensor.config.gyroscope
        theo_g = np.sqrt(
            theoretical_allan_variance(
                tau_g,
                noise_density=gyr_cfg.noise_density_radps_per_sqrthz[0],
                bias_instability=gyr_cfg.bias_instability_radps[0],
                bias_correlation_time_s=gyr_cfg.bias_correlation_time_s[0],
                random_walk=gyr_cfg.random_walk_radps_per_sqrts[0],
                quantization_step=gyr_cfg.quantization_step_radps,
                sample_rate_hz=fs,
            )["total"]
        )
        mask_g = tau_g <= 50.0
        mape_g = float(np.mean(np.abs(np.sqrt(av_g[mask_g]) - theo_g[mask_g]) / theo_g[mask_g]))
        self.assertLess(mape_g, 0.08)


if __name__ == "__main__":
    unittest.main()
