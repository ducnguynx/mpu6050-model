"""Unit tests for black-box MPU-6050 Roll-angle estimation (Pair 2: Wiener vs. Kalman)."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpu6050_sim import MPU6050, STANDARD_GRAVITY
from mpu6050_sim.attitude_filtering import (
    accel_to_roll_atan2,
    apply_wiener_complementary_causal,
    apply_wiener_complementary_freq,
    baseline_complementary_filter,
    baseline_gyro_integration,
    check_linearization_accuracy,
    compute_attitude_innovation_diagnostics,
    identify_attitude_noise_from_static,
    run_attitude_kalman_2state,
)
from mpu6050_sim.filtering import compute_filter_metrics

CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


class TestAttitudeFiltering(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fs = 200.0
        sensor = MPU6050.from_config(CONFIG_PATH)
        sensor.update_parameters(simulation={"sample_rate_hz": cls.fs, "random_seed": 801})
        n_st = int(round(600.0 * cls.fs))
        acc_st = np.tile(np.array([0.0, 0.0, STANDARD_GRAVITY]), (n_st, 1))
        res_st = sensor.simulate(acc_st, np.zeros((n_st, 3)))
        cls.ay_st = res_st.accelerometer[:, 1]
        cls.az_st = res_st.accelerometer[:, 2]
        cls.gx_st = res_st.gyroscope[:, 0]

        # Dynamic sinusoidal Roll + 4 Hz vibration (20 s, separate seed)
        sensor.reset(seed=802)
        t = sensor.time_vector(duration_s=20.0)
        amp = 12.0 * DEG2RAD
        w = 2.0 * math.pi * 0.3
        cls.theta_true = amp * np.sin(w * t)
        cls.omega_true = amp * w * np.cos(w * t)
        a_vib = 1.5 * np.sin(2.0 * math.pi * 4.0 * t)
        acc_dyn = np.column_stack(
            [
                np.zeros_like(t),
                STANDARD_GRAVITY * np.sin(cls.theta_true) + a_vib,
                STANDARD_GRAVITY * np.cos(cls.theta_true),
            ]
        )
        gyr_dyn = np.column_stack([cls.omega_true, np.zeros_like(t), np.zeros_like(t)])
        res_dyn = sensor.simulate(acc_dyn, gyr_dyn)
        cls.ay_dyn = res_dyn.accelerometer[:, 1]
        cls.az_dyn = res_dyn.accelerometer[:, 2]
        cls.gx_dyn = res_dyn.gyroscope[:, 0]

        cls.noise_model = identify_attitude_noise_from_static(
            cls.ay_st,
            cls.az_st,
            cls.gx_st,
            cls.ay_dyn,
            cls.az_dyn,
            cls.gx_dyn,
            fs=cls.fs,
            accel_fsr_mps2=sensor.config.accelerometer.full_scale_range_g * STANDARD_GRAVITY,
            gyro_fsr_radps=sensor.config.gyroscope.full_scale_range_dps * DEG2RAD,
        )
        cls.true_gyro_bias = sensor.config.gyroscope.static_bias_radps[0]

    def test_linearization_bound_for_small_angles(self) -> None:
        info = check_linearization_accuracy(self.theta_true)
        self.assertAlmostEqual(info["max_angle_deg"], 12.0, delta=0.05)
        self.assertLess(info["max_rel_error_pct"], 1.0)

    def test_dual_sensor_blackbox_identification(self) -> None:
        gm = self.noise_model.gyro_rate_model
        am = self.noise_model.accel_angle_model
        # True gyro discrete white noise = 0.05 deg/s = 8.727e-4 rad/s
        self.assertAlmostEqual(gm.sigma_v_hat, 8.727e-4, delta=4e-5)
        # True accel angle white noise = 0.03923 / 9.80665 = 0.0040 rad
        self.assertAlmostEqual(am.sigma_v_hat, 0.0040, delta=2e-4)
        self.assertGreater(self.noise_model.wiener_causal_omega_c_radps, 0.1)
        self.assertLess(self.noise_model.wiener_causal_omega_c_radps, 0.5)

    def test_wiener_and_kalman_reject_vibration_far_better_than_baselines(self) -> None:
        za = accel_to_roll_atan2(
            self.ay_dyn - self.noise_model.ay_bias_hat_mps2,
            self.az_dyn - self.noise_model.az_bias_hat_mps2,
        )
        w_cal = self.gx_dyn - self.noise_model.gyro_rate_model.bias_hat

        b_cf = baseline_complementary_filter(za, w_cal, fs=self.fs, alpha=0.96)
        w_freq = apply_wiener_complementary_freq(za, w_cal, self.noise_model)
        w_causal, _ = apply_wiener_complementary_causal(za, w_cal, self.noise_model)
        kf = run_attitude_kalman_2state(za, self.gx_dyn, self.noise_model, use_calibrated_gyro=False)

        m_raw = compute_filter_metrics(za, self.theta_true, za, fs=self.fs, burn_in=200)
        m_cf = compute_filter_metrics(b_cf, self.theta_true, za, fs=self.fs, burn_in=200)
        m_wf = compute_filter_metrics(w_freq.theta_est_rad, self.theta_true, za, fs=self.fs, burn_in=200)
        m_wc = compute_filter_metrics(w_causal, self.theta_true, za, fs=self.fs, burn_in=200)
        m_kf = compute_filter_metrics(kf.theta_est_rad, self.theta_true, za, fs=self.fs, burn_in=200, est_std=kf.theta_std_rad)

        # Raw accel under 1.5 m/s^2 vibration has > 5 deg RMSE; Wiener & Kalman crush it below 0.3 deg!
        self.assertGreater(m_raw.rmse * RAD2DEG, 5.0)
        self.assertLess(m_wf.rmse * RAD2DEG, 0.30)
        self.assertLess(m_wc.rmse * RAD2DEG, 0.30)
        self.assertLess(m_kf.rmse * RAD2DEG, 0.30)
        self.assertLess(m_wf.rmse, m_cf.rmse * 0.3)
        self.assertLess(m_kf.rmse, m_cf.rmse * 0.3)

    def test_kalman_tracks_uncalibrated_gyro_bias_and_is_nis_consistent(self) -> None:
        # Clean 60 s tilt with +0.01 rad/s uncalibrated thermal bias step
        sensor = MPU6050.from_config(CONFIG_PATH)
        base_b = list(sensor.config.gyroscope.static_bias_radps)
        target_bias = base_b[0] + 0.010
        sensor.update_parameters(
            simulation={"sample_rate_hz": self.fs, "random_seed": 803},
            gyroscope={"static_bias_radps": [target_bias, base_b[1], base_b[2]]},
        )
        t = sensor.time_vector(duration_s=60.0)
        th = (10.0 * DEG2RAD) * np.sin(2.0 * math.pi * 0.2 * t)
        om = (10.0 * DEG2RAD) * (2.0 * math.pi * 0.2) * np.cos(2.0 * math.pi * 0.2 * t)
        acc = np.column_stack([np.zeros_like(t), STANDARD_GRAVITY * np.sin(th), STANDARD_GRAVITY * np.cos(th)])
        gyr = np.column_stack([om, np.zeros_like(t), np.zeros_like(t)])
        res = sensor.simulate(acc, gyr)

        za = accel_to_roll_atan2(
            res.accelerometer[:, 1] - self.noise_model.ay_bias_hat_mps2,
            res.accelerometer[:, 2] - self.noise_model.az_bias_hat_mps2,
        )
        gyr_int = baseline_gyro_integration(res.gyroscope[:, 0], fs=self.fs, theta0=za[0], bias_offset=self.noise_model.gyro_rate_model.bias_hat)
        kf = run_attitude_kalman_2state(za, res.gyroscope[:, 0], self.noise_model, use_calibrated_gyro=False)

        # Pure gyro integration drifts by > 30 deg in 60 s
        self.assertGreater(abs(gyr_int[-1] - th[-1]) * RAD2DEG, 30.0)
        # Kalman converges to the true total gyro bias within 1.5e-3 rad/s
        self.assertAlmostEqual(float(kf.bias_est_radps[-1]), target_bias, delta=1.5e-3)
        # After bias convergence (burn_in=20s=4000 samples), angle RMSE is tiny (< 0.15 deg)
        m_kf = compute_filter_metrics(kf.theta_est_rad, th, za, fs=self.fs, burn_in=4000)
        self.assertLess(m_kf.rmse * RAD2DEG, 0.15)


if __name__ == "__main__":
    unittest.main()
