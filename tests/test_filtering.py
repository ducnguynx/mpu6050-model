"""Unit tests for black-box identification, baselines, Wiener, and Kalman filtering."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpu6050_sim import MPU6050, STANDARD_GRAVITY
from mpu6050_sim.filtering import (
    apply_frequency_wiener,
    compute_filter_metrics,
    compute_innovation_diagnostics,
    design_and_apply_fir_wiener,
    identify_noise_from_static_data,
    ljung_box_test,
    lowpass_first_order_filter,
    moving_average_filter,
    run_assumption_tests,
    tune_kalman_qu_mle,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class TestFilteringPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fs = 200.0
        sensor = MPU6050.from_config(CONFIG_PATH)
        sensor.update_parameters(simulation={"sample_rate_hz": cls.fs, "random_seed": 701})
        # 600 s static data for fast unit testing
        n_static = int(round(600.0 * cls.fs))
        acc_in = np.tile(np.array([0.0, 0.0, STANDARD_GRAVITY]), (n_static, 1))
        gyr_in = np.zeros((n_static, 3))
        res_static = sensor.simulate(acc_in, gyr_in)
        cls.y_static = res_static.accelerometer[:, 0]
        cls.fsr = sensor.config.accelerometer.full_scale_range_g * STANDARD_GRAVITY
        cls.noise_model = identify_noise_from_static_data(cls.y_static, fs=cls.fs)

        # 10 s multi-frequency dynamic test run with separate seed
        sensor.reset(seed=702)
        t = sensor.time_vector(duration_s=10.0)
        cls.ideal_ax = 0.5 * np.sin(2.0 * np.pi * 0.5 * t) + 0.25 * np.sin(2.0 * np.pi * 3.5 * t)
        acc_dyn = np.column_stack([cls.ideal_ax, np.zeros_like(t), np.full_like(t, STANDARD_GRAVITY)])
        res_dyn = sensor.simulate(acc_dyn, np.zeros((t.size, 3)))
        cls.meas_ax = res_dyn.accelerometer[:, 0]

    def test_assumption_checks_detect_bias_and_gaussianity(self) -> None:
        report = run_assumption_tests(self.y_static, self.meas_ax, fs=self.fs, fsr_limit=self.fsr)
        self.assertTrue(report.saturation_free)
        self.assertTrue(report.has_nonzero_bias)
        self.assertTrue(report.approx_gaussian)
        self.assertTrue(report.has_colored_noise)

    def test_blackbox_noise_identification_accuracy(self) -> None:
        # White noise sigma_v should match 0.03923 m/s^2 within 3%
        self.assertAlmostEqual(self.noise_model.sigma_v_hat, 0.03923, delta=0.0015)
        self.assertGreater(self.noise_model.R_hat, 0.0)
        self.assertGreater(self.noise_model.phi_b_hat, 0.99)
        self.assertLess(self.noise_model.phi_b_hat, 1.0)

    def test_ljung_box_passes_on_pure_white_noise(self) -> None:
        rng = np.random.default_rng(1234)
        white = rng.normal(0.0, 1.0, size=2000)
        _, p_val, _ = ljung_box_test(white, lags=20)
        self.assertGreater(p_val, 0.05)

    def test_wiener_and_kalman_beat_raw_and_baselines(self) -> None:
        y_cal = self.meas_ax - self.noise_model.bias_hat
        ma = moving_average_filter(y_cal, window=15)
        lpf = lowpass_first_order_filter(y_cal, fs=self.fs, cutoff_hz=5.0)
        wfir = design_and_apply_fir_wiener(y_cal, self.noise_model, order=31)
        wfreq, _, _ = apply_frequency_wiener(y_cal, self.noise_model)
        _, _, kf = tune_kalman_qu_mle(y_cal, self.noise_model, include_bias_state=True, signal_damping_tau_s=20.0)

        m_cal = compute_filter_metrics(y_cal, self.ideal_ax, self.meas_ax, fs=self.fs)
        m_ma = compute_filter_metrics(ma, self.ideal_ax, self.meas_ax, fs=self.fs)
        m_lpf = compute_filter_metrics(lpf, self.ideal_ax, self.meas_ax, fs=self.fs)
        m_wfir = compute_filter_metrics(wfir.estimate, self.ideal_ax, self.meas_ax, fs=self.fs)
        m_wfreq = compute_filter_metrics(wfreq, self.ideal_ax, self.meas_ax, fs=self.fs)
        m_kf = compute_filter_metrics(kf.signal_est, self.ideal_ax, self.meas_ax, fs=self.fs, est_std=kf.signal_std)

        # Both Wiener and Kalman strongly outperform raw & naive baselines
        self.assertLess(m_wfir.rmse, m_cal.rmse * 0.65)
        self.assertLess(m_kf.rmse, m_cal.rmse * 0.72)
        self.assertLess(m_kf.rmse, m_ma.rmse)
        self.assertLess(m_kf.rmse, m_lpf.rmse)
        self.assertLess(m_wfreq.rmse, m_cal.rmse * 0.45)

        # Kalman no-ground-truth consistency check
        diag = compute_innovation_diagnostics(kf, burn_in=40, lags=20)
        self.assertTrue(diag.nis_consistent)
        self.assertGreater(diag.ljung_box_p, 0.01)
        self.assertIsNotNone(m_kf.coverage_95_pct)
        self.assertGreater(m_kf.coverage_95_pct, 85.0)


if __name__ == "__main__":
    unittest.main()
