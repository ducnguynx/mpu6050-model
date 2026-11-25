"""Quantitative validation of the MPU-6050 Python simulator against published theory.

Validates:
    1. InvenSense PS-MPU-6000A-00 Rev 3.4 datasheet RMS noise check:
       At f_s = 200 Hz (Nyquist BW = 100 Hz, DLPF_CFG=2), gyroscope broadband RMS noise
       matches 0.05 deg/s-rms (8.72665e-4 rad/s) and accelerometer matches 4.00 mg-rms.
    2. Farrell et al. (2022) & El-Sheimy et al. (2008) analytical Allan deviation check:
       Long stationary simulations of the MPU-6050 accelerometer and gyroscope match
       the closed-form analytical Allan variance Eq. (22) across white noise (-1/2 slope),
       first-order Gauss-Markov bias instability (hump at tau = 1.89 * T_B), and
       rate/acceleration random walk (+1/2 slope).
    3. Pure Gauss-Markov analytical peak check (Farrell et al., 2022):
       Verifies empirical peak Allan deviation at tau = 1.89 * T_B equals
       0.43654 * sqrt(S_B * T_B) = 0.66428 * B.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpu6050_sim import (
    MPU6050,
    STANDARD_GRAVITY,
    overlapping_allan_variance,
    theoretical_allan_variance,
)
from mpu6050_sim.visualization import plot_allan_deviation

CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def validate_datasheet_rms_noise() -> dict[str, float]:
    """Check broadband discrete RMS noise against InvenSense PS-MPU-6000A-00 Rev 3.4."""
    sensor = MPU6050.from_config(CONFIG_PATH)
    # Isolate white noise only, disable quantization for exact analytical comparison
    sensor.update_parameters(
        simulation={"sample_rate_hz": 200.0, "random_seed": 123, "stationary_bias_init": False},
        accelerometer={
            "static_bias_mps2": [0.0, 0.0, 0.0],
            "bias_instability_mps2": [0.0, 0.0, 0.0],
            "random_walk_mps2_per_sqrts": [0.0, 0.0, 0.0],
            "enable_quantization": False,
        },
        gyroscope={
            "static_bias_radps": [0.0, 0.0, 0.0],
            "bias_instability_radps": [0.0, 0.0, 0.0],
            "random_walk_radps_per_sqrts": [0.0, 0.0, 0.0],
            "enable_quantization": False,
        },
    )
    n_samples = 200_000  # 1000 s at 200 Hz
    acc_in = np.zeros((n_samples, 3))
    gyr_in = np.zeros((n_samples, 3))
    res = sensor.simulate(acc_in, gyr_in)

    gyro_rms_dps = float(np.mean(np.std(res.gyroscope, axis=0)) * (180.0 / math.pi))
    accel_rms_mg = float(np.mean(np.std(res.accelerometer, axis=0)) / STANDARD_GRAVITY * 1e3)

    expected_gyro_dps = 0.05  # PS-MPU-6000A-00 §6.1 at 100 Hz BW
    expected_accel_mg = 4.00  # 400 ug/sqrt(Hz) * sqrt(100 Hz)

    print("=== 1. Datasheet Broadband RMS Noise Validation (fs = 200 Hz, BW = 100 Hz) ===")
    print(
        f"Gyroscope RMS noise:     empirical = {gyro_rms_dps:.5f} deg/s-rms | "
        f"datasheet = {expected_gyro_dps:.5f} deg/s-rms | "
        f"rel err = {abs(gyro_rms_dps - expected_gyro_dps) / expected_gyro_dps * 100:.3f}%"
    )
    print(
        f"Accelerometer RMS noise: empirical = {accel_rms_mg:.5f} mg-rms    | "
        f"datasheet = {expected_accel_mg:.5f} mg-rms    | "
        f"rel err = {abs(accel_rms_mg - expected_accel_mg) / expected_accel_mg * 100:.3f}%"
    )
    return {
        "gyro_rms_dps": gyro_rms_dps,
        "expected_gyro_dps": expected_gyro_dps,
        "accel_rms_mg": accel_rms_mg,
        "expected_accel_mg": expected_accel_mg,
    }


def validate_allan_deviation() -> dict[str, float]:
    """Compare simulated overlapping Allan deviation against Farrell (2022) Eq. (22)."""
    # Use 50 Hz over 3600 s (180,000 samples) so tau spans 0.02 s to ~1200 s
    # covering white noise (-1/2), Gauss-Markov bias instability hump (~189 s), and random walk (+1/2).
    sensor = MPU6050.from_config(CONFIG_PATH)
    fs = 50.0
    duration_s = 3600.0
    sensor.update_parameters(
        simulation={"sample_rate_hz": fs, "duration_s": duration_s, "random_seed": 2022},
    )
    n_samples = int(round(fs * duration_s))
    acc_in = np.tile(np.array([0.0, 0.0, STANDARD_GRAVITY]), (n_samples, 1))
    gyr_in = np.zeros((n_samples, 3))
    res = sensor.simulate(acc_in, gyr_in)

    # Accelerometer X-axis Allan deviation
    tau_a, avar_a = overlapping_allan_variance(res.accelerometer[:, 0], fs, max_clusters=70)
    adev_a_emp = np.sqrt(avar_a)
    acc_cfg = sensor.config.accelerometer
    theo_a = theoretical_allan_variance(
        tau_a,
        noise_density=acc_cfg.noise_density_mps2_per_sqrthz[0],
        bias_instability=acc_cfg.bias_instability_mps2[0],
        bias_correlation_time_s=acc_cfg.bias_correlation_time_s[0],
        random_walk=acc_cfg.random_walk_mps2_per_sqrts[0],
        quantization_step=acc_cfg.quantization_step_mps2 if acc_cfg.enable_quantization else 0.0,
        sample_rate_hz=fs,
    )
    adev_a_theo = np.sqrt(theo_a["total"])

    # Gyroscope X-axis Allan deviation
    tau_g, avar_g = overlapping_allan_variance(res.gyroscope[:, 0], fs, max_clusters=70)
    adev_g_emp = np.sqrt(avar_g)
    gyr_cfg = sensor.config.gyroscope
    theo_g = theoretical_allan_variance(
        tau_g,
        noise_density=gyr_cfg.noise_density_radps_per_sqrthz[0],
        bias_instability=gyr_cfg.bias_instability_radps[0],
        bias_correlation_time_s=gyr_cfg.bias_correlation_time_s[0],
        random_walk=gyr_cfg.random_walk_radps_per_sqrts[0],
        quantization_step=gyr_cfg.quantization_step_radps if gyr_cfg.enable_quantization else 0.0,
        sample_rate_hz=fs,
    )
    adev_g_theo = np.sqrt(theo_g["total"])

    # Evaluate accuracy across well-averaged regime (tau <= 100 s, >= 36 independent clusters)
    mask_a = tau_a <= 100.0
    mask_g = tau_g <= 100.0
    mape_a = float(np.mean(np.abs(adev_a_emp[mask_a] - adev_a_theo[mask_a]) / adev_a_theo[mask_a]) * 100.0)
    mape_g = float(np.mean(np.abs(adev_g_emp[mask_g] - adev_g_theo[mask_g]) / adev_g_theo[mask_g]) * 100.0)

    # Check ARW/VRW at tau = 1 s
    idx_1s_a = int(np.argmin(np.abs(tau_a - 1.0)))
    idx_1s_g = int(np.argmin(np.abs(tau_g - 1.0)))

    print("\n=== 2. Allan Deviation Validation vs. Farrell et al. (2022) & El-Sheimy (2008) ===")
    print(f"Duration: {duration_s:.0f} s at fs = {fs:.0f} Hz ({n_samples} samples)")
    print(
        f"Accelerometer X-axis: MAPE (tau<=100 s) = {mape_a:.2f}% | "
        f"ADEV(1s) emp = {adev_a_emp[idx_1s_a]:.6f} vs theo = {adev_a_theo[idx_1s_a]:.6f} m/s^2"
    )
    print(
        f"Gyroscope     X-axis: MAPE (tau<=100 s) = {mape_g:.2f}% | "
        f"ADEV(1s) emp = {adev_g_emp[idx_1s_g]:.6e} vs theo = {adev_g_theo[idx_1s_g]:.6e} rad/s"
    )

    plot_allan_deviation(
        tau_a,
        adev_a_emp,
        theo_a,
        unit="m/s²",
        title="Validation: MPU-6050 Accelerometer Allan Deviation (X-Axis)",
        config_note=f"T=3600 s, fs=50 Hz, N_a={acc_cfg.noise_density_mps2_per_sqrthz[0]:.5f}, B_a={acc_cfg.bias_instability_mps2[0]}, T_B={acc_cfg.bias_correlation_time_s[0]:.0f}s, K_a={acc_cfg.random_walk_mps2_per_sqrts[0]}",
        save_path=OUTPUT_DIR / "validation_accel_allan_deviation.png",
    )
    plot_allan_deviation(
        tau_g,
        adev_g_emp,
        theo_g,
        unit="rad/s",
        title="Validation: MPU-6050 Gyroscope Allan Deviation (X-Axis)",
        config_note=f"T=3600 s, fs=50 Hz, N_g={gyr_cfg.noise_density_radps_per_sqrthz[0]:.2e}, B_g={gyr_cfg.bias_instability_radps[0]:.1e}, T_B={gyr_cfg.bias_correlation_time_s[0]:.0f}s, K_g={gyr_cfg.random_walk_radps_per_sqrts[0]:.1e}",
        save_path=OUTPUT_DIR / "validation_gyro_allan_deviation.png",
    )

    return {
        "mape_accel_pct": mape_a,
        "mape_gyro_pct": mape_g,
        "adev_accel_1s_emp": float(adev_a_emp[idx_1s_a]),
        "adev_accel_1s_theo": float(adev_a_theo[idx_1s_a]),
        "adev_gyro_1s_emp": float(adev_g_emp[idx_1s_g]),
        "adev_gyro_1s_theo": float(adev_g_theo[idx_1s_g]),
    }


def validate_pure_gauss_markov_peak() -> dict[str, float]:
    """Verify Farrell et al. (2022) Gauss-Markov peak: sigma_A(1.89*T_B) = 0.66428 * B."""
    fs = 20.0
    t_b = 5.0
    b_target = 1.0e-3
    n_samples = 200_000  # 10,000 s = 2,000 correlation times for tight statistical convergence

    sensor = MPU6050.from_config(CONFIG_PATH)
    sensor.update_parameters(
        simulation={"sample_rate_hz": fs, "random_seed": 777, "stationary_bias_init": True},
        gyroscope={
            "static_bias_radps": [0.0, 0.0, 0.0],
            "noise_density_radps_per_sqrthz": [0.0, 0.0, 0.0],
            "bias_instability_radps": [b_target, b_target, b_target],
            "bias_correlation_time_s": [t_b, t_b, t_b],
            "random_walk_radps_per_sqrts": [0.0, 0.0, 0.0],
            "enable_quantization": False,
        },
    )
    res = sensor.simulate(np.zeros((n_samples, 3)), np.zeros((n_samples, 3)))

    # Average Allan variance across the 3 independent axes for lower estimation variance
    tau, av0 = overlapping_allan_variance(res.gyroscope[:, 0], fs, max_clusters=70)
    _, av1 = overlapping_allan_variance(res.gyroscope[:, 1], fs, max_clusters=70)
    _, av2 = overlapping_allan_variance(res.gyroscope[:, 2], fs, max_clusters=70)
    adev_emp = np.sqrt((av0 + av1 + av2) / 3.0)

    theo = theoretical_allan_variance(
        tau,
        noise_density=0.0,
        bias_instability=b_target,
        bias_correlation_time_s=t_b,
        random_walk=0.0,
    )
    adev_theo = np.sqrt(theo["total"])

    idx_peak_emp = int(np.argmax(adev_emp))
    tau_peak_emp = float(tau[idx_peak_emp])
    val_peak_emp = float(adev_emp[idx_peak_emp])
    tau_peak_theo = 1.89 * t_b
    val_peak_theo = math.sqrt(2.0 * math.log(2.0) / math.pi) * b_target

    print("\n=== 3. Farrell et al. (2022) First-Order Gauss-Markov Peak Validation ===")
    print(f"Expected peak location: tau = 1.89 * T_B = {tau_peak_theo:.2f} s | Empirical = {tau_peak_emp:.2f} s")
    print(
        f"Expected peak ADEV:     0.66428 * B = {val_peak_theo:.6e} | "
        f"Empirical = {val_peak_emp:.6e} | "
        f"rel err = {abs(val_peak_emp - val_peak_theo) / val_peak_theo * 100:.2f}%"
    )
    plot_allan_deviation(
        tau,
        adev_emp,
        theo,
        unit="rad/s",
        title="Validation: Pure First-Order Gauss-Markov Allan Deviation (Farrell et al., 2022)",
        config_note=f"T=10,000 s (3-axis mean), T_B={t_b:.1f} s, B={b_target:.1e} rad/s -> peak at tau=1.89*T_B={tau_peak_theo:.2f} s",
        save_path=OUTPUT_DIR / "validation_gauss_markov_peak.png",
    )
    return {
        "tau_peak_emp": tau_peak_emp,
        "tau_peak_theo": tau_peak_theo,
        "val_peak_emp": val_peak_emp,
        "val_peak_theo": val_peak_theo,
    }


if __name__ == "__main__":
    validate_datasheet_rms_noise()
    validate_allan_deviation()
    validate_pure_gauss_markov_peak()
    print(f"\nValidation plots written to {OUTPUT_DIR}")
