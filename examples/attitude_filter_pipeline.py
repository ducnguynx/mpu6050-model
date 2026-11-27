"""End-to-end black-box Roll-angle (tilt) estimation pipeline for MPU-6050 (Pair 2).

Compares Required Baselines vs. Pair 2 (Optimal Wiener Complementary Filter vs.
2-State Kalman Filter with AR(1) Gyro Bias) strictly treating the MPU-6050 model
as unknown (black-box):
    Step 1: Generate 1-hour static bench data (dual-sensor Accel + Gyro) and
            4 physically consistent Roll-angle dynamic scenarios.
    Step 2: Run assumption checks (saturation, zero-mean, Gaussianity, WSS,
            colored noise, small-angle linearization) and black-box stochastic
            identification with uncertainty for BOTH Accel-angle & Gyro-rate.
    Step 3: Design and evaluate:
            - Baseline 0a: Raw Accelerometer Angle atan2(a_y, a_z)
            - Baseline 0b: Raw Gyroscope Integration (shows unbounded drift)
            - Baseline 1 : Fixed-gain Complementary Filter (alpha = 0.96)
            - Method 1a  : Optimal Wiener Complementary Filter (Freq-Domain from
                           empirical static PSDs S_na(f) & S_nw(f))
            - Method 1b  : Causal 1st-Order Wiener Complementary Filter
                           (spectral-factorized omega_c* = sqrt(S_w / S_a))
            - Method 2   : Recursive 2-State Kalman Filter x = [theta, b_g]^T
                           with black-box AR(1) gyro bias model
    Step 4: Evaluate without ground truth (Innovation ACF, Ljung-Box, NIS) and
            with ground truth (RMSE [deg], MAE, Bias, lag, 95% coverage,
            Monte Carlo mean +/- std, vibration & uncalibrated-bias stress tests).

Outputs 6 figures and results_summary.json to outputs/attitude/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpu6050_sim import MPU6050, STANDARD_GRAVITY
from mpu6050_sim.attitude_filtering import (
    AttitudeNoiseModel,
    accel_to_roll_atan2,
    accel_to_roll_linearized,
    apply_wiener_complementary_causal,
    apply_wiener_complementary_freq,
    baseline_complementary_filter,
    baseline_gyro_integration,
    check_linearization_accuracy,
    compute_attitude_innovation_diagnostics,
    identify_attitude_noise_from_static,
    run_attitude_kalman_4state,
    tune_attitude_kalman_4state_qu_mle,
)
from mpu6050_sim.filtering import compute_filter_metrics, norm_ppf

CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "attitude"
RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0


# ==============================================================================
# Physically Consistent Roll-Angle Simulator Wrapper
# ==============================================================================


def simulate_roll_motion(
    theta_rad: np.ndarray,
    omega_x_radps: np.ndarray,
    *,
    seed: int,
    fs: float = 200.0,
    a_lin_y_mps2: np.ndarray | None = None,
    gyro_overrides: dict | None = None,
    accel_overrides: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MPU6050]:
    """Simulate physically consistent MPU-6050 6-axis measurements for Roll motion around X.

    Physics:
        omega_ideal = [d(theta)/dt, 0, 0]
        f_ideal     = [0, g0 * sin(theta) + a_lin_y, g0 * cos(theta)]
    Returns (time_s, ay_meas, az_meas, gx_meas, sensor).
    """
    sensor = MPU6050.from_config(CONFIG_PATH)
    sensor.update_parameters(
        simulation={"sample_rate_hz": fs, "random_seed": seed},
        accelerometer=accel_overrides,
        gyroscope=gyro_overrides,
    )
    n = theta_rad.size
    a_lin = np.zeros(n, dtype=float) if a_lin_y_mps2 is None else np.asarray(a_lin_y_mps2, dtype=float)
    ideal_acc = np.column_stack(
        [
            np.zeros(n, dtype=float),
            STANDARD_GRAVITY * np.sin(theta_rad) + a_lin,
            STANDARD_GRAVITY * np.cos(theta_rad),
        ]
    )
    ideal_gyr = np.column_stack(
        [
            omega_x_radps,
            np.zeros(n, dtype=float),
            np.zeros(n, dtype=float),
        ]
    )
    res = sensor.simulate(ideal_acc, ideal_gyr)
    return res.time_s, res.accelerometer[:, 1], res.accelerometer[:, 2], res.gyroscope[:, 0], sensor


def build_roll_scenarios(fs: float = 200.0) -> dict[str, dict[str, np.ndarray]]:
    """Define 4 physically consistent Roll-angle test scenarios."""
    t30 = np.arange(int(round(30.0 * fs)), dtype=float) / fs
    t20 = np.arange(int(round(20.0 * fs)), dtype=float) / fs
    t180 = np.arange(int(round(180.0 * fs)), dtype=float) / fs

    # S1: Static level (30 s)
    s1_th = np.zeros_like(t30)
    s1_om = np.zeros_like(t30)

    # S2: Smooth small-angle oscillation (12 deg at 0.3 Hz, within |theta| <= 15 deg linear regime)
    a2 = 12.0 * DEG2RAD
    w2 = 2.0 * math.pi * 0.3
    s2_th = a2 * np.sin(w2 * t20)
    s2_om = a2 * w2 * np.cos(w2 * t20)

    # S3: Multi-frequency tilt (10 deg @ 0.25 Hz + 4 deg @ 1.2 Hz) + 4 Hz lateral vibration (1.5 m/s^2)
    a3_1, w3_1 = 10.0 * DEG2RAD, 2.0 * math.pi * 0.25
    a3_2, w3_2 = 4.0 * DEG2RAD, 2.0 * math.pi * 1.2
    s3_th = a3_1 * np.sin(w3_1 * t20) + a3_2 * np.sin(w3_2 * t20)
    s3_om = a3_1 * w3_1 * np.cos(w3_1 * t20) + a3_2 * w3_2 * np.cos(w3_2 * t20)
    s3_vib = 1.5 * np.sin(2.0 * math.pi * 4.0 * t20)

    # S4: Long-term 3-minute tilt (10 deg @ 0.15 Hz) with realistic uncalibrated thermal bias step (+0.008 rad/s)
    a4, w4 = 10.0 * DEG2RAD, 2.0 * math.pi * 0.15
    s4_th = a4 * np.sin(w4 * t180)
    s4_om = a4 * w4 * np.cos(w4 * t180)

    return {
        "S1_Static_30s": {"t": t30, "theta": s1_th, "omega": s1_om, "a_lin": None, "extra_gyro_bias": 0.0},
        "S2_SmallAngle_Sin_20s": {"t": t20, "theta": s2_th, "omega": s2_om, "a_lin": None, "extra_gyro_bias": 0.0},
        "S3_MultiFreq_Vib_20s": {"t": t20, "theta": s3_th, "omega": s3_om, "a_lin": s3_vib, "extra_gyro_bias": 0.0},
        "S4_LongTerm_Drift_180s": {"t": t180, "theta": s4_th, "omega": s4_om, "a_lin": None, "extra_gyro_bias": 0.008},
    }


def run_all_attitude_methods(
    ay: np.ndarray,
    az: np.ndarray,
    gx: np.ndarray,
    noise_model: AttitudeNoiseModel,
    *,
    cf_alpha: float = 0.96,
    fixed_qu: float | None = None,
) -> dict[str, dict]:
    """Run all Baselines + Method 1 (Wiener) + Method 2 (Kalman 4-State [u, u', u'', b]^T)."""
    fs = noise_model.sample_rate_hz
    n = ay.size
    gm = noise_model.gyro_rate_model

    # Measured Roll angle from 2-axis flat-bench calibrated accelerometer (removes both b_0y & b_0z!)
    theta_acc_cal = accel_to_roll_atan2(
        ay - noise_model.ay_bias_hat_mps2,
        az - noise_model.az_bias_hat_mps2,
    )
    gx_cal = gx - gm.bias_hat

    # Baseline 0a: Raw calibrated Accel angle
    b_acc = theta_acc_cal.copy()

    # Baseline 0b: Pure Gyro integration (using static-calibrated gyro -> still drifts from residual/AR1/RW!)
    t0 = time.perf_counter()
    b_gyr = baseline_gyro_integration(gx, fs=fs, theta0=theta_acc_cal[0], bias_offset=gm.bias_hat)
    t_gyr = time.perf_counter() - t0

    # Baseline 1: Fixed-gain Complementary Filter (alpha = 0.96, typical textbook/Arduino baseline)
    t0 = time.perf_counter()
    b_cf = baseline_complementary_filter(theta_acc_cal, gx_cal, fs=fs, alpha=cf_alpha)
    t_cf = time.perf_counter() - t0

    # Method 1a: Optimal Wiener Complementary Filter (Frequency-Domain directly from static Welch PSDs)
    t0 = time.perf_counter()
    w_freq = apply_wiener_complementary_freq(theta_acc_cal, gx_cal, noise_model, use_parametric_psd=False)
    t_wf = time.perf_counter() - t0

    # Method 1b: Causal 1st-Order Wiener Complementary Filter (Spectral Factorization omega_c*)
    t0 = time.perf_counter()
    w_causal_est, alpha_opt = apply_wiener_complementary_causal(theta_acc_cal, gx_cal, noise_model)
    t_wc = time.perf_counter() - t0

    # Method 2: Recursive 4-State Kalman Filter x_k = [u_k, u_dot_k, u_ddot_k, b_k]^T
    # Fuses 2D measurement y_k = [theta_acc_cal[k], gx[k]]^T; q_u tuned via 2D Innovation MLE (zero ground truth!)
    if fixed_qu is None:
        q_u_mle, log10_qu_se, kf_res = tune_attitude_kalman_4state_qu_mle(theta_acc_cal, gx, noise_model)
    else:
        q_u_mle = fixed_qu
        log10_qu_se = float("nan")
        kf_res = run_attitude_kalman_4state(theta_acc_cal, gx, noise_model, q_u=q_u_mle)

    # Measure single-pass online Kalman runtime
    t0 = time.perf_counter()
    _ = run_attitude_kalman_4state(theta_acc_cal, gx, noise_model, q_u=q_u_mle)
    t_kf = time.perf_counter() - t0

    return {
        "Baseline_AccelRaw": {"est": b_acc, "time_ms_per_sample": 0.0},
        "Baseline_GyroInt": {"est": b_gyr, "time_ms_per_sample": t_gyr * 1e3 / n},
        "Baseline_CF_Fixed": {"est": b_cf, "alpha": cf_alpha, "time_ms_per_sample": t_cf * 1e3 / n},
        "Method1_Wiener_Freq": {
            "est": w_freq.theta_est_rad,
            "result": w_freq,
            "time_ms_per_sample": t_wf * 1e3 / n,
        },
        "Method1_Wiener_Causal": {
            "est": w_causal_est,
            "alpha_opt": alpha_opt,
            "time_ms_per_sample": t_wc * 1e3 / n,
        },
        "Method2_Kalman_4State": {
            "est": kf_res.theta_est_rad,
            "std": kf_res.theta_std_rad,
            "result": kf_res,
            "q_u_mle": q_u_mle,
            "log10_qu_se": log10_qu_se,
            "time_ms_per_sample": t_kf * 1e3 / n,
        },
    }


# ==============================================================================
# Publication-Grade Figures
# ==============================================================================


def plot_fig1_dual_assumptions(
    noise_model: AttitudeNoiseModel,
    y_theta_static: np.ndarray,
    gx_static: np.ndarray,
    save_path: Path,
) -> None:
    """Figure 1: Static PSDs, Gyro Q-Q Gaussianity, and Gyro 1 s ACF (Colored Bias Check)."""
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    am = noise_model.accel_angle_model
    gm = noise_model.gyro_rate_model
    ga = noise_model.gyro_rate_assumptions

    # (a) Static PSDs of Accel Roll-angle vs Integrated Gyro Roll-angle -> crossing point = Wiener crossover!
    ax = axes[0, 0]
    f_pos = noise_model.wiener_freqs_hz[1:]
    dt = 1.0 / noise_model.sample_rate_hz
    d_sq = (4.0 * np.sin(math.pi * f_pos * dt) ** 2) / (dt * dt)
    s_na = am.psd_noise[1:]
    s_ng = np.interp(f_pos, gm.psd_freqs[1:], gm.psd_noise[1:]) / d_sq
    ax.loglog(f_pos, s_na, color="#ff7f0e", linewidth=1.2, alpha=0.85, label=r"Accel Angle PSD $\hat{S}_{n_a}(f)$ [rad²/Hz]")
    ax.loglog(f_pos, s_ng, color="#1f77b4", linewidth=1.3, alpha=0.9, label=r"Integrated Gyro PSD $\hat{S}_{n_g}(f) = \hat{S}_{n_\omega}(f)/|D(f)|^2$")
    ax.axvline(
        noise_model.wiener_causal_fc_hz,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label=f"Optimal Crossover $f_c^* = {noise_model.wiener_causal_fc_hz:.4f}$ Hz",
    )
    ax.set_title("(a) Static Noise PSDs: Accel Angle vs. Integrated Gyro Angle", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Single-Sided Angle PSD [rad²/Hz]")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

    # (b) Optimal Wiener Complementary Transfer Functions G_W(f) and 1 - G_W(f)
    ax = axes[0, 1]
    ax.semilogx(f_pos, noise_model.wiener_G_acc[1:], color="#ff7f0e", linewidth=1.8, label=r"Accel Path $G_W(f)$ (Low-Pass, $G_W(0)=1$)")
    ax.semilogx(f_pos, 1.0 - noise_model.wiener_G_acc[1:], color="#1f77b4", linewidth=1.8, label=r"Gyro Path $1 - G_W(f)$ (High-Pass)")
    ax.axvline(noise_model.wiener_causal_fc_hz, color="#d62728", linestyle="--", linewidth=1.2, label=f"$f_c^* = {noise_model.wiener_causal_fc_hz:.4f}$ Hz")
    ax.set_title("(b) Empirical Optimal Wiener Complementary Filter Gain $G_W(f)$", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Filter Weight (Dimensionless)")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="center right")

    # (c) Gyro Q-Q Plot of first differences (Gaussianity check)
    ax = axes[1, 0]
    dg = np.diff(gx_static)
    dg_z = (dg - np.mean(dg)) / np.std(dg, ddof=1)
    probs = (np.arange(1, 501) - 0.5) / 500.0
    ax.scatter(norm_ppf(probs), np.quantile(dg_z, probs), s=8, alpha=0.7, color="#2ca02c", label=r"Gyro $\Delta \omega_{x,k}$ quantiles")
    ax.plot([-3.5, 3.5], [-3.5, 3.5], "k--", linewidth=1.3, label="Ideal Standard Normal")
    ax.set_title(
        f"(c) Gyro Gaussianity Q-Q (Skew={ga.skewness:+.4f}, ExKurt={ga.excess_kurtosis:+.4f}, JB p={ga.jarque_bera_p_value:.3f})",
        fontsize=9.3,
        fontweight="bold",
    )
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Standardized Sample Quantiles")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="upper left")

    # (d) Gyro 1-second block ACF (Ljung-Box colored noise & drift proof)
    ax = axes[1, 1]
    lags = np.arange(1, ga.block_1s_acf.size)
    n_sec = gx_static.size // int(round(noise_model.sample_rate_hz))
    bound = 1.96 / math.sqrt(n_sec)
    ax.stem(lags, ga.block_1s_acf[1:], linefmt="#9467bd", markerfmt="o", basefmt="k-")
    ax.axhline(bound, color="#111111", linestyle="--", linewidth=1.1, label=f"95% White-Noise Bound (±{bound:.3f})")
    ax.axhline(-bound, color="#111111", linestyle="--", linewidth=1.1)
    ax.set_title(
        f"(d) Gyro 1 s Block ACF (Ljung-Box Q={ga.block_1s_ljung_box_q:.1f}, p={ga.block_1s_ljung_box_p:.2e})",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.set_xlabel("Lag [s]")
    ax.set_ylabel(r"Autocorrelation $\hat{\rho}[\tau]$")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="upper right")

    fig.suptitle(
        "Step 1-2: Dual-Sensor Black-Box PSD Characterization, Wiener Crossover & Assumption Checks",
        fontsize=11.4,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig2_gyro_allan(noise_model: AttitudeNoiseModel, save_path: Path) -> None:
    """Figure 2: Gyroscope X-axis Allan deviation black-box identification for Kalman AR(1) bias."""
    gm = noise_model.gyro_rate_model
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    tau = gm.tau
    ax.loglog(tau, gm.adev_emp, color="#1f77b4", linewidth=2.0, label=r"Empirical Overlapping ADEV $\hat{\sigma}_A(\tau)$")
    ax.fill_between(
        tau,
        np.maximum(gm.adev_emp - 1.96 * gm.adev_se, 1e-9),
        gm.adev_emp + 1.96 * gm.adev_se,
        color="#1f77b4",
        alpha=0.2,
        label=r"95% Confidence Band ($\pm 1.96\,\mathrm{SE}$)",
    )
    ax.loglog(tau, gm.adev_fit, "k--", linewidth=1.8, label="Black-Box Weighted LM Fit (Total)")
    comps = gm.adev_fit_components
    ax.loglog(tau, np.sqrt(comps["white_noise"]), color="#2ca02c", linestyle=":", linewidth=1.4, label=f"White Noise (ARW): $\\hat{{N}}_g = {gm.N_av_hat:.2e} \\pm {gm.N_av_se:.1e}$")
    ax.loglog(tau, np.sqrt(comps["gauss_markov"]), color="#ff7f0e", linestyle="-.", linewidth=1.4, label=f"AR(1) GM Bias: $\\hat{{B}}_g = {gm.B_av_hat:.2e}$, $\\hat{{T}}_B = {gm.TB_av_hat:.1f}$ s")
    ax.loglog(tau, np.sqrt(comps["random_walk"]), color="#d62728", linestyle=":", linewidth=1.4, label=f"Bias Drift (RRW): $\\hat{{K}}_g = {gm.K_av_hat:.2e}$")
    ax.set_xlabel(r"Averaging Cluster Time $\tau$ [s]")
    ax.set_ylabel(r"Gyroscope X-Axis Allan Deviation $\sigma_A(\tau)$ [rad/s]")
    ax.set_title("Step 2: Black-Box Gyroscope Stochastic Identification via Allan Deviation (for Kalman AR(1) Model)", fontsize=10.4, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.3, loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig3_vibration_comparison(
    t: np.ndarray,
    theta_true: np.ndarray,
    outputs: dict[str, dict],
    save_path: Path,
) -> None:
    """Figure 3: Scenario 3 (Multi-frequency Roll + 4 Hz Lateral Vibration) comparison."""
    mask = (t >= 4.0) & (t <= 10.0)
    tz = t[mask]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.2, 7.2), sharex=True)

    ax1.plot(tz, outputs["Baseline_AccelRaw"]["est"][mask] * RAD2DEG, color="#cccccc", linewidth=0.9, label="Baseline: Raw Accel atan2 (corrupted by vibration!)")
    ax1.plot(tz, outputs["Baseline_CF_Fixed"]["est"][mask] * RAD2DEG, color="#ff7f0e", linewidth=1.3, alpha=0.9, label=r"Baseline: Fixed CF ($\alpha=0.96$)")
    ax1.plot(tz, outputs["Method1_Wiener_Freq"]["est"][mask] * RAD2DEG, color="#2ca02c", linewidth=1.7, label="Method 1: Optimal Wiener Complementary (Freq)")
    ax1.plot(tz, outputs["Method2_Kalman_4State"]["est"][mask] * RAD2DEG, color="#1f77b4", linewidth=1.7, label=r"Method 2: 4-State Kalman $[u, \dot{u}, \ddot{u}, b]^T$")
    ax1.plot(tz, theta_true[mask] * RAD2DEG, "k--", linewidth=1.4, label="Ground Truth Roll $\\theta(t)$")
    ax1.set_ylabel("Roll Angle [deg]")
    ax1.set_title("Scenario 3: Multi-Frequency Roll + 4 Hz Lateral Acceleration Disturbance (1.5 m/s²) — Zoom 4–10 s", fontsize=10.4, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(ncol=2, fontsize=8.3, loc="upper right")

    ax2.plot(tz, (outputs["Baseline_CF_Fixed"]["est"][mask] - theta_true[mask]) * RAD2DEG, color="#ff7f0e", linewidth=1.2, alpha=0.85, label=r"Error: Fixed CF ($\alpha=0.96$)")
    ax2.plot(tz, (outputs["Method1_Wiener_Causal"]["est"][mask] - theta_true[mask]) * RAD2DEG, color="#17becf", linewidth=1.2, alpha=0.85, label="Error: Causal Wiener CF")
    ax2.plot(tz, (outputs["Method1_Wiener_Freq"]["est"][mask] - theta_true[mask]) * RAD2DEG, color="#2ca02c", linewidth=1.5, label="Error: Optimal Wiener CF (Freq)")
    ax2.plot(tz, (outputs["Method2_Kalman_4State"]["est"][mask] - theta_true[mask]) * RAD2DEG, color="#1f77b4", linewidth=1.6, label=r"Error: 4-State Kalman $[u, \dot{u}, \ddot{u}, b]^T$")
    ax2.axhline(0.0, color="#111111", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Roll Error [deg]")
    ax2.set_title("Angle Estimation Error ($\\hat{\\theta}_k - \\theta_k$): Both Wiener & Kalman Reject Lateral Vibration Dramatically Better than Baselines", fontsize=9.8, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(ncol=2, fontsize=8.3, loc="upper right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig4_drift_and_bias_tracking(
    t: np.ndarray,
    theta_true: np.ndarray,
    outputs: dict[str, dict],
    true_total_gyro_bias_radps: float,
    save_path: Path,
) -> None:
    """Figure 4: Scenario 4 (180 s Long-Term with +0.008 rad/s Thermal Gyro Bias Shift)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.2, 7.2), sharex=True)

    # (a) Angle error over 180 s (note: raw gyro drifts to ~80 deg, annotate instead of blowing up y-axis)
    err_gyr_final = abs((outputs["Baseline_GyroInt"]["est"][-1] - theta_true[-1]) * RAD2DEG)
    ax1.plot(t, (outputs["Baseline_CF_Fixed"]["est"] - theta_true) * RAD2DEG, color="#ff7f0e", linewidth=1.1, alpha=0.85, label=r"Baseline: Fixed CF ($\alpha=0.96$)")
    ax1.plot(t, (outputs["Method1_Wiener_Causal"]["est"] - theta_true) * RAD2DEG, color="#17becf", linewidth=1.2, alpha=0.85, label="Method 1b: Causal 1st-Order Wiener CF (biased by uncalibrated $b_g$!)")
    ax1.plot(t, (outputs["Method1_Wiener_Freq"]["est"] - theta_true) * RAD2DEG, color="#2ca02c", linewidth=1.5, label=r"Method 1a: Freq-Domain Optimal Wiener CF ($H_{\mathrm{gyr}}(0)=0$ rejects DC bias!)")
    ax1.plot(t, (outputs["Method2_Kalman_4State"]["est"] - theta_true) * RAD2DEG, color="#1f77b4", linewidth=1.5, label=r"Method 2: 4-State Kalman $[u, \dot{u}, \ddot{u}, b]^T$ (online bias tracking $\to$ zero steady-state error!)")
    ax1.axhline(0.0, color="#111111", linestyle="--", linewidth=1.0)
    ax1.text(
        0.015,
        0.08,
        f"Note: Pure Gyro Integration drifts to {err_gyr_final:.1f}° after 180 s (off-scale)",
        transform=ax1.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round", "facecolor": "#fff3cd", "edgecolor": "#cc9a06"},
    )
    ax1.set_ylabel("Roll Angle Error [deg]")
    ax1.set_title("Scenario 4 (180 s Run with +0.008 rad/s Uncalibrated Thermal Gyro Bias): Long-Term Bias Rejection", fontsize=10.3, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8.2, loc="upper right")

    # (b) Kalman real-time gyro bias estimate b_k_hat +/- 1.96*sigma vs. true gyro bias!
    kf_res = outputs["Method2_Kalman_4State"]["result"]
    b_est = kf_res.bias_est_radps * RAD2DEG
    b_sd = kf_res.bias_std_radps * RAD2DEG
    ax2.plot(t, b_est, color="#1f77b4", linewidth=1.6, label=r"4-State Kalman Online Bias Estimate $\hat{b}_k$ [deg/s]")
    ax2.fill_between(t, b_est - 1.96 * b_sd, b_est + 1.96 * b_sd, color="#1f77b4", alpha=0.22, label=r"95% Posterior Band ($\pm 1.96\sqrt{P_{bb}}$)")
    ax2.axhline(true_total_gyro_bias_radps * RAD2DEG, color="#d62728", linestyle="--", linewidth=1.5, label=f"True Total Gyro Static + Thermal Bias ({true_total_gyro_bias_radps*RAD2DEG:.3f} deg/s)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Gyroscope Bias [deg/s]")
    ax2.set_title(r"Observability Proof: 4-State Kalman $[u, \dot{u}, \ddot{u}, b]^T$ Converges to Unknown Gyro Bias Online", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8.4, loc="lower right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig5_kalman_consistency(
    t: np.ndarray,
    theta_true: np.ndarray,
    kf_res,
    diag,
    save_path: Path,
) -> None:
    """Figure 5: Kalman 95% posterior coverage + No-ground-truth Innovation ACF & NIS (Scenario 2)."""
    burn = 100
    tb = t[burn:]
    err_deg = (kf_res.theta_est_rad[burn:] - theta_true[burn:]) * RAD2DEG
    sd_deg = kf_res.theta_std_rad[burn:] * RAD2DEG
    cov95 = float(np.mean(np.abs(err_deg) <= 1.96 * sd_deg) * 100.0)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8))

    ax = axes[0, 0]
    ax.plot(tb, err_deg, color="#1f77b4", linewidth=0.9, label=r"Kalman Angle Error $\hat{\theta}_k - \theta_k$")
    ax.fill_between(tb, -1.96 * sd_deg, 1.96 * sd_deg, color="#ff7f0e", alpha=0.25, label=r"95% Posterior Band ($\pm 1.96\sqrt{P_{\theta\theta}}$)")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.9)
    ax.set_title(f"(a) Roll Error vs. 95% Uncertainty Band (Coverage = {cov95:.1f}%)", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Roll Error [deg]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="upper right")

    ax = axes[0, 1]
    z_nu = kf_res.innovations_rad[burn:] / np.sqrt(kf_res.innovation_var_angle[burn:])
    ax.plot(tb, z_nu, color="#2ca02c", linewidth=0.6, alpha=0.85, label=r"Normalized Angle Innovation $\nu_{a,k} / \sqrt{S_{aa,k}}$")
    ax.axhline(1.96, color="#d62728", linestyle="--", linewidth=1.1, label="±1.96 (95% Bounds)")
    ax.axhline(-1.96, color="#d62728", linestyle="--", linewidth=1.1)
    ax.set_title("(b) Normalized Innovations (No Ground Truth Needed)", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\nu_{a,k} / \sqrt{S_{aa,k}}$")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="upper right")

    ax = axes[1, 0]
    lags = np.arange(1, diag.acf.size)
    bound = 1.96 / math.sqrt(kf_res.innovations_rad[burn:].size)
    ax.stem(lags, diag.acf[1:], linefmt="#1f77b4", markerfmt="o", basefmt="k-")
    ax.axhline(bound, color="#d62728", linestyle="--", linewidth=1.1, label=f"95% White Bound (±{bound:.3f})")
    ax.axhline(-bound, color="#d62728", linestyle="--", linewidth=1.1)
    ax.set_title(f"(c) Innovation ACF Whiteness (Ljung-Box Q={diag.ljung_box_q:.1f}, p={diag.ljung_box_p:.3f})", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Lag [samples]")
    ax.set_ylabel(r"ACF $\hat{\rho}_\nu[\tau]$")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="upper right")

    ax = axes[1, 1]
    nis_seq = kf_res.nis_dof[burn:]
    running_nis = np.cumsum(nis_seq) / np.arange(1, nis_seq.size + 1)
    ax.plot(tb, running_nis, color="#9467bd", linewidth=1.4, label=f"Running Mean NIS/2 (Final = {diag.nis_mean:.3f})")
    ax.axhline(1.0, color="k", linestyle="-", linewidth=1.0, label="Ideal E[NIS/2] = 1.0")
    ax.axhspan(diag.nis_ci95[0], diag.nis_ci95[1], color="#2ca02c", alpha=0.2, label=f"95% $\\chi^2$ Band [{diag.nis_ci95[0]:.3f}, {diag.nis_ci95[1]:.3f}]")
    ax.set_title("(d) 2D Normalized Innovation Squared (NIS / 2) Consistency", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Running Mean NIS / 2")
    ax.set_ylim(0.5, 1.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="upper right")

    fig.suptitle(r"Step 3-4: 4-State Kalman $[u, \dot{u}, \ddot{u}, b]^T$ Uncertainty & No-Ground-Truth Diagnostics (Scenario 2)", fontsize=11.0, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig6_montecarlo_and_stress(
    mc_summary: dict,
    vib_sweep: dict,
    bias_sweep: dict,
    save_path: Path,
) -> None:
    """Figure 6: Monte Carlo RMSE bar chart + Vibration & Uncalibrated Gyro Bias Stress Tests."""
    fig = plt.figure(figsize=(13.2, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    # Top spanning panel: Monte Carlo RMSE [deg] (log scale so AccelRaw 6 deg & Wiener/Kalman 0.05 deg are all crisp!)
    ax_mc = fig.add_subplot(gs[0, :])
    scenarios = list(mc_summary.keys())
    methods = [
        ("Baseline_AccelRaw", "Baseline: Raw Accel atan2", "#999999"),
        ("Baseline_CF_Fixed", r"Baseline: Fixed CF ($\alpha=0.96$)", "#ff7f0e"),
        ("Method1_Wiener_Causal", "Method 1b: Causal Wiener CF", "#17becf"),
        ("Method1_Wiener_Freq", "Method 1a: Freq Wiener CF", "#2ca02c"),
        ("Method2_Kalman_4State", r"Method 2: 4-State Kalman $[u, \dot{u}, \ddot{u}, b]^T$", "#1f77b4"),
    ]
    x = np.arange(len(scenarios))
    width = 0.15
    for idx, (m_key, m_label, color) in enumerate(methods):
        means = [mc_summary[s][m_key]["rmse_deg_mean"] for s in scenarios]
        stds = [mc_summary[s][m_key]["rmse_deg_std"] for s in scenarios]
        offset = (idx - (len(methods) - 1) / 2.0) * width
        bars = ax_mc.bar(x + offset, means, width=width * 0.92, yerr=stds, capsize=2.5, color=color, alpha=0.92, label=m_label)
        for b, val, sd in zip(bars, means, stds):
            ax_mc.text(
                b.get_x() + b.get_width() / 2.0,
                (val + sd) * 1.12,
                f"{val:.2f}°",
                ha="center",
                va="bottom",
                fontsize=7.4,
                rotation=90,
            )
    ax_mc.set_yscale("log")
    ax_mc.set_ylim(0.01, 60.0)
    ax_mc.set_xticks(x)
    ax_mc.set_xticklabels(
        [
            "S1: Static (30 s)",
            "S2: Small-Angle Sin (20 s)",
            "S3: Multi-Freq + 4 Hz Vib",
            "S4: Long-Term + Thermal Bias (180 s)",
        ],
        fontsize=9.5,
    )
    ax_mc.set_ylabel("Roll Angle RMSE [deg] (log scale, lower is better)")
    ax_mc.set_title("(a) Monte Carlo Roll Angle Benchmark (Mean ± 1 SD RMSE across Independent Seeds)", fontsize=10.5, fontweight="bold")
    ax_mc.grid(True, which="both", axis="y", linestyle=":", alpha=0.6)
    ax_mc.legend(ncol=3, fontsize=8.5, loc="upper left")

    styles = [
        ("Baseline_AccelRaw", "Raw Accel atan2", "#888888", "--", "s"),
        ("Baseline_CF_Fixed", r"Fixed CF ($\alpha=0.96$)", "#ff7f0e", "-.", "^"),
        ("Method1_Wiener_Causal", "Causal Wiener CF", "#17becf", "-.", "v"),
        ("Method1_Wiener_Freq", "Freq Wiener CF", "#2ca02c", "-", "o"),
        ("Method2_Kalman_4State", r"4-State Kalman $[u,\dot{u},\ddot{u},b]^T$", "#1f77b4", "-", "D"),
    ]

    # Bottom-left: Lateral Vibration Amplitude Stress Test
    ax_v = fig.add_subplot(gs[1, 0])
    for key, label, color, ls, mk in styles:
        ax_v.plot(vib_sweep["amps"], vib_sweep[key], color=color, linestyle=ls, marker=mk, linewidth=1.6, label=label)
    ax_v.set_title("(b) Stress Test 1: Lateral Vibration Amplitude (4 Hz)", fontsize=10, fontweight="bold")
    ax_v.set_xlabel("Lateral Linear Acceleration Amplitude [m/s²]")
    ax_v.set_ylabel("Roll Angle RMSE [deg]")
    ax_v.grid(True, linestyle=":", alpha=0.6)
    ax_v.legend(fontsize=8.0, loc="upper left")

    # Bottom-right: Uncalibrated Gyro Bias Step Stress Test (120 s run)
    ax_b = fig.add_subplot(gs[1, 1])
    for key, label, color, ls, mk in styles:
        ax_b.plot(bias_sweep["bias_dps"], bias_sweep[key], color=color, linestyle=ls, marker=mk, linewidth=1.6, label=label)
    ax_b.set_title("(c) Stress Test 2: Uncalibrated Gyro Bias Step (120 s Run)", fontsize=10, fontweight="bold")
    ax_b.set_xlabel("Uncalibrated Gyroscope Bias Offset [deg/s]")
    ax_b.set_ylabel("Roll Angle RMSE [deg]")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(fontsize=8.0, loc="upper left")

    fig.suptitle("Step 4: Monte Carlo Benchmark & Robustness Stress Tests (MPU-6050 Roll Estimation)", fontsize=11.6, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ==============================================================================
# Main Execution
# ==============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fs = 200.0
    print("=" * 80)
    print("MPU-6050 BLACK-BOX ROLL-ANGLE ESTIMATION PIPELINE (PAIR 2: WIENER vs. KALMAN)")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # STEP 1: Dual-Sensor 1-Hour Static Data Acquisition (seed=1001)
    # --------------------------------------------------------------------------
    print("\n[Step 1] Generating 1-hour static bench dataset (seed=1001, fs=200 Hz)...")
    n_static = int(round(3600.0 * fs))
    _, ay_st, az_st, gx_st, st_sensor = simulate_roll_motion(
        np.zeros(n_static), np.zeros(n_static), seed=1001, fs=fs
    )
    scenarios = build_roll_scenarios(fs=fs)
    s2 = scenarios["S2_SmallAngle_Sin_20s"]
    _, ay_s2_ref, az_s2_ref, gx_s2_ref, _ = simulate_roll_motion(
        s2["theta"], s2["omega"], seed=2026, fs=fs
    )

    # --------------------------------------------------------------------------
    # STEP 2: Dual-Sensor Assumption Checks & Black-Box Stochastic Identification
    # --------------------------------------------------------------------------
    print("[Step 2] Running dual-sensor assumption checks & black-box noise identification...")
    acc_fsr = st_sensor.config.accelerometer.full_scale_range_g * STANDARD_GRAVITY
    gyr_fsr = st_sensor.config.gyroscope.full_scale_range_dps * DEG2RAD
    noise_model = identify_attitude_noise_from_static(
        ay_st,
        az_st,
        gx_st,
        ay_s2_ref,
        az_s2_ref,
        gx_s2_ref,
        fs=fs,
        accel_fsr_mps2=acc_fsr,
        gyro_fsr_radps=gyr_fsr,
        calibration_window_s=10.0,
        welch_nperseg=8192,
    )
    lin_s2 = check_linearization_accuracy(s2["theta"])
    # Also compare actual empirical RMSE of linearized a_y_cal/g0 vs atan2(a_y_cal, a_z_cal) on S2
    th_lin_s2 = accel_to_roll_linearized(ay_s2_ref - noise_model.ay_bias_hat_mps2)
    th_atan_s2 = accel_to_roll_atan2(ay_s2_ref - noise_model.ay_bias_hat_mps2, az_s2_ref - noise_model.az_bias_hat_mps2)
    lin_vs_atan_diff_deg = float(np.sqrt(np.mean((th_lin_s2 - th_atan_s2) ** 2)) * RAD2DEG)

    am = noise_model.accel_angle_model
    gm = noise_model.gyro_rate_model
    aa = noise_model.accel_angle_assumptions
    ga = noise_model.gyro_rate_assumptions

    print(f"  Small-angle linearization check (S2, max={lin_s2['max_angle_deg']:.1f} deg): max |theta-sin(theta)| = {lin_s2['max_lin_error_deg']:.3f} deg ({lin_s2['max_rel_error_pct']:.2f}%), RMS(lin - atan2) = {lin_vs_atan_diff_deg:.3f} deg")
    print(f"  Accel-angle static: bias={am.bias_hat*RAD2DEG:+.4f} deg, sigma_v={am.sigma_v_hat*RAD2DEG:.4f} deg (CI95 [{am.sigma_v_ci95[0]*RAD2DEG:.4f}, {am.sigma_v_ci95[1]*RAD2DEG:.4f}]), JB p={aa.jarque_bera_p_value:.3f}, 1s LB p={aa.block_1s_ljung_box_p:.2e}")
    print(f"  Gyro-X rate static: bias={gm.bias_hat:.5f} +/- {gm.bias_se_block:.5f} rad/s (t={ga.zero_mean_t_stat:.1f}), sigma_v={gm.sigma_v_hat:.6f} rad/s, JB p={ga.jarque_bera_p_value:.3f}, 1s LB Q={ga.block_1s_ljung_box_q:.1f} (p={ga.block_1s_ljung_box_p:.2e})")
    print(f"  Gyro Allan LM fit : N_g={gm.N_av_hat:.3e} +/- {gm.N_av_se:.1e}, B_g={gm.B_av_hat:.3e} +/- {gm.B_av_se:.1e}, T_B={gm.TB_av_hat:.1f} +/- {gm.TB_av_se:.1f} s, K_g={gm.K_av_hat:.2e} +/- {gm.K_av_se:.1e}")
    print(f"  Optimal Wiener crossover: omega_c* = {noise_model.wiener_causal_omega_c_radps:.4f} rad/s (f_c* = {noise_model.wiener_causal_fc_hz:.4f} Hz), theo min RMSE = {noise_model.wiener_theo_rmse_rad*RAD2DEG:.4f} deg")

    # Hidden truth post-hoc check (ONLY for report validation table!)
    g_cfg = st_sensor.config.gyroscope
    a_cfg = st_sensor.config.accelerometer
    true_sigma_th_a = (a_cfg.noise_density_mps2_per_sqrthz[1] * math.sqrt(fs)) / STANDARD_GRAVITY
    true_sigma_w_g = g_cfg.noise_density_radps_per_sqrthz[0] * math.sqrt(fs)
    print("\n  [Post-Hoc Check vs. Hidden Simulator Truth (NOT used in filters!)]")
    print(f"    True accel-angle sigma_v = {true_sigma_th_a*RAD2DEG:.4f} deg vs hat = {am.sigma_v_hat*RAD2DEG:.4f} deg (err={abs(am.sigma_v_hat-true_sigma_th_a)/true_sigma_th_a*100:.2f}%)")
    print(f"    True gyro-X      sigma_v = {true_sigma_w_g:.6f} rad/s vs hat = {gm.sigma_v_hat:.6f} rad/s (err={abs(gm.sigma_v_hat-true_sigma_w_g)/true_sigma_w_g*100:.2f}%)")
    print(f"    True gyro-X static bias  = {g_cfg.static_bias_radps[0]:.5f} rad/s vs hat = {gm.bias_hat:.5f} rad/s (err={abs(gm.bias_hat-g_cfg.static_bias_radps[0])/abs(g_cfg.static_bias_radps[0])*100:.2f}%)")
    print(f"    True gyro (N, B, T_B, K) = ({g_cfg.noise_density_radps_per_sqrthz[0]:.3e}, {g_cfg.bias_instability_radps[0]:.2e}, {g_cfg.bias_correlation_time_s[0]:.1f}, {g_cfg.random_walk_radps_per_sqrts[0]:.2e})")

    plot_fig1_dual_assumptions(noise_model, accel_to_roll_atan2(ay_st, az_st), gx_st, OUTPUT_DIR / "fig1_dual_assumptions_and_wiener_psd.png")
    plot_fig2_gyro_allan(noise_model, OUTPUT_DIR / "fig2_gyro_allan_identification.png")

    # --------------------------------------------------------------------------
    # STEP 3 & 4a: Single-Run Evaluation Across All 4 Roll Scenarios (seed=2026)
    # --------------------------------------------------------------------------
    print("\n[Step 3-4] Evaluating Baselines, Wiener, and 4-State Kalman [u, u', u'', b]^T across 4 Roll scenarios (seed=2026)...")
    single_run_summary: dict[str, dict] = {}
    base_gyr_bias = list(g_cfg.static_bias_radps)

    for s_name, sc in scenarios.items():
        g_ovr = None
        if sc["extra_gyro_bias"] != 0.0:
            g_ovr = {"static_bias_radps": [base_gyr_bias[0] + sc["extra_gyro_bias"], base_gyr_bias[1], base_gyr_bias[2]]}
        t_vec, ay, az, gx, _ = simulate_roll_motion(
            sc["theta"], sc["omega"], seed=2026, fs=fs, a_lin_y_mps2=sc["a_lin"], gyro_overrides=g_ovr
        )
        outputs = run_all_attitude_methods(ay, az, gx, noise_model, cf_alpha=0.96)
        kf_res = outputs["Method2_Kalman_4State"]["result"]
        kf_diag = compute_attitude_innovation_diagnostics(kf_res, burn_in=100, lags=20)

        if s_name == "S2_SmallAngle_Sin_20s":
            plot_fig5_kalman_consistency(t_vec, sc["theta"], kf_res, kf_diag, OUTPUT_DIR / "fig5_kalman_consistency_s2.png")
        elif s_name == "S3_MultiFreq_Vib_20s":
            plot_fig3_vibration_comparison(t_vec, sc["theta"], outputs, OUTPUT_DIR / "fig3_vibration_comparison_s3.png")
        elif s_name == "S4_LongTerm_Drift_180s":
            true_b_tot = base_gyr_bias[0] + sc["extra_gyro_bias"]
            plot_fig4_drift_and_bias_tracking(t_vec, sc["theta"], outputs, true_b_tot, OUTPUT_DIR / "fig4_drift_and_bias_tracking_s4.png")

        print(f"\n  --- {s_name} (Kalman q_u_MLE={outputs['Method2_Kalman_4State']['q_u_mle']:.2e}, NIS/2={kf_diag.nis_mean:.3f} in [{kf_diag.nis_ci95[0]:.3f}, {kf_diag.nis_ci95[1]:.3f}], LB p={kf_diag.ljung_box_p:.3f}) ---")
        raw_acc_ref = outputs["Baseline_AccelRaw"]["est"]
        s_metrics: dict[str, dict] = {}
        for m_name, m_data in outputs.items():
            burn = 1000 if s_name == "S4_LongTerm_Drift_180s" else 200
            met = compute_filter_metrics(
                m_data["est"], sc["theta"], raw_acc_ref, fs=fs, burn_in=burn, est_std=m_data.get("std")
            )
            s_metrics[m_name] = {
                "rmse_deg": met.rmse * RAD2DEG,
                "mae_deg": met.mae * RAD2DEG,
                "bias_deg": met.bias_mean * RAD2DEG,
                "snr_gain_vs_accel_db": met.snr_gain_db,
                "lag_ms": met.lag_ms,
                "coverage_95_pct": met.coverage_95_pct,
                "time_ms_per_sample": m_data["time_ms_per_sample"],
            }
            cov_str = f" | cov95={met.coverage_95_pct:.1f}%" if met.coverage_95_pct is not None else ""
            print(
                f"    {m_name:22s}: RMSE={met.rmse*RAD2DEG:6.3f} deg | MAE={met.mae*RAD2DEG:6.3f} deg | "
                f"Bias={met.bias_mean*RAD2DEG:+6.3f} deg | lag={met.lag_ms:+4.1f} ms{cov_str}"
            )

        single_run_summary[s_name] = {
            "metrics": s_metrics,
            "kalman_diagnostics": {
                "q_u_mle": outputs["Method2_Kalman_4State"]["q_u_mle"],
                "log10_qu_se": outputs["Method2_Kalman_4State"]["log10_qu_se"],
                "nis_mean": kf_diag.nis_mean,
                "nis_ci95": list(kf_diag.nis_ci95),
                "nis_consistent": kf_diag.nis_consistent,
                "ljung_box_q": kf_diag.ljung_box_q,
                "ljung_box_p": kf_diag.ljung_box_p,
                "acf_within_95_pct": kf_diag.acf_within_95_pct,
                "final_bias_est_radps": float(kf_res.bias_est_radps[-1]),
                "final_bias_std_radps": float(kf_res.bias_std_radps[-1]),
            },
        }

    # --------------------------------------------------------------------------
    # STEP 4b: Monte Carlo Benchmark (20 runs on S1-S3, 10 runs on S4)
    # --------------------------------------------------------------------------
    print("\n[Step 4b] Running Monte Carlo benchmark across independent seeds...")
    method_names = [
        "Baseline_AccelRaw",
        "Baseline_GyroInt",
        "Baseline_CF_Fixed",
        "Method1_Wiener_Causal",
        "Method1_Wiener_Freq",
        "Method2_Kalman_4State",
    ]
    mc_summary: dict[str, dict] = {}
    for s_name, sc in scenarios.items():
        n_runs = 10 if s_name == "S4_LongTerm_Drift_180s" else 20
        burn = 1000 if s_name == "S4_LongTerm_Drift_180s" else 200
        qu_ref = single_run_summary[s_name]["kalman_diagnostics"]["q_u_mle"]
        g_ovr = None
        if sc["extra_gyro_bias"] != 0.0:
            g_ovr = {"static_bias_radps": [base_gyr_bias[0] + sc["extra_gyro_bias"], base_gyr_bias[1], base_gyr_bias[2]]}

        acc_rmse = {m: [] for m in method_names}
        acc_cov = []
        for r_idx in range(n_runs):
            _, ay, az, gx, _ = simulate_roll_motion(
                sc["theta"], sc["omega"], seed=3000 + r_idx, fs=fs, a_lin_y_mps2=sc["a_lin"], gyro_overrides=g_ovr
            )
            outs = run_all_attitude_methods(ay, az, gx, noise_model, cf_alpha=0.96, fixed_qu=qu_ref)
            raw_acc_ref = outs["Baseline_AccelRaw"]["est"]
            for m in method_names:
                met = compute_filter_metrics(
                    outs[m]["est"], sc["theta"], raw_acc_ref, fs=fs, burn_in=burn, est_std=outs[m].get("std")
                )
                acc_rmse[m].append(met.rmse * RAD2DEG)
                if m == "Method2_Kalman_4State" and met.coverage_95_pct is not None:
                    acc_cov.append(met.coverage_95_pct)

        mc_summary[s_name] = {}
        print(f"\n  Monte Carlo ({n_runs} runs) — {s_name}:")
        for m in method_names:
            r_mean = float(np.mean(acc_rmse[m]))
            r_std = float(np.std(acc_rmse[m], ddof=1))
            mc_summary[s_name][m] = {"rmse_deg_mean": r_mean, "rmse_deg_std": r_std, "n_runs": n_runs}
            if m == "Method2_Kalman_4State":
                mc_summary[s_name][m]["coverage_95_mean"] = float(np.mean(acc_cov))
            print(f"    {m:22s}: RMSE = {r_mean:7.3f} +/- {r_std:5.3f} deg")

    # --------------------------------------------------------------------------
    # STEP 4c: Robustness Stress Tests (Lateral Vibration & Uncalibrated Gyro Bias)
    # --------------------------------------------------------------------------
    print("\n[Step 4c] Running Robustness Stress Tests (Vibration & Uncalibrated Gyro Bias)...")
    sweep_keys = ["Baseline_AccelRaw", "Baseline_CF_Fixed", "Method1_Wiener_Causal", "Method1_Wiener_Freq", "Method2_Kalman_4State"]

    # Stress Test 1: 4 Hz lateral vibration amplitude sweep on S2 trajectory (0 to 3.0 m/s^2)
    vib_amps = [0.0, 0.5, 1.0, 2.0, 3.0]
    vib_sweep: dict[str, list[float]] = {"amps": vib_amps, **{k: [] for k in sweep_keys}}
    t20 = s2["t"]
    qu_s2 = single_run_summary["S2_SmallAngle_Sin_20s"]["kalman_diagnostics"]["q_u_mle"]
    for amp in vib_amps:
        a_lin = amp * np.sin(2.0 * math.pi * 4.0 * t20)
        _, ay, az, gx, _ = simulate_roll_motion(s2["theta"], s2["omega"], seed=4100, fs=fs, a_lin_y_mps2=a_lin)
        outs = run_all_attitude_methods(ay, az, gx, noise_model, cf_alpha=0.96, fixed_qu=qu_s2)
        raw_ref = outs["Baseline_AccelRaw"]["est"]
        for k in sweep_keys:
            met = compute_filter_metrics(outs[k]["est"], s2["theta"], raw_ref, fs=fs, burn_in=200)
            vib_sweep[k].append(met.rmse * RAD2DEG)

    # Stress Test 2: Uncalibrated gyro bias step sweep (0 to 2.0 deg/s) on 120 s trajectory
    bias_steps_dps = [0.0, 0.25, 0.5, 1.0, 2.0]
    bias_sweep: dict[str, list[float]] = {"bias_dps": bias_steps_dps, **{k: [] for k in sweep_keys}}
    t120 = np.arange(int(round(120.0 * fs)), dtype=float) / fs
    th120 = (10.0 * DEG2RAD) * np.sin(2.0 * math.pi * 0.15 * t120)
    om120 = (10.0 * DEG2RAD) * (2.0 * math.pi * 0.15) * np.cos(2.0 * math.pi * 0.15 * t120)
    qu_s4 = single_run_summary["S4_LongTerm_Drift_180s"]["kalman_diagnostics"]["q_u_mle"]
    for b_dps in bias_steps_dps:
        g_ovr = {"static_bias_radps": [base_gyr_bias[0] + b_dps * DEG2RAD, base_gyr_bias[1], base_gyr_bias[2]]}
        _, ay, az, gx, _ = simulate_roll_motion(th120, om120, seed=4200, fs=fs, gyro_overrides=g_ovr)
        outs = run_all_attitude_methods(ay, az, gx, noise_model, cf_alpha=0.96, fixed_qu=qu_s4)
        raw_ref = outs["Baseline_AccelRaw"]["est"]
        for k in sweep_keys:
            met = compute_filter_metrics(outs[k]["est"], th120, raw_ref, fs=fs, burn_in=1000)
            bias_sweep[k].append(met.rmse * RAD2DEG)

    plot_fig6_montecarlo_and_stress(mc_summary, vib_sweep, bias_sweep, OUTPUT_DIR / "fig6_montecarlo_and_stress.png")

    # Export complete JSON summary
    summary = {
        "linearization_s2": {**lin_s2, "rms_lin_vs_atan2_deg": lin_vs_atan_diff_deg},
        "assumptions": {
            "accel_angle": {
                "saturation_free": aa.saturation_free,
                "static_mean_deg": aa.static_mean * RAD2DEG,
                "static_mean_se_block_deg": aa.static_mean_se_block * RAD2DEG,
                "zero_mean_t_stat": aa.zero_mean_t_stat,
                "skewness": aa.skewness,
                "excess_kurtosis": aa.excess_kurtosis,
                "jarque_bera_stat": aa.jarque_bera_stat,
                "jarque_bera_p_value": aa.jarque_bera_p_value,
                "long_horizon_nonstationary": aa.long_horizon_nonstationary,
                "block_1s_ljung_box_q": aa.block_1s_ljung_box_q,
                "block_1s_ljung_box_p": aa.block_1s_ljung_box_p,
            },
            "gyro_rate": {
                "saturation_free": ga.saturation_free,
                "static_mean_radps": ga.static_mean,
                "static_mean_se_naive_radps": ga.static_mean_se_naive,
                "static_mean_se_block_radps": ga.static_mean_se_block,
                "zero_mean_t_stat": ga.zero_mean_t_stat,
                "skewness": ga.skewness,
                "excess_kurtosis": ga.excess_kurtosis,
                "jarque_bera_stat": ga.jarque_bera_stat,
                "jarque_bera_p_value": ga.jarque_bera_p_value,
                "short_mean_spread_radps": ga.short_mean_spread,
                "long_mean_spread_radps": ga.long_mean_spread,
                "long_horizon_nonstationary": ga.long_horizon_nonstationary,
                "block_1s_ljung_box_q": ga.block_1s_ljung_box_q,
                "block_1s_ljung_box_p": ga.block_1s_ljung_box_p,
            },
        },
        "identified_parameters": {
            "accel_angle_bias_hat_deg": am.bias_hat * RAD2DEG,
            "accel_angle_sigma_v_deg": am.sigma_v_hat * RAD2DEG,
            "accel_angle_sigma_v_ci95_deg": [am.sigma_v_ci95[0] * RAD2DEG, am.sigma_v_ci95[1] * RAD2DEG],
            "accel_angle_R_hat_rad2": am.R_hat,
            "accel_angle_N_hat": am.N_diff_hat,
            "gyro_bias_hat_radps": gm.bias_hat,
            "gyro_bias_se_block_radps": gm.bias_se_block,
            "gyro_sigma_v_radps": gm.sigma_v_hat,
            "gyro_sigma_v_ci95_radps": list(gm.sigma_v_ci95),
            "gyro_N_diff_hat": gm.N_diff_hat,
            "gyro_N_psd_hat": gm.N_psd_hat,
            "gyro_N_av_hat": gm.N_av_hat,
            "gyro_N_av_se": gm.N_av_se,
            "gyro_B_av_hat": gm.B_av_hat,
            "gyro_B_av_se": gm.B_av_se,
            "gyro_TB_av_hat": gm.TB_av_hat,
            "gyro_TB_av_se": gm.TB_av_se,
            "gyro_K_av_hat": gm.K_av_hat,
            "gyro_K_av_se": gm.K_av_se,
            "gyro_phi_b_hat": gm.phi_b_hat,
            "gyro_q_b_hat": gm.q_b_hat,
            "gyro_q_k_hat": gm.q_k_hat,
            "wiener_causal_omega_c_radps": noise_model.wiener_causal_omega_c_radps,
            "wiener_causal_fc_hz": noise_model.wiener_causal_fc_hz,
            "wiener_causal_alpha_opt": 1.0 / (1.0 + noise_model.wiener_causal_omega_c_radps / fs),
            "wiener_theo_rmse_deg": noise_model.wiener_theo_rmse_rad * RAD2DEG,
        },
        "hidden_truth_reference": {
            "accel_angle_sigma_v_deg": true_sigma_th_a * RAD2DEG,
            "gyro_static_bias_radps": g_cfg.static_bias_radps[0],
            "gyro_sigma_v_radps": true_sigma_w_g,
            "gyro_N": g_cfg.noise_density_radps_per_sqrthz[0],
            "gyro_B": g_cfg.bias_instability_radps[0],
            "gyro_TB": g_cfg.bias_correlation_time_s[0],
            "gyro_K": g_cfg.random_walk_radps_per_sqrts[0],
        },
        "single_run": single_run_summary,
        "monte_carlo": mc_summary,
        "stress_tests": {
            "vibration_sweep_deg": vib_sweep,
            "uncalibrated_bias_sweep_deg": bias_sweep,
        },
    }
    (OUTPUT_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll attitude figures and summary JSON written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
