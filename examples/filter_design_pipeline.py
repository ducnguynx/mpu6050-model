"""End-to-end black-box filter design and evaluation pipeline for MPU-6050 (Direction A).

Implements a complete Random Signal Processing final-project workflow on the
1-axis accelerometer X-axis output of MPU-6050, strictly treating the sensor
model as unknown (black-box):
    Step 1: Generate independent static characterization data & dynamic test data.
    Step 2: Run statistical assumption tests & identify stochastic noise parameters
            with uncertainty purely from static sensor measurements.
    Step 3: Design and run:
            - Required Baselines: Raw, Bias-Calibrated Raw, Moving Average (MA),
              1st-order Low-Pass Filter (LPF)
            - Method 1: Causal FIR Wiener Filter (+ Non-causal Freq-Domain Wiener)
            - Method 2: Recursive Kalman Filter with AR(1) Bias State & MLE Tuning
    Step 4: Evaluate both without ground truth (Innovation ACF, Ljung-Box, NIS)
            and with ground truth (RMSE, MAE, SNR gain, lag, 95% coverage,
            Monte Carlo mean +/- std, and noise/drift stress tests).

Outputs plots and JSON summary to outputs/filtering/.
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
from mpu6050_sim.filtering import (
    IdentifiedNoiseModel,
    apply_frequency_wiener,
    compute_filter_metrics,
    compute_innovation_diagnostics,
    design_and_apply_fir_wiener,
    identify_noise_from_static_data,
    lowpass_first_order_filter,
    moving_average_filter,
    norm_ppf,
    run_assumption_tests,
    run_kalman_filter,
    tune_kalman_qu_mle,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "filtering"


# ==============================================================================
# Scenario Builders (1-Axis Accelerometer X-axis focus, Z holds gravity g0)
# ==============================================================================


def simulate_1axis_accel(
    ideal_ax: np.ndarray,
    *,
    seed: int,
    fs: float = 200.0,
    accel_overrides: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MPU6050]:
    """Run black-box MPU-6050 simulator and return (time_s, ideal_ax, meas_ax, sensor)."""
    sensor = MPU6050.from_config(CONFIG_PATH)
    sim_upd = {"sample_rate_hz": fs, "random_seed": seed}
    sensor.update_parameters(simulation=sim_upd, accelerometer=accel_overrides)
    n = ideal_ax.size
    ideal_acc = np.column_stack(
        [
            ideal_ax,
            np.zeros(n, dtype=float),
            np.full(n, STANDARD_GRAVITY, dtype=float),
        ]
    )
    ideal_gyr = np.zeros((n, 3), dtype=float)
    res = sensor.simulate(ideal_acc, ideal_gyr)
    return res.time_s, res.ideal_acceleration[:, 0], res.accelerometer[:, 0], sensor


def build_scenarios(fs: float = 200.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Define 4 test scenarios: static, sinusoidal, multi-frequency, non-stationary long."""
    t10 = np.arange(int(round(10.0 * fs)), dtype=float) / fs
    t120 = np.arange(int(round(120.0 * fs)), dtype=float) / fs

    s1_static = np.zeros_like(t10)
    s2_sin = 0.5 * np.sin(2.0 * np.pi * 0.8 * t10)
    s3_multi = 0.5 * np.sin(2.0 * np.pi * 0.5 * t10) + 0.25 * np.sin(2.0 * np.pi * 3.5 * t10)

    # Non-stationary regime shift (0-60 s low freq; 60-120 s higher freq burst + slow bias drift)
    s4_nonstat = np.where(
        t120 < 60.0,
        0.4 * np.sin(2.0 * np.pi * 0.5 * t120),
        0.4 * np.sin(2.0 * np.pi * 0.5 * t120) + 0.3 * np.sin(2.0 * np.pi * 3.0 * t120),
    )

    return {
        "S1_Static_10s": (t10, s1_static),
        "S2_Sinusoidal_10s": (t10, s2_sin),
        "S3_MultiFreq_10s": (t10, s3_multi),
        "S4_NonStat_120s": (t120, s4_nonstat),
    }


def run_all_methods_on_signal(
    meas_ax: np.ndarray,
    noise_model: IdentifiedNoiseModel,
    *,
    ma_window: int = 15,
    lpf_cutoff_hz: float = 5.0,
    wiener_order: int = 31,
    kalman_damping_tau_s: float | None = 20.0,
    fixed_qu: float | None = None,
) -> dict[str, dict]:
    """Run all baselines + Wiener + Kalman strictly from observed meas_ax."""
    fs = noise_model.sample_rate_hz
    y_cal = meas_ax - noise_model.bias_hat

    # Baseline 0a & 0b
    t0 = time.perf_counter()
    raw_est = meas_ax.copy()
    cal_est = y_cal.copy()
    t_cal = time.perf_counter() - t0

    # Baseline 1: Causal Moving Average
    t0 = time.perf_counter()
    ma_est = moving_average_filter(y_cal, window=ma_window)
    t_ma = time.perf_counter() - t0

    # Baseline 2: Causal 1st-Order Low-Pass
    t0 = time.perf_counter()
    lpf_est = lowpass_first_order_filter(y_cal, fs=fs, cutoff_hz=lpf_cutoff_hz)
    t_lpf = time.perf_counter() - t0

    # Method 1a: Causal FIR Wiener (primary Method 1)
    t0 = time.perf_counter()
    w_fir = design_and_apply_fir_wiener(y_cal, noise_model, order=wiener_order)
    t_wfir = time.perf_counter() - t0

    # Method 1b: Non-causal Frequency-Domain Wiener (offline reference)
    t0 = time.perf_counter()
    w_freq_est, f_w, h_w = apply_frequency_wiener(y_cal, noise_model)
    t_wfreq = time.perf_counter() - t0

    # Method 2: Recursive Kalman Filter with identified AR(1) bias state
    t0 = time.perf_counter()
    if fixed_qu is None:
        q_u_mle, log10_qu_se, kf_res = tune_kalman_qu_mle(
            y_cal,
            noise_model,
            include_bias_state=True,
            signal_damping_tau_s=kalman_damping_tau_s,
        )
    else:
        q_u_mle = fixed_qu
        log10_qu_se = float("nan")
        kf_res = run_kalman_filter(
            y_cal,
            noise_model,
            q_u=q_u_mle,
            include_bias_state=True,
            signal_damping_tau_s=kalman_damping_tau_s,
        )
    t_kf_total = time.perf_counter() - t0

    # Measure single-pass online Kalman runtime (without MLE grid search)
    t0 = time.perf_counter()
    _ = run_kalman_filter(
        y_cal,
        noise_model,
        q_u=q_u_mle,
        include_bias_state=True,
        signal_damping_tau_s=kalman_damping_tau_s,
    )
    t_kf_online = time.perf_counter() - t0

    n = meas_ax.size
    return {
        "Baseline_Raw": {"est": raw_est, "time_ms_per_sample": 0.0},
        "Baseline_BiasCal": {"est": cal_est, "time_ms_per_sample": t_cal * 1e3 / n},
        "Baseline_MA15": {"est": ma_est, "time_ms_per_sample": t_ma * 1e3 / n},
        "Baseline_LPF5Hz": {"est": lpf_est, "time_ms_per_sample": t_lpf * 1e3 / n},
        "Method1_Wiener_FIR": {
            "est": w_fir.estimate,
            "result": w_fir,
            "time_ms_per_sample": t_wfir * 1e3 / n,
        },
        "Method1_Wiener_Freq": {
            "est": w_freq_est,
            "freqs": f_w,
            "H": h_w,
            "time_ms_per_sample": t_wfreq * 1e3 / n,
        },
        "Method2_Kalman_AR1": {
            "est": kf_res.signal_est,
            "std": kf_res.signal_std,
            "result": kf_res,
            "q_u_mle": q_u_mle,
            "log10_qu_se": log10_qu_se,
            "time_ms_per_sample": t_kf_online * 1e3 / n,
            "tune_time_s": t_kf_total,
        },
    }


# ==============================================================================
# Plotting Utilities
# ==============================================================================


def plot_fig1_assumptions(
    t_static: np.ndarray,
    y_static: np.ndarray,
    assump,
    noise_model: IdentifiedNoiseModel,
    save_path: Path,
) -> None:
    """Figure 1: Static data assumption checks (drift, Gaussianity Q-Q, PSD, 1s ACF)."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    # (a) 1-second averaged static series showing slow bias drift (WSS violation)
    fs = noise_model.sample_rate_hz
    sps = int(round(fs))
    n_sec = y_static.size // sps
    y_1s = y_static[: n_sec * sps].reshape(n_sec, sps).mean(axis=1)
    t_1s = np.arange(n_sec, dtype=float)
    ax = axes[0, 0]
    ax.plot(t_1s / 60.0, y_1s, color="#1f77b4", linewidth=0.8, alpha=0.85, label="1 s block average")
    # 10-window means overlay
    blk_t = np.linspace(0.0, n_sec / 60.0, assump.long_window_means.size + 1)
    blk_centers = 0.5 * (blk_t[:-1] + blk_t[1:])
    ax.plot(
        blk_centers,
        assump.long_window_means,
        "o-",
        color="#d62728",
        linewidth=1.8,
        markersize=5,
        label="10-window means (non-stationary drift)",
    )
    ax.axhline(assump.static_mean, color="#111111", linestyle="--", linewidth=1.1, label=f"Mean = {assump.static_mean:.4f} m/s²")
    ax.set_title("(a) Static 1 s Averages & Window Means (WSS Check)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Measured $a_x$ [m/s²]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="best")

    # (b) Gaussianity check: Q-Q plot of standardized first differences
    ax = axes[0, 1]
    dy = np.diff(y_static)
    dy_z = (dy - np.mean(dy)) / np.std(dy, ddof=1)
    # Subsample sorted quantiles for crisp plotting
    n_q = 500
    probs = (np.arange(1, n_q + 1) - 0.5) / n_q
    theo_q = norm_ppf(probs)
    emp_q = np.quantile(dy_z, probs)
    ax.scatter(theo_q, emp_q, s=8, alpha=0.65, color="#2ca02c", label="Empirical quantiles ($\\Delta y_k$)")
    ax.plot([-3.5, 3.5], [-3.5, 3.5], "k--", linewidth=1.3, label="Ideal Standard Normal")
    ax.set_title(
        f"(b) Q-Q Plot of $\\Delta y_k$ (Skew={assump.skewness:+.4f}, ExKurt={assump.excess_kurtosis:+.4f}, JB p={assump.jarque_bera_p_value:.3f})",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Standardized Sample Quantiles")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper left")

    # (c) Welch PSD of static noise (flat white floor + low-freq colored rise)
    ax = axes[1, 0]
    f_pos = noise_model.psd_freqs[1:]
    p_pos = noise_model.psd_noise[1:]
    ax.loglog(f_pos, p_pos, color="#1f77b4", linewidth=1.2, label="Empirical Welch PSD $\\hat{S}_{vv}(f)$")
    white_floor = 2.0 * (noise_model.N_diff_hat ** 2)
    ax.axhline(
        white_floor,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label=f"White floor $2\\hat{{N}}^2$ = {white_floor:.2e} (m/s²)²/Hz",
    )
    ax.set_title("(c) Static Noise Welch PSD (White Floor + Low-Freq Colored Rise)", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Single-sided PSD [(m/s²)²/Hz]")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

    # (d) ACF of 1-second averages + Ljung-Box whiteness rejection
    ax = axes[1, 1]
    lags = np.arange(1, assump.block_1s_acf.size)
    ax.stem(lags, assump.block_1s_acf[1:], linefmt="#ff7f0e", markerfmt="o", basefmt="k-")
    bound = 1.96 / math.sqrt(n_sec)
    ax.axhline(bound, color="#111111", linestyle="--", linewidth=1.1, label=f"95% White-Noise Bound (±{bound:.3f})")
    ax.axhline(-bound, color="#111111", linestyle="--", linewidth=1.1)
    ax.set_title(
        f"(d) ACF of 1 s Block Averages (Ljung-Box Q={assump.block_1s_ljung_box_q:.1f}, p={assump.block_1s_ljung_box_p:.2e})",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.set_xlabel("Lag [s]")
    ax.set_ylabel("Autocorrelation $\\hat{\\rho}[\\tau]$")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Step 1-2: Statistical Assumption Tests on Static MPU-6050 X-Axis Data (Black-Box)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig2_allan_identification(
    noise_model: IdentifiedNoiseModel,
    save_path: Path,
) -> None:
    """Figure 2: Empirical Allan deviation with IEEE error bars + black-box fitted model."""
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    tau = noise_model.tau
    adev = noise_model.adev_emp
    se = noise_model.adev_se

    ax.loglog(tau, adev, color="#1f77b4", linewidth=2.0, label="Empirical Overlapping ADEV $\\hat{\\sigma}_A(\\tau)$")
    ax.fill_between(
        tau,
        np.maximum(adev - 1.96 * se, 1e-8),
        adev + 1.96 * se,
        color="#1f77b4",
        alpha=0.2,
        label="95% Confidence Band ($\\pm 1.96\\,\\mathrm{SE}$)",
    )
    ax.loglog(tau, noise_model.adev_fit, "k--", linewidth=1.8, label="Black-Box Weighted LM Fit (Total)")

    comps = noise_model.adev_fit_components
    ax.loglog(tau, np.sqrt(comps["white_noise"]), color="#2ca02c", linestyle=":", linewidth=1.4, label=f"White Noise: $\\hat{{N}} = {noise_model.N_av_hat:.5f} \\pm {noise_model.N_av_se:.5f}$")
    ax.loglog(tau, np.sqrt(comps["gauss_markov"]), color="#ff7f0e", linestyle="-.", linewidth=1.4, label=f"AR(1) GM Bias: $\\hat{{B}} = {noise_model.B_av_hat:.5f}$, $\\hat{{T}}_B = {noise_model.TB_av_hat:.1f}$ s")
    ax.loglog(tau, np.sqrt(comps["random_walk"]), color="#d62728", linestyle=":", linewidth=1.4, label=f"Bias Drift RW: $\\hat{{K}} = {noise_model.K_av_hat:.2e}$")

    ax.set_xlabel(r"Averaging Cluster Time $\tau$ [s]")
    ax.set_ylabel(r"Allan Deviation $\sigma_A(\tau)$ [m/s²]")
    ax.set_title(
        "Step 2: Black-Box Stochastic Identification via Overlapping Allan Deviation (Accel X-Axis)",
        fontsize=10.8,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig3_multifreq_comparison(
    t: np.ndarray,
    truth: np.ndarray,
    outputs: dict[str, dict],
    save_path: Path,
) -> None:
    """Figure 3: Multi-frequency time-domain comparison (Baselines vs. Wiener vs. Kalman)."""
    mask = (t >= 2.0) & (t <= 4.5)
    tz = t[mask]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)

    ax1.plot(tz, outputs["Baseline_BiasCal"]["est"][mask], color="#bbbbbb", linewidth=0.8, label="Baseline: Bias-Cal Raw")
    ax1.plot(tz, outputs["Baseline_MA15"]["est"][mask], color="#ff7f0e", linewidth=1.2, alpha=0.85, label="Baseline: MA (W=15)")
    ax1.plot(tz, outputs["Baseline_LPF5Hz"]["est"][mask], color="#9467bd", linewidth=1.2, alpha=0.85, label="Baseline: 1st-Order LPF (5 Hz)")
    ax1.plot(tz, outputs["Method1_Wiener_FIR"]["est"][mask], color="#2ca02c", linewidth=1.5, label="Method 1: Causal FIR Wiener")
    ax1.plot(tz, outputs["Method2_Kalman_AR1"]["est"][mask], color="#1f77b4", linewidth=1.7, label="Method 2: Kalman (CA + AR(1) Bias)")
    ax1.plot(tz, truth[mask], "k--", linewidth=1.4, label="Ground Truth (Ideal $a_x$)")
    ax1.set_ylabel("Acceleration $a_x$ [m/s²]")
    ax1.set_title("Scenario 3 (Multi-Frequency 0.5 Hz + 3.5 Hz): Signal Tracking Comparison (Zoom 2.0–4.5 s)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(ncol=3, fontsize=8.3, loc="upper right")

    ax2.plot(tz, outputs["Baseline_MA15"]["est"][mask] - truth[mask], color="#ff7f0e", linewidth=1.1, alpha=0.8, label="Error: MA (W=15)")
    ax2.plot(tz, outputs["Baseline_LPF5Hz"]["est"][mask] - truth[mask], color="#9467bd", linewidth=1.1, alpha=0.8, label="Error: LPF (5 Hz)")
    ax2.plot(tz, outputs["Method1_Wiener_FIR"]["est"][mask] - truth[mask], color="#2ca02c", linewidth=1.3, label="Error: Causal FIR Wiener")
    ax2.plot(tz, outputs["Method2_Kalman_AR1"]["est"][mask] - truth[mask], color="#1f77b4", linewidth=1.5, label="Error: Kalman (CA + AR(1))")
    ax2.axhline(0.0, color="#111111", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Estimation Error [m/s²]")
    ax2.set_title("Instantaneous Estimation Error ($\\hat{u}_k - u_k$): Notice Phase-Lag Error in MA & LPF vs. Wiener & Kalman", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(ncol=2, fontsize=8.5, loc="upper right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig4_kalman_diagnostics(
    t: np.ndarray,
    truth: np.ndarray,
    kf_res,
    diag,
    save_path: Path,
) -> None:
    """Figure 4: Kalman +/- 2sigma uncertainty band + Innovation whiteness & NIS checks."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8))

    # (a) Estimation error with +/- 1.96 sigma posterior uncertainty band
    burn = 40
    err = kf_res.signal_est[burn:] - truth[burn:]
    sd = kf_res.signal_std[burn:]
    tb = t[burn:]
    cov95 = float(np.mean(np.abs(err) <= 1.96 * sd) * 100.0)
    ax = axes[0, 0]
    ax.plot(tb, err, color="#1f77b4", linewidth=0.9, label="Kalman Error $\\hat{u}_k - u_k$")
    ax.fill_between(tb, -1.96 * sd, 1.96 * sd, color="#ff7f0e", alpha=0.25, label=f"95% Posterior Band ($\\pm 1.96\\sqrt{{P_{{uu}}}}$)")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.9)
    ax.set_title(f"(a) Kalman Error vs. Posterior 95% Band (Coverage = {cov95:.1f}%)", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error [m/s²]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="upper right")

    # (b) Standardized innovations nu_k / sqrt(S_k) (no ground truth needed)
    ax = axes[0, 1]
    z_innov = kf_res.innovations[burn:] / np.sqrt(kf_res.innovation_var[burn:])
    ax.plot(tb, z_innov, color="#2ca02c", linewidth=0.65, alpha=0.85, label="Normalized Innovation $\\nu_k / \\sqrt{S_k}$")
    ax.axhline(1.96, color="#d62728", linestyle="--", linewidth=1.1, label="±1.96 (95% Theoretical Bounds)")
    ax.axhline(-1.96, color="#d62728", linestyle="--", linewidth=1.1)
    ax.set_title("(b) Normalized Innovations (No Ground Truth Required)", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$\\nu_k / \\sqrt{S_k}$")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="upper right")

    # (c) Innovation ACF whiteness test
    ax = axes[1, 0]
    lags = np.arange(1, diag.acf.size)
    bound = 1.96 / math.sqrt(kf_res.innovations[burn:].size)
    ax.stem(lags, diag.acf[1:], linefmt="#1f77b4", markerfmt="o", basefmt="k-")
    ax.axhline(bound, color="#d62728", linestyle="--", linewidth=1.1, label=f"95% White Bound (±{bound:.3f})")
    ax.axhline(-bound, color="#d62728", linestyle="--", linewidth=1.1)
    ax.set_title(
        f"(c) Innovation ACF Whiteness (Ljung-Box Q={diag.ljung_box_q:.1f}, p={diag.ljung_box_p:.3f})",
        fontsize=9.8,
        fontweight="bold",
    )
    ax.set_xlabel("Lag [samples]")
    ax.set_ylabel("ACF $\\hat{\\rho}_\\nu[\\tau]$")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="upper right")

    # (d) Running mean NIS consistency check
    ax = axes[1, 1]
    nis_seq = kf_res.nis[burn:]
    running_nis = np.cumsum(nis_seq) / np.arange(1, nis_seq.size + 1)
    ax.plot(tb, running_nis, color="#9467bd", linewidth=1.4, label=f"Running Mean NIS (Final = {diag.nis_mean:.3f})")
    ax.axhline(1.0, color="k", linestyle="-", linewidth=1.0, label="Ideal E[NIS] = 1.0")
    ax.axhspan(diag.nis_ci95[0], diag.nis_ci95[1], color="#2ca02c", alpha=0.2, label=f"95% $\\chi^2$ Consistency Band [{diag.nis_ci95[0]:.3f}, {diag.nis_ci95[1]:.3f}]")
    ax.set_title("(d) Normalized Innovation Squared (NIS) Consistency Check", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Running Mean NIS")
    ax.set_ylim(0.5, 1.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.2, loc="upper right")

    fig.suptitle(
        "Step 3-4: Kalman Filter Uncertainty Quantification & No-Ground-Truth Consistency Diagnostics (Scenario 3)",
        fontsize=11.2,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig5_montecarlo_and_drift(
    mc_summary: dict[str, dict[str, dict[str, float]]],
    save_path: Path,
) -> None:
    """Figure 5: Monte Carlo RMSE (mean +/- std) across all 4 scenarios."""
    scenarios = list(mc_summary.keys())
    methods = [
        ("Baseline_BiasCal", "Bias-Cal Raw", "#999999"),
        ("Baseline_MA15", "MA (W=15)", "#ff7f0e"),
        ("Baseline_LPF5Hz", "LPF (5 Hz)", "#9467bd"),
        ("Method1_Wiener_FIR", "Wiener (FIR)", "#2ca02c"),
        ("Method1_Wiener_Freq", "Wiener (Freq, Offline)", "#17becf"),
        ("Method2_Kalman_AR1", "Kalman (AR1 Bias)", "#1f77b4"),
    ]

    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    x = np.arange(len(scenarios))
    width = 0.13
    max_bar = max(
        mc_summary[s][m_key]["rmse_mean"] * 1e3 + mc_summary[s][m_key]["rmse_std"] * 1e3
        for s in scenarios
        for m_key, _, _ in methods
    )
    for idx, (m_key, m_label, color) in enumerate(methods):
        means = [mc_summary[s][m_key]["rmse_mean"] * 1e3 for s in scenarios]
        stds = [mc_summary[s][m_key]["rmse_std"] * 1e3 for s in scenarios]
        offset = (idx - (len(methods) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=3,
            color=color,
            alpha=0.9,
            label=m_label,
        )
        for b, val, sd in zip(bars, means, stds):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                val + sd + 2.0,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "S1: Static (10 s)",
            "S2: Sinusoidal 0.8 Hz",
            "S3: Multi-Freq (0.5+3.5 Hz)",
            "S4: Non-Stationary (120 s)",
        ],
        fontsize=9.5,
    )
    ax.set_ylabel("RMSE [mm/s²] (lower is better)")
    ax.set_title(
        "Step 4: Monte Carlo Benchmark (Mean ± 1 SD RMSE across Independent Seeds) — Baselines vs. Wiener vs. Kalman",
        fontsize=10.8,
        fontweight="bold",
    )
    ax.set_ylim(0, max_bar * 1.24)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(ncol=3, fontsize=8.8, loc="upper left")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fig6_stress_tests(
    noise_sweep: dict[str, list[float]],
    drift_sweep: dict[str, list[float]],
    save_path: Path,
) -> None:
    """Figure 6: Stress-test sensitivity to white noise scaling & unmodeled bias drift."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.2))

    styles = [
        ("Baseline_BiasCal", "Bias-Cal Raw", "#888888", "--", "s"),
        ("Baseline_MA15", "MA (W=15)", "#ff7f0e", "-.", "^"),
        ("Baseline_LPF5Hz", "LPF (5 Hz)", "#9467bd", "-.", "v"),
        ("Method1_Wiener_FIR", "Wiener (FIR)", "#2ca02c", "-", "o"),
        ("Method2_Kalman_AR1", "Kalman (AR1 Bias)", "#1f77b4", "-", "D"),
    ]

    scales_n = noise_sweep["scales"]
    for key, label, color, ls, mk in styles:
        ax1.plot(scales_n, [v * 1e3 for v in noise_sweep[key]], color=color, linestyle=ls, marker=mk, linewidth=1.6, label=label)
    ax1.set_title("(a) White Noise Stress Test (Scenario 3, Noise Density Sweep)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("White Noise Density Multiplier ($\\times$ Nominal)")
    ax1.set_ylabel("RMSE [mm/s²]")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8.3, loc="lower right")

    scales_k = drift_sweep["scales"]
    for key, label, color, ls, mk in styles:
        ax2.plot(scales_k, [v * 1e3 for v in drift_sweep[key]], color=color, linestyle=ls, marker=mk, linewidth=1.6, label=label)
    ax2.set_title("(b) Long-Term Drift Stress Test (120 s Run, Random-Walk $K_a$ Sweep)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Bias Random-Walk Drift Multiplier ($\\times$ Nominal)")
    ax2.set_ylabel("RMSE [mm/s²]")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8.3, loc="center right")

    fig.suptitle("Step 4: Robustness & Sensitivity Analysis (Stress Tests)", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ==============================================================================
# Main Pipeline Execution
# ==============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fs = 200.0
    print("=" * 78)
    print("MPU-6050 BLACK-BOX FILTER DESIGN PIPELINE (DIRECTION A: ACCEL X-AXIS)")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # STEP 1: Black-Box Data Acquisition (Independent Seeds!)
    # --------------------------------------------------------------------------
    print("\n[Step 1] Generating 1-hour static characterization data (seed=1001, fs=200 Hz)...")
    t_static_len = int(round(3600.0 * fs))
    t_static, _, y_static, static_sensor = simulate_1axis_accel(
        np.zeros(t_static_len, dtype=float), seed=1001, fs=fs
    )
    scenarios = build_scenarios(fs=fs)
    t_s3, ideal_s3 = scenarios["S3_MultiFreq_10s"]
    _, _, y_s3_sample, _ = simulate_1axis_accel(ideal_s3, seed=2026, fs=fs)

    # --------------------------------------------------------------------------
    # STEP 2: Statistical Assumption Tests & Black-Box Stochastic Identification
    # --------------------------------------------------------------------------
    print("[Step 2] Running statistical assumption tests & black-box noise identification...")
    fsr_limit = static_sensor.config.accelerometer.full_scale_range_g * STANDARD_GRAVITY
    assump = run_assumption_tests(y_static, y_s3_sample, fs=fs, fsr_limit=fsr_limit)
    noise_model = identify_noise_from_static_data(
        y_static, fs=fs, calibration_window_s=10.0, wiener_max_lag=128, welch_nperseg=4096
    )

    print(f"  Saturation-free (<95% FSR={fsr_limit:.2f} m/s^2): {assump.saturation_free} (max_dyn={assump.max_abs_dynamic:.3f})")
    print(f"  Static mean (effective bias): {assump.static_mean:.5f} +/- {assump.static_mean_se_block:.5f} m/s^2 (t={assump.zero_mean_t_stat:.1f}, p={assump.zero_mean_p_value:.2e})")
    print(f"  Gaussianity (first diff): skew={assump.skewness:+.4f}, ex_kurt={assump.excess_kurtosis:+.4f}, JB={assump.jarque_bera_stat:.2f}, p={assump.jarque_bera_p_value:.3f}")
    print(f"  WSS Check: short-10s mean spread={assump.short_mean_spread*1e3:.3f} mm/s^2 | long-1h mean spread={assump.long_mean_spread*1e3:.3f} mm/s^2 (nonstationary={assump.long_horizon_nonstationary})")
    print(f"  Colored-noise check (1 s block Ljung-Box): Q={assump.block_1s_ljung_box_q:.1f}, p={assump.block_1s_ljung_box_p:.2e} -> colored={assump.has_colored_noise}")
    print("\n  Black-Box Identified Parameters (with Uncertainty):")
    print(f"    b_0_hat (10 s cal) = {noise_model.bias_hat:.5f} +/- {noise_model.bias_se_naive:.5f} (naive SE) / +/- {noise_model.bias_se_block:.5f} (block SE) m/s^2")
    print(f"    sigma_v_hat (diff) = {noise_model.sigma_v_hat:.5f} m/s^2, 95% CI = [{noise_model.sigma_v_ci95[0]:.5f}, {noise_model.sigma_v_ci95[1]:.5f}] -> R_hat = {noise_model.R_hat:.6e}")
    print(f"    N_hat (diff / PSD / ADEV) = {noise_model.N_diff_hat:.6f} / {noise_model.N_psd_hat:.6f} / {noise_model.N_av_hat:.6f} +/- {noise_model.N_av_se:.6f}")
    print(f"    B_hat (ADEV LM)    = {noise_model.B_av_hat:.6f} +/- {noise_model.B_av_se:.6f} m/s^2")
    print(f"    T_B_hat (ADEV LM)  = {noise_model.TB_av_hat:.1f} +/- {noise_model.TB_av_se:.1f} s -> phi_B = {noise_model.phi_b_hat:.7f}")
    print(f"    K_hat (ADEV LM)    = {noise_model.K_av_hat:.2e} +/- {noise_model.K_av_se:.2e} (m/s^2)/sqrt(s)")

    # Hidden ground-truth parameters (ONLY printed for post-hoc verification in report!)
    acc_cfg = static_sensor.config.accelerometer
    true_b_eff = acc_cfg.static_bias_mps2[0] + acc_cfg.misalignment[0][2] * STANDARD_GRAVITY
    true_sigma_v = acc_cfg.noise_density_mps2_per_sqrthz[0] * math.sqrt(fs)
    print("\n  [Post-Hoc Verification vs. Hidden Simulator Truth (NOT used in filters!)]")
    print(f"    True effective static bias (b0_x + M_xz*g0) = {true_b_eff:.5f} m/s^2 (rel err = {abs(noise_model.bias_hat - true_b_eff)/abs(true_b_eff)*100:.2f}%)")
    print(f"    True white noise sigma_v                    = {true_sigma_v:.5f} m/s^2 (rel err = {abs(noise_model.sigma_v_hat - true_sigma_v)/true_sigma_v*100:.2f}%)")
    print(f"    True (N, B, T_B, K)                         = ({acc_cfg.noise_density_mps2_per_sqrthz[0]:.6f}, {acc_cfg.bias_instability_mps2[0]:.6f}, {acc_cfg.bias_correlation_time_s[0]:.1f}, {acc_cfg.random_walk_mps2_per_sqrts[0]:.2e})")

    plot_fig1_assumptions(t_static, y_static, assump, noise_model, OUTPUT_DIR / "fig1_assumption_checks.png")
    plot_fig2_allan_identification(noise_model, OUTPUT_DIR / "fig2_allan_identification.png")

    # --------------------------------------------------------------------------
    # STEP 3 & 4a: Single-Run Detailed Evaluation Across All 4 Scenarios
    # --------------------------------------------------------------------------
    print("\n[Step 3-4] Evaluating Baselines, Wiener, and Kalman across 4 scenarios (seed=2026)...")
    single_run_summary: dict[str, dict] = {}
    s3_outputs_saved = None
    s3_truth_saved = None

    for s_name, (t_vec, ideal_ax) in scenarios.items():
        _, tru, meas, _ = simulate_1axis_accel(ideal_ax, seed=2026, fs=fs)
        # For zero-signal static scenario, disable damping bias pull; for AC dynamic signals, enable slow damping (tau=20s)
        damping = None if s_name == "S1_Static_10s" else 20.0
        outputs = run_all_methods_on_signal(meas, noise_model, kalman_damping_tau_s=damping)

        kf_res = outputs["Method2_Kalman_AR1"]["result"]
        kf_diag = compute_innovation_diagnostics(kf_res, burn_in=40, lags=20)
        wfir_res = outputs["Method1_Wiener_FIR"]["result"]

        if s_name == "S3_MultiFreq_10s":
            s3_outputs_saved = outputs
            s3_truth_saved = tru
            plot_fig3_multifreq_comparison(t_vec, tru, outputs, OUTPUT_DIR / "fig3_scenario_comparison.png")
            plot_fig4_kalman_diagnostics(t_vec, tru, kf_res, kf_diag, OUTPUT_DIR / "fig4_kalman_uncertainty_and_innovations.png")

        print(f"\n  --- {s_name} (Kalman q_u_MLE = {outputs['Method2_Kalman_AR1']['q_u_mle']:.2e}, NIS = {kf_diag.nis_mean:.3f} in {kf_diag.nis_ci95}, LB p = {kf_diag.ljung_box_p:.3f}) ---")
        s_metrics: dict[str, dict] = {}
        for m_name, m_data in outputs.items():
            est_std = m_data.get("std")
            met = compute_filter_metrics(m_data["est"], tru, meas, fs=fs, burn_in=40, est_std=est_std)
            s_metrics[m_name] = {
                "rmse": met.rmse,
                "mae": met.mae,
                "bias_mean": met.bias_mean,
                "snr_in_db": met.snr_in_db,
                "snr_out_db": met.snr_out_db,
                "snr_gain_db": met.snr_gain_db,
                "lag_samples": met.lag_samples,
                "lag_ms": met.lag_ms,
                "coverage_95_pct": met.coverage_95_pct,
                "time_ms_per_sample": m_data["time_ms_per_sample"],
            }
            cov_str = f" | cov95={met.coverage_95_pct:.1f}%" if met.coverage_95_pct is not None else ""
            print(
                f"    {m_name:22s}: RMSE={met.rmse*1e3:6.2f} mm/s^2 | MAE={met.mae*1e3:6.2f} | "
                f"SNR_gain={met.snr_gain_db:+6.2f} dB | lag={met.lag_ms:+5.1f} ms{cov_str}"
            )

        single_run_summary[s_name] = {
            "metrics": s_metrics,
            "wiener_fir_rmse_theory": wfir_res.rmse_theory,
            "kalman_diagnostics": {
                "q_u_mle": outputs["Method2_Kalman_AR1"]["q_u_mle"],
                "log10_qu_se": outputs["Method2_Kalman_AR1"]["log10_qu_se"],
                "nis_mean": kf_diag.nis_mean,
                "nis_ci95": list(kf_diag.nis_ci95),
                "nis_consistent": kf_diag.nis_consistent,
                "ljung_box_q": kf_diag.ljung_box_q,
                "ljung_box_p": kf_diag.ljung_box_p,
                "acf_within_95_pct": kf_diag.acf_within_95_pct,
            },
        }

    # --------------------------------------------------------------------------
    # STEP 4b: Monte Carlo Evaluation (20 Seeds on S1-S3, 10 Seeds on S4)
    # --------------------------------------------------------------------------
    print("\n[Step 4b] Running Monte Carlo evaluation across independent seeds...")
    mc_summary: dict[str, dict[str, dict[str, float]]] = {}
    method_names = [
        "Baseline_Raw",
        "Baseline_BiasCal",
        "Baseline_MA15",
        "Baseline_LPF5Hz",
        "Method1_Wiener_FIR",
        "Method1_Wiener_Freq",
        "Method2_Kalman_AR1",
    ]
    for s_name, (_, ideal_ax) in scenarios.items():
        n_runs = 10 if s_name == "S4_NonStat_120s" else 20
        damping = None if s_name == "S1_Static_10s" else 20.0
        # Reuse the scenario's MLE-tuned q_u for fast Monte Carlo consistency
        qu_ref = single_run_summary[s_name]["kalman_diagnostics"]["q_u_mle"]
        acc_rmse = {m: [] for m in method_names}
        acc_snr = {m: [] for m in method_names}
        acc_cov = []

        for r_idx in range(n_runs):
            _, tru, meas, _ = simulate_1axis_accel(ideal_ax, seed=3000 + r_idx, fs=fs)
            outs = run_all_methods_on_signal(
                meas, noise_model, kalman_damping_tau_s=damping, fixed_qu=qu_ref
            )
            for m in method_names:
                met = compute_filter_metrics(
                    outs[m]["est"], tru, meas, fs=fs, burn_in=40, est_std=outs[m].get("std")
                )
                acc_rmse[m].append(met.rmse)
                acc_snr[m].append(met.snr_gain_db)
                if m == "Method2_Kalman_AR1" and met.coverage_95_pct is not None:
                    acc_cov.append(met.coverage_95_pct)

        mc_summary[s_name] = {}
        print(f"\n  Monte Carlo ({n_runs} runs) — {s_name}:")
        for m in method_names:
            r_mean = float(np.mean(acc_rmse[m]))
            r_std = float(np.std(acc_rmse[m], ddof=1))
            g_mean = float(np.mean(acc_snr[m]))
            g_std = float(np.std(acc_snr[m], ddof=1))
            mc_summary[s_name][m] = {
                "rmse_mean": r_mean,
                "rmse_std": r_std,
                "snr_gain_mean": g_mean,
                "snr_gain_std": g_std,
                "n_runs": n_runs,
            }
            if m == "Method2_Kalman_AR1":
                mc_summary[s_name][m]["coverage_95_mean"] = float(np.mean(acc_cov))
            print(f"    {m:22s}: RMSE = {r_mean*1e3:6.2f} +/- {r_std*1e3:4.2f} mm/s^2 | SNR gain = {g_mean:+6.2f} +/- {g_std:4.2f} dB")

    plot_fig5_montecarlo_and_drift(mc_summary, OUTPUT_DIR / "fig5_long_term_drift_and_montecarlo.png")

    # --------------------------------------------------------------------------
    # STEP 4c: Stress-Test Sensitivity Analysis (Noise Scaling & Drift Scaling)
    # --------------------------------------------------------------------------
    print("\n[Step 4c] Running Stress-Test sensitivity sweeps...")
    sweep_methods = ["Baseline_BiasCal", "Baseline_MA15", "Baseline_LPF5Hz", "Method1_Wiener_FIR", "Method2_Kalman_AR1"]

    # Sweep 1: White noise density scaling on Scenario 3 (black-box re-identifying R from 60 s static check)
    noise_scales = [0.5, 1.0, 2.0, 4.0]
    noise_sweep: dict[str, list[float]] = {"scales": noise_scales, **{m: [] for m in sweep_methods}}
    base_n = list(acc_cfg.noise_density_mps2_per_sqrthz)
    for sc in noise_scales:
        n_upd = {"noise_density_mps2_per_sqrthz": [v * sc for v in base_n]}
        _, _, y_st_sc, _ = simulate_1axis_accel(np.zeros(int(round(60.0 * fs))), seed=4100, fs=fs, accel_overrides=n_upd)
        # Update only R_hat & r_vv from 60 s static check (keeping AR(1) params from main identification)
        dy_sc = np.diff(y_st_sc)
        r_hat_sc = 0.5 * float(np.var(dy_sc, ddof=1))
        b_hat_sc = float(np.mean(y_st_sc[: int(10.0 * fs)]))
        r_vv_sc = np.zeros_like(noise_model.r_vv)
        r_vv_sc[0] = r_hat_sc
        nm_sc = IdentifiedNoiseModel(
            **{
                **noise_model.__dict__,
                "bias_hat": b_hat_sc,
                "sigma_v_hat": math.sqrt(r_hat_sc),
                "R_hat": r_hat_sc,
                "r_vv": r_vv_sc,
            }
        )
        _, tru, meas, _ = simulate_1axis_accel(ideal_s3, seed=4200, fs=fs, accel_overrides=n_upd)
        outs = run_all_methods_on_signal(meas, nm_sc, kalman_damping_tau_s=20.0)
        for m in sweep_methods:
            met = compute_filter_metrics(outs[m]["est"], tru, meas, fs=fs, burn_in=40)
            noise_sweep[m].append(met.rmse)

    # Sweep 2: Unexpected long-term bias random-walk drift K_a multiplier on 120 s Scenario 4 (NO re-calibration!)
    drift_scales = [0.0, 1.0, 5.0, 15.0, 30.0]
    drift_sweep: dict[str, list[float]] = {"scales": drift_scales, **{m: [] for m in sweep_methods}}
    base_k = list(acc_cfg.random_walk_mps2_per_sqrts)
    _, ideal_s4 = scenarios["S4_NonStat_120s"]
    qu_s4 = single_run_summary["S4_NonStat_120s"]["kalman_diagnostics"]["q_u_mle"]
    for sc in drift_scales:
        k_upd = {"random_walk_mps2_per_sqrts": [v * sc for v in base_k]}
        # Average over 5 seeds for smooth drift curve
        tmp = {m: [] for m in sweep_methods}
        for s_i in range(5):
            _, tru, meas, _ = simulate_1axis_accel(ideal_s4, seed=5000 + s_i, fs=fs, accel_overrides=k_upd)
            # For unexpected drift stress test, give Kalman higher bias diffusion tolerance via online MLE or nominal model
            outs = run_all_methods_on_signal(meas, noise_model, kalman_damping_tau_s=20.0, fixed_qu=qu_s4)
            for m in sweep_methods:
                met = compute_filter_metrics(outs[m]["est"], tru, meas, fs=fs, burn_in=40)
                tmp[m].append(met.rmse)
        for m in sweep_methods:
            drift_sweep[m].append(float(np.mean(tmp[m])))

    plot_fig6_stress_tests(noise_sweep, drift_sweep, OUTPUT_DIR / "fig6_stress_test_sensitivity.png")

    # Save structured summary JSON for exact citation in Markdown report
    summary_payload = {
        "assumptions": {
            "saturation_free": assump.saturation_free,
            "max_abs_static": assump.max_abs_static,
            "max_abs_dynamic": assump.max_abs_dynamic,
            "fsr_limit": assump.fsr_limit,
            "static_mean": assump.static_mean,
            "static_mean_se_naive": assump.static_mean_se_naive,
            "static_mean_se_block": assump.static_mean_se_block,
            "zero_mean_t_stat": assump.zero_mean_t_stat,
            "zero_mean_p_value": assump.zero_mean_p_value,
            "skewness": assump.skewness,
            "excess_kurtosis": assump.excess_kurtosis,
            "jarque_bera_stat": assump.jarque_bera_stat,
            "jarque_bera_p_value": assump.jarque_bera_p_value,
            "short_mean_spread": assump.short_mean_spread,
            "short_std_ratio": assump.short_std_ratio,
            "long_mean_spread": assump.long_mean_spread,
            "long_std_ratio": assump.long_std_ratio,
            "long_horizon_nonstationary": assump.long_horizon_nonstationary,
            "block_1s_ljung_box_q": assump.block_1s_ljung_box_q,
            "block_1s_ljung_box_p": assump.block_1s_ljung_box_p,
            "raw_ljung_box_q": assump.raw_ljung_box_q,
            "raw_ljung_box_p": assump.raw_ljung_box_p,
            "has_colored_noise": assump.has_colored_noise,
        },
        "identified_noise_model": {
            "bias_hat": noise_model.bias_hat,
            "bias_se_naive": noise_model.bias_se_naive,
            "bias_se_block": noise_model.bias_se_block,
            "sigma_v_hat": noise_model.sigma_v_hat,
            "sigma_v_ci95": list(noise_model.sigma_v_ci95),
            "R_hat": noise_model.R_hat,
            "N_diff_hat": noise_model.N_diff_hat,
            "N_psd_hat": noise_model.N_psd_hat,
            "N_av_hat": noise_model.N_av_hat,
            "N_av_se": noise_model.N_av_se,
            "B_av_hat": noise_model.B_av_hat,
            "B_av_se": noise_model.B_av_se,
            "TB_av_hat": noise_model.TB_av_hat,
            "TB_av_se": noise_model.TB_av_se,
            "K_av_hat": noise_model.K_av_hat,
            "K_av_se": noise_model.K_av_se,
            "phi_b_hat": noise_model.phi_b_hat,
            "q_b_hat": noise_model.q_b_hat,
            "q_k_hat": noise_model.q_k_hat,
            "sigma_gm_hat": noise_model.sigma_gm_hat,
        },
        "hidden_truth_reference": {
            "effective_static_bias": true_b_eff,
            "sigma_v": true_sigma_v,
            "N": acc_cfg.noise_density_mps2_per_sqrthz[0],
            "B": acc_cfg.bias_instability_mps2[0],
            "T_B": acc_cfg.bias_correlation_time_s[0],
            "K": acc_cfg.random_walk_mps2_per_sqrts[0],
        },
        "single_run": single_run_summary,
        "monte_carlo": mc_summary,
        "stress_tests": {
            "white_noise_sweep": noise_sweep,
            "drift_sweep": drift_sweep,
        },
    }
    json_path = OUTPUT_DIR / "results_summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"\nAll figures and summary JSON written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
