"""Black-box noise identification, assumption testing, baselines, Wiener, and Kalman filters.

Implements Direction A (1-axis MPU-6050 signal filtering) purely from sensor data
(treating the underlying sensor model as unknown):
    - Statistical assumption checks (linearity/saturation, zero-mean t-test,
      Gaussianity Jarque-Bera + Q-Q, WSS window check, Ljung-Box whiteness test)
    - Black-box stochastic parameter identification with uncertainty (static bias,
      first-difference white noise R, Welch PSD, Allan deviation + weighted
      Levenberg-Marquardt fit for N, B, T_B, K and AR(1) parameters)
    - Required baselines: Raw, Bias-Calibrated Raw, Moving Average, 1st-order Low-Pass
    - Method 1: Causal FIR Wiener filter (Wiener-Hopf) & Non-causal Frequency Wiener
    - Method 2: Kalman filter with AR(1) colored bias state & MLE innovation tuning
    - Diagnostics without ground truth (Innovation ACF, Ljung-Box, NIS) and with
      ground truth (RMSE, MAE, Bias, SNR improvement, phase lag, 95% coverage)

Pure NumPy + Python stdlib (no SciPy dependency required).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .noise import (
    BIAS_INSTABILITY_AV_FACTOR,
    GAUSS_MARKOV_PEAK_FACTOR,
    bias_instability_to_gm_psd,
    overlapping_allan_variance,
    theoretical_allan_variance,
)


# ==============================================================================
# Statistical Helpers (Pure NumPy / Stdlib)
# ==============================================================================


def normal_cdf(x: float) -> float:
    """Standard normal CDF Phi(x)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def normal_sf_two_sided(z: float) -> float:
    """Two-sided p-value for standard normal z-statistic."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def norm_ppf(p: np.ndarray) -> np.ndarray:
    """Acklam's rational approximation to standard normal quantile (max error < 1.15e-9)."""
    p_arr = np.asarray(p, dtype=float)
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow

    out = np.empty_like(p_arr)
    low = p_arr < plow
    high = p_arr > phigh
    mid = (~low) & (~high)

    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p_arr[low]))
        out[low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if np.any(mid):
        q = p_arr[mid] - 0.5
        r = q * q
        out[mid] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p_arr[high]))
        out[high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    return out


def chi2_sf_even_df(x: float, df: int) -> float:
    """Exact survival function P(chi2_df > x) for positive even degrees of freedom."""
    if df <= 0 or df % 2 != 0:
        raise ValueError(f"df must be a positive even integer, got {df}")
    if x <= 0.0:
        return 1.0
    half_x = 0.5 * x
    k = df // 2
    term = 1.0
    total = 1.0
    for j in range(1, k):
        term *= half_x / j
        total += term
    # Guard against underflow/overflow for very large x
    log_val = -half_x + math.log(total)
    if log_val < -700.0:
        return 0.0
    return min(1.0, math.exp(log_val))


def chi2_mean_ci95(n_dof: int) -> tuple[float, float]:
    """95% confidence bounds for chi2(n_dof) / n_dof using Wilson-Hilferty approximation."""
    z = 1.959963984540054
    h = 2.0 / (9.0 * n_dof)
    lo = (1.0 - h - z * math.sqrt(h)) ** 3
    hi = (1.0 - h + z * math.sqrt(h)) ** 3
    return float(max(0.0, lo)), float(hi)


def sample_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocorrelation function rho[0..max_lag] via FFT."""
    arr = np.asarray(x, dtype=float)
    arr = arr - np.mean(arr)
    n = arr.size
    if max_lag >= n:
        raise ValueError(f"max_lag ({max_lag}) must be < len(x) ({n})")
    denom = float(np.dot(arr, arr))
    if denom == 0.0:
        out = np.zeros(max_lag + 1, dtype=float)
        out[0] = 1.0
        return out
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(arr, n=nfft)
    acov = np.fft.irfft(spec * np.conj(spec), n=nfft)[: max_lag + 1]
    return acov / denom


def sample_autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocovariance R[0..max_lag] = (1/N) sum x[k] x[k-lag].

    Biased estimator guarantees a positive-definite Toeplitz autocorrelation matrix
    for Wiener-Hopf equations.
    """
    arr = np.asarray(x, dtype=float)
    n = arr.size
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(arr, n=nfft)
    acov = np.fft.irfft(spec * np.conj(spec), n=nfft)[: max_lag + 1] / n
    return acov


def ljung_box_test(x: np.ndarray, lags: int = 20) -> tuple[float, float, np.ndarray]:
    """Ljung-Box portmanteau whiteness test (even lags -> exact chi2 p-value)."""
    if lags % 2 != 0:
        lags += 1
    arr = np.asarray(x, dtype=float)
    n = arr.size
    acf = sample_acf(arr, max_lag=lags)
    rho = acf[1:]
    idx = np.arange(1, lags + 1, dtype=float)
    q_stat = float(n * (n + 2) * np.sum((rho ** 2) / (n - idx)))
    p_val = chi2_sf_even_df(q_stat, df=lags)
    return q_stat, p_val, acf


def welch_psd(
    x: np.ndarray,
    fs: float,
    nperseg: int = 2048,
    overlap_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided Welch Power Spectral Density [unit^2 / Hz] in pure NumPy."""
    arr = np.asarray(x, dtype=float)
    n = arr.size
    seg_len = min(nperseg, n)
    step = max(1, int(round(seg_len * (1.0 - overlap_ratio))))
    window = np.hanning(seg_len)
    win_power = float(np.sum(window ** 2))

    starts = range(0, n - seg_len + 1, step)
    psd_accum = np.zeros(seg_len // 2 + 1, dtype=float)
    count = 0
    for s in starts:
        seg = arr[s : s + seg_len]
        seg = seg - np.mean(seg)
        fft_val = np.fft.rfft(seg * window)
        psd_accum += (np.abs(fft_val) ** 2) / (fs * win_power)
        count += 1

    psd = psd_accum / max(1, count)
    # Single-sided scaling (double interior frequencies, keep DC and Nyquist)
    if seg_len % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
    return freqs, psd


# ==============================================================================
# Step 1 & 2: Statistical Assumption Tests + Black-Box Noise Identification
# ==============================================================================


@dataclass(frozen=True)
class AssumptionTestReport:
    """Summary of statistical assumption tests computed purely from sensor data."""

    # 1. Linearity / no saturation
    max_abs_static: float
    max_abs_dynamic: float
    fsr_limit: float
    saturation_free: bool

    # 2. Zero-mean check on static data
    static_mean: float
    static_mean_se_naive: float
    static_mean_se_block: float
    zero_mean_t_stat: float
    zero_mean_p_value: float
    has_nonzero_bias: bool

    # 3. Gaussianity on first-difference residuals (removes slow drift)
    diff_std: float
    skewness: float
    excess_kurtosis: float
    jarque_bera_stat: float
    jarque_bera_p_value: float
    approx_gaussian: bool

    # 4. Wide-sense stationarity (short horizon vs. long horizon)
    short_window_means: np.ndarray
    short_window_stds: np.ndarray
    short_mean_spread: float
    short_std_ratio: float
    long_window_means: np.ndarray
    long_window_stds: np.ndarray
    long_mean_spread: float
    long_std_ratio: float
    long_horizon_nonstationary: bool

    # 5. White vs. colored noise (Ljung-Box on 1 s block averages)
    block_1s_ljung_box_q: float
    block_1s_ljung_box_p: float
    block_1s_acf: np.ndarray
    raw_ljung_box_q: float
    raw_ljung_box_p: float
    has_colored_noise: bool


def run_assumption_tests(
    y_static: np.ndarray,
    y_dynamic: np.ndarray,
    fs: float,
    fsr_limit: float,
    *,
    short_horizon_s: float = 10.0,
) -> AssumptionTestReport:
    """Run all statistical assumption checks strictly on observed measurements."""
    ys = np.asarray(y_static, dtype=float)
    yd = np.asarray(y_dynamic, dtype=float)
    n_s = ys.size

    # 1. Saturation / linearity bound
    max_s = float(np.max(np.abs(ys)))
    max_d = float(np.max(np.abs(yd)))
    sat_free = (max_s < 0.95 * fsr_limit) and (max_d < 0.95 * fsr_limit)

    # 2. Zero-mean test (naive SE vs. 10-block SE accounting for slow drift)
    mean_s = float(np.mean(ys))
    std_s = float(np.std(ys, ddof=1))
    se_naive = std_s / math.sqrt(n_s)
    n_blocks = 10
    blk_len = n_s // n_blocks
    blk_means = np.array([np.mean(ys[i * blk_len : (i + 1) * blk_len]) for i in range(n_blocks)])
    blk_stds = np.array([np.std(ys[i * blk_len : (i + 1) * blk_len], ddof=1) for i in range(n_blocks)])
    se_block = float(np.std(blk_means, ddof=1) / math.sqrt(n_blocks))
    t_stat = mean_s / max(se_block, 1e-15)
    p_zero = normal_sf_two_sided(t_stat)

    # 3. Gaussianity test on first differences dy[k] = y[k] - y[k-1]
    dy = np.diff(ys)
    dy_c = dy - np.mean(dy)
    dy_std = float(np.std(dy_c, ddof=1))
    z_dy = dy_c / max(dy_std, 1e-15)
    skew = float(np.mean(z_dy ** 3))
    ex_kurt = float(np.mean(z_dy ** 4) - 3.0)
    jb_stat = float((dy.size / 6.0) * (skew ** 2 + 0.25 * ex_kurt ** 2))
    jb_p = chi2_sf_even_df(jb_stat, df=2)

    # 4. WSS check: short horizon (5 windows over first short_horizon_s) vs. long horizon (10 blocks)
    n_short = min(n_s, int(round(short_horizon_s * fs)))
    ys_short = ys[:n_short]
    s_blks = 5
    s_len = n_short // s_blks
    short_means = np.array([np.mean(ys_short[i * s_len : (i + 1) * s_len]) for i in range(s_blks)])
    short_stds = np.array([np.std(ys_short[i * s_len : (i + 1) * s_len], ddof=1) for i in range(s_blks)])
    short_spread = float(np.max(short_means) - np.min(short_means))
    short_std_ratio = float(np.max(short_stds) / max(np.min(short_stds), 1e-15))

    long_spread = float(np.max(blk_means) - np.min(blk_means))
    long_std_ratio = float(np.max(blk_stds) / max(np.min(blk_stds), 1e-15))
    # Expected 10-block mean spread under pure white noise: ~3.08 * sigma / sqrt(blk_len)
    white_expected_se = (dy_std / math.sqrt(2.0)) / math.sqrt(blk_len)
    long_nonstat = long_spread > 5.0 * white_expected_se

    # 5. White vs. colored noise: Ljung-Box on raw high-rate samples & 1-second block averages
    # 1 s averaging suppresses white noise by sqrt(fs) ~ 14.1x so slow AR(1)/RW emerges clearly
    samples_per_sec = max(1, int(round(fs)))
    n_sec = n_s // samples_per_sec
    y_1s = ys[: n_sec * samples_per_sec].reshape(n_sec, samples_per_sec).mean(axis=1)
    q_1s, p_1s, acf_1s = ljung_box_test(y_1s, lags=20)
    q_raw, p_raw, _ = ljung_box_test(ys[: min(n_s, 20000)], lags=20)

    return AssumptionTestReport(
        max_abs_static=max_s,
        max_abs_dynamic=max_d,
        fsr_limit=fsr_limit,
        saturation_free=sat_free,
        static_mean=mean_s,
        static_mean_se_naive=se_naive,
        static_mean_se_block=se_block,
        zero_mean_t_stat=t_stat,
        zero_mean_p_value=p_zero,
        has_nonzero_bias=(p_zero < 0.05),
        diff_std=dy_std,
        skewness=skew,
        excess_kurtosis=ex_kurt,
        jarque_bera_stat=jb_stat,
        jarque_bera_p_value=jb_p,
        approx_gaussian=(abs(skew) < 0.1 and abs(ex_kurt) < 0.2),
        short_window_means=short_means,
        short_window_stds=short_stds,
        short_mean_spread=short_spread,
        short_std_ratio=short_std_ratio,
        long_window_means=blk_means,
        long_window_stds=blk_stds,
        long_mean_spread=long_spread,
        long_std_ratio=long_std_ratio,
        long_horizon_nonstationary=long_nonstat,
        block_1s_ljung_box_q=q_1s,
        block_1s_ljung_box_p=p_1s,
        block_1s_acf=acf_1s,
        raw_ljung_box_q=q_raw,
        raw_ljung_box_p=p_raw,
        has_colored_noise=(p_1s < 0.01),
    )


@dataclass(frozen=True)
class IdentifiedNoiseModel:
    """Black-box identified noise parameters with uncertainty estimates."""

    sample_rate_hz: float
    # Static calibration (effective bias)
    bias_hat: float
    bias_se_naive: float
    bias_se_block: float
    # White measurement noise (first-difference estimator + PSD cross-check)
    sigma_v_hat: float
    sigma_v_ci95: tuple[float, float]
    R_hat: float
    N_diff_hat: float
    N_psd_hat: float
    # Allan deviation curve & weighted nonlinear LS fit (N, B, T_B, K)
    tau: np.ndarray
    adev_emp: np.ndarray
    adev_se: np.ndarray
    adev_fit: np.ndarray
    adev_fit_components: dict[str, np.ndarray]
    N_av_hat: float
    N_av_se: float
    B_av_hat: float
    B_av_se: float
    TB_av_hat: float
    TB_av_se: float
    K_av_hat: float
    K_av_se: float
    # Discrete AR(1) state-space parameters derived solely from identified values
    phi_b_hat: float
    q_b_hat: float
    q_k_hat: float
    sigma_gm_hat: float
    # Welch PSD of stationary noise (used directly by Wiener filter)
    psd_freqs: np.ndarray
    psd_noise: np.ndarray
    # Noise autocovariance (used by FIR Wiener-Hopf)
    r_vv: np.ndarray


def _fit_allan_variance_lm(
    tau: np.ndarray,
    avar_emp: np.ndarray,
    duration_s: float,
    n_init: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Levenberg-Marquardt fit of log(AVAR) for theta = log([N, B, T_B, K]).

    Fits in log-parameter space so positivity is guaranteed, and returns
    (params=[N, B, T_B, K], se=[SE_N, SE_B, SE_TB, SE_K]) via delta method.
    """
    # Relative variance of empirical log(AVAR): Var(log AVAR) ~ 2 / (T/tau - 1)
    n_indep = np.maximum(duration_s / tau - 1.0, 1.5)
    weights = 1.0 / (2.0 / n_indep)
    # Keep well-averaged clusters (at least 4 independent clusters)
    mask = (duration_s / tau) >= 4.0
    t_fit = tau[mask]
    y_fit = np.log(avar_emp[mask])
    w_fit = weights[mask]
    sqrt_w = np.sqrt(w_fit)

    def model_log_avar(log_p: np.ndarray) -> np.ndarray:
        n_p, b_p, tb_p, k_p = np.exp(log_p)
        theo = theoretical_allan_variance(
            t_fit,
            noise_density=n_p,
            bias_instability=b_p,
            bias_correlation_time_s=tb_p,
            random_walk=k_p,
        )
        return np.log(np.maximum(theo["total"], 1e-30))

    # Multi-start initialization from empirical curve features (completely data-driven!)
    adev_fit_emp = np.sqrt(avar_emp[mask])
    idx_min = int(np.argmin(adev_fit_emp))
    b_guess = max(float(adev_fit_emp[idx_min]) / BIAS_INSTABILITY_AV_FACTOR, 1e-6)
    tb_guess_base = max(float(t_fit[idx_min]) / 1.89, 5.0)
    k_guess = max(float(adev_fit_emp[-1]) * math.sqrt(3.0 / float(t_fit[-1])) * 0.5, 1e-7)

    best_cost = float("inf")
    best_p = np.log([n_init, b_guess, tb_guess_base, k_guess])

    # Bounds in log space to keep parameters physically sensible
    lb = np.log([n_init * 0.2, 1e-7, 1.0, 1e-8])
    ub = np.log([n_init * 5.0, 1.0, duration_s * 2.0, 1.0])

    starts = [
        [n_init, b_guess, tb_guess_base, k_guess],
        [n_init, b_guess * 0.7, max(20.0, tb_guess_base * 0.5), k_guess * 1.5],
        [n_init, b_guess * 1.2, min(duration_s, tb_guess_base * 2.0), k_guess * 0.5],
    ]

    for p0_lin in starts:
        p = np.clip(np.log(p0_lin), lb, ub)
        lam = 1e-2
        r = sqrt_w * (model_log_avar(p) - y_fit)
        cost = float(np.dot(r, r))
        for _ in range(80):
            # Finite-difference Jacobian
            jac = np.empty((t_fit.size, 4), dtype=float)
            eps = 1e-5
            f0 = model_log_avar(p)
            for j in range(4):
                dp = np.zeros(4)
                dp[j] = eps
                jac[:, j] = sqrt_w * (model_log_avar(p + dp) - f0) / eps
            jtj = jac.T @ jac
            g = jac.T @ r
            diag = np.maximum(np.diag(jtj), 1e-8)
            try:
                step = -np.linalg.solve(jtj + lam * np.diag(diag), g)
            except np.linalg.LinAlgError:
                break
            p_new = np.clip(p + step, lb, ub)
            r_new = sqrt_w * (model_log_avar(p_new) - y_fit)
            cost_new = float(np.dot(r_new, r_new))
            if cost_new < cost:
                p = p_new
                r = r_new
                cost = cost_new
                lam = max(lam * 0.3, 1e-7)
                if np.linalg.norm(step) < 1e-6:
                    break
            else:
                lam = min(lam * 5.0, 1e6)
        if cost < best_cost:
            best_cost = cost
            best_p = p

    # Compute covariance in log-parameter space from final Jacobian
    jac = np.empty((t_fit.size, 4), dtype=float)
    eps = 1e-5
    f0 = model_log_avar(best_p)
    r = sqrt_w * (f0 - y_fit)
    for j in range(4):
        dp = np.zeros(4)
        dp[j] = eps
        jac[:, j] = sqrt_w * (model_log_avar(best_p + dp) - f0) / eps

    dof = max(1, t_fit.size - 4)
    s2 = float(np.dot(r, r)) / dof
    jtj = jac.T @ jac
    cov_log = np.linalg.pinv(jtj + 1e-9 * np.eye(4)) * max(s2, 1.0)
    params = np.exp(best_p)
    # Delta method: SE(exp(u)) = exp(u) * SE(u)
    se_params = params * np.sqrt(np.maximum(np.diag(cov_log), 0.0))
    return params, se_params


def identify_noise_from_static_data(
    y_static: np.ndarray,
    fs: float,
    *,
    calibration_window_s: float = 10.0,
    wiener_max_lag: int = 128,
    welch_nperseg: int = 2048,
) -> IdentifiedNoiseModel:
    """Identify full stochastic noise model purely from static sensor measurements."""
    ys = np.asarray(y_static, dtype=float)
    n = ys.size
    dt = 1.0 / fs
    duration_s = n * dt

    # 1. Static bias calibration (from initial bench calibration window + block SE)
    n_cal = min(n, int(round(calibration_window_s * fs)))
    y_cal_seg = ys[:n_cal]
    bias_hat = float(np.mean(y_cal_seg))
    bias_se_naive = float(np.std(y_cal_seg, ddof=1) / math.sqrt(n_cal))
    n_blks = 10
    blk_len = n // n_blks
    blk_means = np.array([np.mean(ys[i * blk_len : (i + 1) * blk_len]) for i in range(n_blks)])
    bias_se_block = float(np.std(blk_means, ddof=1) / math.sqrt(n_blks))

    # Zero-centered noise record (using full static mean for stationary spectral analysis)
    z0 = ys - float(np.mean(ys))

    # 2. White noise variance via first-difference estimator (immune to slow bias/drift)
    dy = np.diff(ys)
    var_dy = float(np.var(dy, ddof=1))
    r_hat = 0.5 * var_dy
    sigma_v_hat = math.sqrt(r_hat)
    # 95% CI via block bootstrap (20 blocks of first differences -> accounts for lag-1 corr of dy)
    n_b_dy = 20
    b_dy_len = dy.size // n_b_dy
    blk_sigma_v = np.array(
        [math.sqrt(0.5 * np.var(dy[i * b_dy_len : (i + 1) * b_dy_len], ddof=1)) for i in range(n_b_dy)]
    )
    se_sigma_v = float(np.std(blk_sigma_v, ddof=1) / math.sqrt(n_b_dy))
    sigma_v_ci95 = (sigma_v_hat - 1.96 * se_sigma_v, sigma_v_hat + 1.96 * se_sigma_v)
    n_diff_hat = sigma_v_hat / math.sqrt(fs)

    # 3. Welch PSD of static noise (also cross-checks N from high-frequency flat floor)
    freqs, psd_noise = welch_psd(z0, fs=fs, nperseg=welch_nperseg)
    hf_mask = (freqs >= 0.2 * (fs / 2.0)) & (freqs <= 0.9 * (fs / 2.0))
    psd_hf_median = float(np.median(psd_noise[hf_mask]))
    # Single-sided PSD = 2 * N^2 => N = sqrt(PSD_1s / 2)
    n_psd_hat = math.sqrt(max(psd_hf_median, 0.0) / 2.0)

    # 4. Overlapping Allan Deviation + IEEE error bars + weighted LM fit
    tau, avar_emp = overlapping_allan_variance(ys, fs, max_clusters=70)
    adev_emp = np.sqrt(avar_emp)
    adev_se = adev_emp / np.sqrt(2.0 * np.maximum(duration_s / tau - 1.0, 1.0))

    params_av, se_av = _fit_allan_variance_lm(tau, avar_emp, duration_s=duration_s, n_init=n_diff_hat)
    n_av, b_av, tb_av, k_av = (float(params_av[0]), float(params_av[1]), float(params_av[2]), float(params_av[3]))
    n_se, b_se, tb_se, k_se = (float(se_av[0]), float(se_av[1]), float(se_av[2]), float(se_av[3]))

    theo_fit = theoretical_allan_variance(
        tau,
        noise_density=n_av,
        bias_instability=b_av,
        bias_correlation_time_s=tb_av,
        random_walk=k_av,
    )
    adev_fit = np.sqrt(theo_fit["total"])

    # 5. Discrete-time AR(1) bias/drift parameters from identified (B, T_B, K)
    phi_b = math.exp(-dt / tb_av)
    s_b = bias_instability_to_gm_psd(b_av, tb_av)
    sigma_gm = math.sqrt(0.5 * s_b * tb_av)
    q_b = 0.5 * s_b * tb_av * (1.0 - math.exp(-2.0 * dt / tb_av))
    q_k = (k_av ** 2) * dt

    # 6. Noise autocovariance on short lags (for FIR Wiener-Hopf)
    # Use a representative stationary segment length matching typical dynamic runs (e.g. 20 s blocks averaged)
    # so r_vv captures both white noise spike at lag 0 and stationary short-lag noise floor cleanly.
    r_vv = sample_autocovariance(z0, max_lag=wiener_max_lag)

    return IdentifiedNoiseModel(
        sample_rate_hz=fs,
        bias_hat=bias_hat,
        bias_se_naive=bias_se_naive,
        bias_se_block=bias_se_block,
        sigma_v_hat=sigma_v_hat,
        sigma_v_ci95=sigma_v_ci95,
        R_hat=r_hat,
        N_diff_hat=n_diff_hat,
        N_psd_hat=n_psd_hat,
        tau=tau,
        adev_emp=adev_emp,
        adev_se=adev_se,
        adev_fit=adev_fit,
        adev_fit_components=theo_fit,
        N_av_hat=n_av,
        N_av_se=n_se,
        B_av_hat=b_av,
        B_av_se=b_se,
        TB_av_hat=tb_av,
        TB_av_se=tb_se,
        K_av_hat=k_av,
        K_av_se=k_se,
        phi_b_hat=phi_b,
        q_b_hat=q_b,
        q_k_hat=q_k,
        sigma_gm_hat=sigma_gm,
        psd_freqs=freqs,
        psd_noise=psd_noise,
        r_vv=r_vv,
    )


# ==============================================================================
# Required Baselines
# ==============================================================================


def moving_average_filter(y: np.ndarray, window: int = 15) -> np.ndarray:
    """Causal Moving Average (Baseline 1). At t < W-1 uses available prefix samples."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    arr = np.asarray(y, dtype=float)
    csum = np.cumsum(arr)
    out = np.empty_like(arr)
    out[:window] = csum[:window] / np.arange(1, min(window, arr.size) + 1)
    if arr.size > window:
        out[window:] = (csum[window:] - csum[:-window]) / window
    return out


def lowpass_first_order_filter(y: np.ndarray, fs: float, cutoff_hz: float = 5.0) -> np.ndarray:
    """Causal 1st-order RC Low-Pass / Exponential Moving Average (Baseline 2)."""
    if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be > 0, got {cutoff_hz}")
    arr = np.asarray(y, dtype=float)
    dt = 1.0 / fs
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for k in range(1, arr.size):
        out[k] = alpha * arr[k] + (1.0 - alpha) * out[k - 1]
    return out


# ==============================================================================
# Method 1: Wiener Filter (Causal FIR Wiener-Hopf + Non-Causal Freq-Domain)
# ==============================================================================


@dataclass(frozen=True)
class FIRWienerResult:
    """Causal FIR Wiener filter weights, filtered estimate, and theoretical J_min."""

    weights: np.ndarray
    estimate: np.ndarray
    j_min_theory: float
    rmse_theory: float
    order: int


def design_and_apply_fir_wiener(
    y_cal: np.ndarray,
    noise_model: IdentifiedNoiseModel,
    *,
    order: int = 31,
    white_noise_only_rvv: bool = False,
) -> FIRWienerResult:
    """Design and apply a causal FIR Wiener filter strictly without ground truth.

    Under signal-noise independence:
        R_yy * w* = r_uy = r_yy - r_vv
    where R_yy is estimated from observed y_cal and r_vv from static noise data.
    """
    arr = np.asarray(y_cal, dtype=float)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    r_yy = sample_autocovariance(arr, max_lag=order - 1)
    if white_noise_only_rvv:
        r_vv = np.zeros(order, dtype=float)
        r_vv[0] = noise_model.R_hat
    else:
        if noise_model.r_vv.size < order:
            raise ValueError(f"noise_model.r_vv length ({noise_model.r_vv.size}) < order ({order})")
        r_vv = noise_model.r_vv[:order].copy()

    # Toeplitz matrix R_yy[i, j] = r_yy[|i - j|]
    idx = np.arange(order)
    R_mat = r_yy[np.abs(idx[:, None] - idx[None, :])]
    # Tiny ridge for numerical safety if signal is nearly constant
    R_mat = R_mat + 1e-12 * np.eye(order)
    r_uy = r_yy - r_vv

    w_opt = np.linalg.solve(R_mat, r_uy)

    # Causal FIR convolution (pad prefix with first sample to avoid startup edge spike)
    padded = np.concatenate([np.full(order - 1, arr[0]), arr])
    est = np.convolve(padded, w_opt, mode="valid")

    r_uu_0 = max(float(r_yy[0] - r_vv[0]), 0.0)
    j_min = max(r_uu_0 - float(np.dot(r_uy, w_opt)), 0.0)

    return FIRWienerResult(
        weights=w_opt,
        estimate=est,
        j_min_theory=j_min,
        rmse_theory=math.sqrt(j_min),
        order=order,
    )


def apply_frequency_wiener(
    y_cal: np.ndarray,
    noise_model: IdentifiedNoiseModel,
    *,
    welch_nperseg: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-causal (zero-phase, batch/offline) frequency-domain Wiener filter.

    H_W(f) = max(S_yy(f) - S_vv(f), 0) / (S_yy(f) + eps)
    Uses Welch-smoothed PSDs interpolated onto the signal's rFFT grid.
    """
    arr = np.asarray(y_cal, dtype=float)
    fs = noise_model.sample_rate_hz
    mean_y = float(np.mean(arr))
    yc = arr - mean_y

    f_y, s_yy = welch_psd(yc, fs=fs, nperseg=min(welch_nperseg, arr.size))
    s_vv_interp = np.interp(f_y, noise_model.psd_freqs, noise_model.psd_noise)
    s_uu_hat = np.maximum(s_yy - s_vv_interp, 0.0)
    h_welch = s_uu_hat / (s_yy + 1e-18)

    fft_freqs = np.fft.rfftfreq(arr.size, d=1.0 / fs)
    h_grid = np.interp(fft_freqs, f_y, h_welch)
    y_fft = np.fft.rfft(yc)
    est = np.fft.irfft(y_fft * h_grid, n=arr.size) + mean_y
    return est, f_y, h_welch


# ==============================================================================
# Method 2: Kalman Filter with Identified AR(1) Bias State & MLE Tuning
# ==============================================================================


@dataclass(frozen=True)
class KalmanFilterResult:
    """Outputs of the 1-axis Kalman filter including posterior uncertainty & innovations."""

    signal_est: np.ndarray          # Posterior signal estimate u_hat[k|k]
    signal_std: np.ndarray          # Posterior std sqrt(P_uu[k|k])
    bias_est: np.ndarray            # Posterior AR(1) bias estimate b_hat[k|k]
    bias_std: np.ndarray            # Posterior std sqrt(P_bb[k|k])
    innovations: np.ndarray         # Innovation nu[k] = y[k] - H x[k|k-1]
    innovation_var: np.ndarray      # Innovation variance S[k]
    nis: np.ndarray                 # Normalized innovation squared nu[k]^2 / S[k]
    log_likelihood: float           # Marginal innovation log-likelihood
    q_u: float                      # Signal process noise intensity used


def _build_kalman_matrices(
    dt: float,
    q_u: float,
    noise_model: IdentifiedNoiseModel,
    *,
    include_bias_state: bool = True,
    signal_damping_tau_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build discrete-time Constant-Acceleration (CA) + AR(1) bias state-space matrices.

    State:
        x = [u, u_dot, u_ddot, b]^T   (if include_bias_state=True)
        x = [u, u_dot, u_ddot]^T      (if include_bias_state=False)

    When signal_damping_tau_s is given, u has weak mean reversion exp(-dt/tau_u)
    so slow DC drift b_k and zero-mean AC motion u_k remain well-separated over long runs.
    """
    alpha_u = 1.0 if signal_damping_tau_s is None else math.exp(-dt / signal_damping_tau_s)
    F_sig = np.array(
        [
            [alpha_u, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # Continuous white jerk noise intensity q_u integrated over dt
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    Q_sig = q_u * np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=float,
    )
    R = float(noise_model.R_hat)

    if not include_bias_state:
        H = np.array([[1.0, 0.0, 0.0]], dtype=float)
        return F_sig, Q_sig, H, R

    F = np.zeros((4, 4), dtype=float)
    F[:3, :3] = F_sig
    F[3, 3] = noise_model.phi_b_hat

    Q = np.zeros((4, 4), dtype=float)
    Q[:3, :3] = Q_sig
    Q[3, 3] = noise_model.q_b_hat + noise_model.q_k_hat

    H = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=float)
    return F, Q, H, R


def run_kalman_filter(
    y_cal: np.ndarray,
    noise_model: IdentifiedNoiseModel,
    *,
    q_u: float,
    include_bias_state: bool = True,
    signal_damping_tau_s: float | None = None,
) -> KalmanFilterResult:
    """Run causal recursive Kalman filter with Joseph-form covariance update."""
    arr = np.asarray(y_cal, dtype=float)
    n = arr.size
    dt = 1.0 / noise_model.sample_rate_hz
    F, Q, H, R = _build_kalman_matrices(
        dt,
        q_u,
        noise_model,
        include_bias_state=include_bias_state,
        signal_damping_tau_s=signal_damping_tau_s,
    )
    dim = F.shape[0]
    I_dim = np.eye(dim)

    # Prior initialization purely from data + identified noise model
    x = np.zeros(dim, dtype=float)
    x[0] = arr[0]
    P = np.diag([max(float(np.var(arr)), R), 10.0, 100.0] + ([noise_model.sigma_gm_hat ** 2 + noise_model.bias_se_block ** 2] if include_bias_state else []))

    u_est = np.empty(n, dtype=float)
    u_std = np.empty(n, dtype=float)
    b_est = np.zeros(n, dtype=float)
    b_std = np.zeros(n, dtype=float)
    innov = np.empty(n, dtype=float)
    s_var = np.empty(n, dtype=float)
    nis = np.empty(n, dtype=float)

    log_lik = 0.0
    h_vec = H[0]

    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q

        y_pred = float(np.dot(h_vec, x))
        nu = float(arr[k]) - y_pred
        Ph = P @ h_vec
        S = float(np.dot(h_vec, Ph)) + R
        K = Ph / S

        x = x + K * nu
        # Joseph stabilized covariance update (guarantees symmetry & positive definiteness)
        IKH = I_dim - np.outer(K, h_vec)
        P = IKH @ P @ IKH.T + R * np.outer(K, K)

        u_est[k] = x[0]
        u_std[k] = math.sqrt(max(P[0, 0], 0.0))
        if include_bias_state:
            b_est[k] = x[3]
            b_std[k] = math.sqrt(max(P[3, 3], 0.0))
        innov[k] = nu
        s_var[k] = S
        nis_k = (nu * nu) / S
        nis[k] = nis_k
        if k >= 5:
            log_lik += -0.5 * (math.log(2.0 * math.pi * S) + nis_k)

    return KalmanFilterResult(
        signal_est=u_est,
        signal_std=u_std,
        bias_est=b_est,
        bias_std=b_std,
        innovations=innov,
        innovation_var=s_var,
        nis=nis,
        log_likelihood=log_lik,
        q_u=q_u,
    )


def tune_kalman_qu_mle(
    y_cal: np.ndarray,
    noise_model: IdentifiedNoiseModel,
    *,
    include_bias_state: bool = True,
    signal_damping_tau_s: float | None = None,
    log10_bounds: tuple[float, float] = (-2.0, 9.0),
) -> tuple[float, float, KalmanFilterResult]:
    """Tune Kalman signal process noise q_u by maximizing innovation log-likelihood.

    100% black-box: uses ONLY observed y_cal (zero ground truth).
    Returns (q_u_mle, log10_qu_se, best_kalman_result).
    """
    # Coarse log-grid scan followed by golden-section refinement
    grid = np.linspace(log10_bounds[0], log10_bounds[1], 23)
    best_ll = -float("inf")
    best_idx = 0
    for idx, lg in enumerate(grid):
        res = run_kalman_filter(
            y_cal,
            noise_model,
            q_u=10.0 ** lg,
            include_bias_state=include_bias_state,
            signal_damping_tau_s=signal_damping_tau_s,
        )
        if res.log_likelihood > best_ll:
            best_ll = res.log_likelihood
            best_idx = idx

    lo = float(grid[max(0, best_idx - 1)])
    hi = float(grid[min(grid.size - 1, best_idx + 1)])

    # Golden-section search in log10(q_u)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc = run_kalman_filter(
        y_cal,
        noise_model,
        q_u=10.0 ** c,
        include_bias_state=include_bias_state,
        signal_damping_tau_s=signal_damping_tau_s,
    ).log_likelihood
    fd = run_kalman_filter(
        y_cal,
        noise_model,
        q_u=10.0 ** d,
        include_bias_state=include_bias_state,
        signal_damping_tau_s=signal_damping_tau_s,
    ).log_likelihood

    for _ in range(18):
        if fc < fd:
            lo = c
            c = d
            fc = fd
            d = lo + inv_phi * (hi - lo)
            fd = run_kalman_filter(
                y_cal,
                noise_model,
                q_u=10.0 ** d,
                include_bias_state=include_bias_state,
                signal_damping_tau_s=signal_damping_tau_s,
            ).log_likelihood
        else:
            hi = d
            d = c
            fd = fc
            c = hi - inv_phi * (hi - lo)
            fc = run_kalman_filter(
                y_cal,
                noise_model,
                q_u=10.0 ** c,
                include_bias_state=include_bias_state,
                signal_damping_tau_s=signal_damping_tau_s,
            ).log_likelihood

    lg_opt = 0.5 * (lo + hi)
    q_opt = 10.0 ** lg_opt
    best_res = run_kalman_filter(
        y_cal,
        noise_model,
        q_u=q_opt,
        include_bias_state=include_bias_state,
        signal_damping_tau_s=signal_damping_tau_s,
    )

    # Approximate standard error of log10(q_u) via observed Fisher information
    h = 0.05
    ll_p = run_kalman_filter(
        y_cal,
        noise_model,
        q_u=10.0 ** (lg_opt + h),
        include_bias_state=include_bias_state,
        signal_damping_tau_s=signal_damping_tau_s,
    ).log_likelihood
    ll_m = run_kalman_filter(
        y_cal,
        noise_model,
        q_u=10.0 ** (lg_opt - h),
        include_bias_state=include_bias_state,
        signal_damping_tau_s=signal_damping_tau_s,
    ).log_likelihood
    d2 = (ll_p - 2.0 * best_res.log_likelihood + ll_m) / (h * h)
    lg_se = 1.0 / math.sqrt(-d2) if d2 < -1e-9 else float("nan")
    return q_opt, lg_se, best_res


# ==============================================================================
# Evaluation Metrics (Without Ground Truth & With Ground Truth)
# ==============================================================================


@dataclass(frozen=True)
class InnovationDiagnostics:
    """No-ground-truth consistency diagnostics for Kalman filter innovations."""

    mean_innovation: float
    std_innovation: float
    nis_mean: float
    nis_ci95: tuple[float, float]
    nis_consistent: bool
    ljung_box_q: float
    ljung_box_p: float
    acf: np.ndarray
    acf_within_95_pct: float


def compute_innovation_diagnostics(
    kf_result: KalmanFilterResult,
    *,
    burn_in: int = 20,
    lags: int = 20,
) -> InnovationDiagnostics:
    """Evaluate Kalman consistency strictly without ground truth."""
    nu = kf_result.innovations[burn_in:]
    nis_vals = kf_result.nis[burn_in:]
    n = nu.size
    nis_mean = float(np.mean(nis_vals))
    nis_lo, nis_hi = chi2_mean_ci95(n)
    q_lb, p_lb, acf = ljung_box_test(nu, lags=lags)
    bound = 1.96 / math.sqrt(n)
    within_pct = float(np.mean(np.abs(acf[1:]) <= bound) * 100.0)
    return InnovationDiagnostics(
        mean_innovation=float(np.mean(nu)),
        std_innovation=float(np.std(nu, ddof=1)),
        nis_mean=nis_mean,
        nis_ci95=(nis_lo, nis_hi),
        nis_consistent=(nis_lo <= nis_mean <= nis_hi),
        ljung_box_q=q_lb,
        ljung_box_p=p_lb,
        acf=acf,
        acf_within_95_pct=within_pct,
    )


@dataclass(frozen=True)
class FilterMetrics:
    """Quantitative filter performance metrics against ground truth."""

    rmse: float
    mae: float
    bias_mean: float
    error_std: float
    snr_in_db: float
    snr_out_db: float
    snr_gain_db: float
    lag_samples: int
    lag_ms: float
    coverage_95_pct: float | None = None


def compute_filter_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    raw_input: np.ndarray,
    fs: float,
    *,
    burn_in: int = 40,
    est_std: np.ndarray | None = None,
) -> FilterMetrics:
    """Compute RMSE, MAE, bias, SNR gain, lag, and optional 95% coverage."""
    est = np.asarray(estimate, dtype=float)[burn_in:]
    tru = np.asarray(truth, dtype=float)[burn_in:]
    raw = np.asarray(raw_input, dtype=float)[burn_in:]

    err = est - tru
    raw_err = raw - tru

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias_mean = float(np.mean(err))
    err_std = float(np.std(err, ddof=1))

    sig_power = float(np.mean(tru ** 2))
    raw_mse = float(np.mean(raw_err ** 2))
    est_mse = float(np.mean(err ** 2))
    if sig_power > 1e-12:
        snr_in = 10.0 * math.log10(sig_power / max(raw_mse, 1e-18))
        snr_out = 10.0 * math.log10(sig_power / max(est_mse, 1e-18))
        snr_gain = snr_out - snr_in
    else:
        # Static zero-signal scenario: report noise power reduction in dB
        snr_in = 0.0
        snr_out = 0.0
        snr_gain = 10.0 * math.log10(max(raw_mse, 1e-18) / max(est_mse, 1e-18))

    # Cross-correlation lag (positive = estimate lags behind truth)
    lag_samples = 0
    if float(np.std(tru)) > 1e-9 and float(np.std(est)) > 1e-9:
        max_shift = min(40, est.size // 4)
        t_c = tru - np.mean(tru)
        e_c = est - np.mean(est)
        best_corr = -float("inf")
        for shift in range(-max_shift, max_shift + 1):
            if shift >= 0:
                c_val = float(np.dot(t_c[: est.size - shift], e_c[shift:]))
            else:
                c_val = float(np.dot(t_c[-shift:], e_c[: est.size + shift]))
            if c_val > best_corr:
                best_corr = c_val
                lag_samples = shift

    coverage = None
    if est_std is not None:
        sd = np.asarray(est_std, dtype=float)[burn_in:]
        coverage = float(np.mean(np.abs(err) <= 1.96 * sd) * 100.0)

    return FilterMetrics(
        rmse=rmse,
        mae=mae,
        bias_mean=bias_mean,
        error_std=err_std,
        snr_in_db=snr_in,
        snr_out_db=snr_out,
        snr_gain_db=snr_gain,
        lag_samples=lag_samples,
        lag_ms=lag_samples * (1000.0 / fs),
        coverage_95_pct=coverage,
    )
