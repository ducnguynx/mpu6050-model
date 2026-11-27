"""Black-box Roll-angle (tilt) estimation for MPU-6050 using Sensor Fusion (Pair 2).

Implements Pair 2 (Wiener vs. Kalman on 1-axis Roll angle estimation around X-axis)
strictly treating the MPU-6050 sensor model as unknown (black-box):
    - Dual-sensor statistical assumption checks & black-box noise identification
      (Accelerometer tilt angle noise PSD & R_theta; Gyroscope rate noise PSD,
      Allan Deviation weighted LM fit -> AR(1) bias parameters with uncertainty)
    - Small-angle linearization analysis: theta_lin = a_y / g0 vs. atan2(a_y, a_z)
    - Required Baselines:
        * Baseline 0a: Raw Accelerometer Angle (atan2)
        * Baseline 0b: Raw Gyroscope Integration
        * Baseline 1 : Fixed-gain Complementary Filter (alpha = 0.96)
    - Method 1: Optimal Wiener Complementary Filter
        * Non-causal Frequency-Domain Optimal Wiener Complementary Filter built
          directly from empirical static PSDs S_na(f) and S_nw(f), using the exact
          ramp-free form Theta_hat(f) = G_W(f) Z_a(f) + H_gyr(f) Omega_x(f)
        * Causal 1st-order Spectral-Factorized Wiener Complementary Filter with
          data-derived optimal cutoff omega_c* = sqrt(S_omega / S_theta_a)
    - Method 2: Recursive 2-State Kalman Filter x = [theta, b_g]^T fusing Gyro +
      Accel with black-box identified AR(1) gyro bias model & Joseph covariance update

Pure NumPy + Python stdlib (no SciPy dependency required).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .configuration import STANDARD_GRAVITY
from .filtering import (
    AssumptionTestReport,
    FilterMetrics,
    IdentifiedNoiseModel,
    InnovationDiagnostics,
    chi2_mean_ci95,
    compute_filter_metrics,
    identify_noise_from_static_data,
    ljung_box_test,
    run_assumption_tests,
    welch_psd,
)


# ==============================================================================
# Angle Measurement & Linearization Helpers
# ==============================================================================


def accel_to_roll_atan2(ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """Exact trigonometric Roll angle [rad] from accelerometer Y and Z axes."""
    return np.arctan2(ay, az)


def accel_to_roll_linearized(ay: np.ndarray, g0: float = STANDARD_GRAVITY) -> np.ndarray:
    """Small-angle linearized Roll angle [rad] around operating point theta_0 = 0: sin(theta) ~ theta."""
    return np.asarray(ay, dtype=float) / g0


def check_linearization_accuracy(theta_true_rad: np.ndarray) -> dict[str, float]:
    """Quantify small-angle linearization error |theta - sin(theta)| over a trajectory."""
    th = np.asarray(theta_true_rad, dtype=float)
    lin_err_rad = np.abs(th - np.sin(th))
    max_angle_rad = float(np.max(np.abs(th)))
    max_err_rad = float(np.max(lin_err_rad))
    rms_err_rad = float(np.sqrt(np.mean(lin_err_rad ** 2)))
    rel_err_pct = (max_err_rad / max_angle_rad * 100.0) if max_angle_rad > 1e-12 else 0.0
    return {
        "max_angle_deg": math.degrees(max_angle_rad),
        "max_lin_error_deg": math.degrees(max_err_rad),
        "rms_lin_error_deg": math.degrees(rms_err_rad),
        "max_rel_error_pct": rel_err_pct,
    }


# ==============================================================================
# Black-Box Dual-Sensor Noise Identification (Static Bench Data)
# ==============================================================================


@dataclass(frozen=True)
class AttitudeNoiseModel:
    """Black-box identified noise models for both Accel Roll-angle and Gyro X-rate."""

    sample_rate_hz: float
    # Static flat-bench calibration offsets for accelerometer Y and Z axes [m/s^2]
    ay_bias_hat_mps2: float
    az_bias_hat_mps2: float
    # Accelerometer Roll-angle channel (theta_acc = atan2(a_y_cal, a_z_cal)) [rad]
    accel_angle_model: IdentifiedNoiseModel
    accel_angle_assumptions: AssumptionTestReport
    # Gyroscope X-axis angular velocity channel (omega_x) [rad/s]
    gyro_rate_model: IdentifiedNoiseModel
    gyro_rate_assumptions: AssumptionTestReport
    # Optimal crossover frequency from white-noise spectral factorization [rad/s]
    wiener_causal_omega_c_radps: float
    wiener_causal_fc_hz: float
    # Empirical frequency-domain Wiener complementary transfer functions
    wiener_freqs_hz: np.ndarray
    wiener_G_acc: np.ndarray         # Low-pass weight on accel angle Z_a(f)
    wiener_H_gyr_mag: np.ndarray     # |H_gyr(f)| weight on gyro rate Omega_x(f)
    wiener_theo_rmse_rad: float      # Theoretical minimum angle RMSE from static PSDs


def identify_attitude_noise_from_static(
    ay_static: np.ndarray,
    az_static: np.ndarray,
    gx_static: np.ndarray,
    ay_dyn_sample: np.ndarray,
    az_dyn_sample: np.ndarray,
    gx_dyn_sample: np.ndarray,
    fs: float,
    *,
    accel_fsr_mps2: float,
    gyro_fsr_radps: float,
    calibration_window_s: float = 10.0,
    welch_nperseg: int = 8192,
) -> AttitudeNoiseModel:
    """Identify full stochastic models for Accel Roll-angle & Gyro X-rate purely from static data."""
    # Level-bench 2-axis accelerometer static offset calibration (known static gravity [0, 0, g0])
    n_cal = min(ay_static.size, int(round(calibration_window_s * fs)))
    ay_bias_hat = float(np.mean(ay_static[:n_cal]))
    az_bias_hat = float(np.mean(az_static[:n_cal]) - STANDARD_GRAVITY)

    theta_a_static = accel_to_roll_atan2(ay_static - ay_bias_hat, az_static - az_bias_hat)
    theta_a_dyn = accel_to_roll_atan2(ay_dyn_sample - ay_bias_hat, az_dyn_sample - az_bias_hat)

    # Assumption tests on both channels (angle FSR limit pi/2 for roll check, gyro FSR from datasheet)
    acc_assump = run_assumption_tests(
        accel_to_roll_atan2(ay_static, az_static),
        theta_a_dyn,
        fs=fs,
        fsr_limit=math.pi / 2.0,
    )
    gyr_assump = run_assumption_tests(
        gx_static,
        gx_dyn_sample,
        fs=fs,
        fsr_limit=gyro_fsr_radps,
    )

    # Full black-box stochastic identification on both channels
    acc_model = identify_noise_from_static_data(
        theta_a_static,
        fs=fs,
        calibration_window_s=calibration_window_s,
        welch_nperseg=welch_nperseg,
    )
    gyr_model = identify_noise_from_static_data(
        gx_static,
        fs=fs,
        calibration_window_s=calibration_window_s,
        welch_nperseg=welch_nperseg,
    )

    # Optimal causal 1st-order Wiener crossover frequency from white-noise floors:
    # omega_c* = sqrt(S_omega / S_theta_a) = N_g / N_theta_a
    omega_c = gyr_model.N_diff_hat / max(acc_model.N_diff_hat, 1e-12)
    fc_hz = omega_c / (2.0 * math.pi)

    # Empirical frequency-domain Wiener complementary transfer functions on Welch grid
    dt = 1.0 / fs
    f_grid = acc_model.psd_freqs
    s_na = acc_model.psd_noise
    s_nw = np.interp(f_grid, gyr_model.psd_freqs, gyr_model.psd_noise)

    # Exact discrete differentiation operator D(f) = (1 - exp(-j 2 pi f dt)) / dt
    # |D(f)|^2 = 4 sin^2(pi f dt) / dt^2
    d_sq = (4.0 * np.sin(math.pi * f_grid * dt) ** 2) / (dt * dt)
    denom = s_nw + d_sq * s_na + 1e-30
    g_w = s_nw / denom
    g_w[0] = 1.0  # Exact DC: trust accelerometer 100% -> completely blocks gyro bias drift

    # |H_gyr(f)| = |1 - G_W(f)| / |D(f)| = |D(f)| * S_na(f) / (S_nw(f) + |D(f)|^2 S_na(f))
    h_gyr_mag = np.sqrt(d_sq) * s_na / denom
    h_gyr_mag[0] = 0.0

    # Theoretical minimum angle MSE predicted by Wiener complementary theory:
    # MSE_min = integral_0^{fs/2} S_e(f) df, where S_e(f) = G_W(f) * S_na(f)
    s_err = g_w * s_na
    df = float(f_grid[1] - f_grid[0]) if f_grid.size > 1 else 0.0
    mse_theo = float(np.sum(s_err[1:]) * df)
    rmse_theo = math.sqrt(max(mse_theo, 0.0))

    return AttitudeNoiseModel(
        sample_rate_hz=fs,
        ay_bias_hat_mps2=ay_bias_hat,
        az_bias_hat_mps2=az_bias_hat,
        accel_angle_model=acc_model,
        accel_angle_assumptions=acc_assump,
        gyro_rate_model=gyr_model,
        gyro_rate_assumptions=gyr_assump,
        wiener_causal_omega_c_radps=omega_c,
        wiener_causal_fc_hz=fc_hz,
        wiener_freqs_hz=f_grid,
        wiener_G_acc=g_w,
        wiener_H_gyr_mag=h_gyr_mag,
        wiener_theo_rmse_rad=rmse_theo,
    )


# ==============================================================================
# Required Baselines
# ==============================================================================


def baseline_gyro_integration(
    gx: np.ndarray,
    fs: float,
    *,
    theta0: float = 0.0,
    bias_offset: float = 0.0,
) -> np.ndarray:
    """Baseline 0b: Pure discrete integration of gyroscope rate."""
    dt = 1.0 / fs
    w = np.asarray(gx, dtype=float) - bias_offset
    out = np.empty_like(w)
    out[0] = theta0
    for k in range(1, w.size):
        out[k] = out[k - 1] + 0.5 * (w[k - 1] + w[k]) * dt
    return out


def baseline_complementary_filter(
    theta_acc_cal: np.ndarray,
    gx_cal: np.ndarray,
    fs: float,
    *,
    alpha: float = 0.96,
) -> np.ndarray:
    """Baseline 1: Classic fixed-gain Complementary Filter.

    theta_hat[k] = alpha * (theta_hat[k-1] + gx_cal[k] * dt) + (1 - alpha) * theta_acc_cal[k]
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    dt = 1.0 / fs
    za = np.asarray(theta_acc_cal, dtype=float)
    w = np.asarray(gx_cal, dtype=float)
    out = np.empty_like(za)
    out[0] = za[0]
    for k in range(1, za.size):
        out[k] = alpha * (out[k - 1] + w[k] * dt) + (1.0 - alpha) * za[k]
    return out


# ==============================================================================
# Method 1: Optimal Wiener Complementary Filter (Frequency-Domain + Causal)
# ==============================================================================


@dataclass(frozen=True)
class WienerAttitudeResult:
    """Outputs of the Optimal Wiener Complementary Filter."""

    theta_est_rad: np.ndarray
    G_acc_grid: np.ndarray
    H_gyr_grid: np.ndarray
    freqs_hz: np.ndarray
    theo_rmse_rad: float


def apply_wiener_complementary_freq(
    theta_acc_cal: np.ndarray,
    gx_cal: np.ndarray,
    noise_model: AttitudeNoiseModel,
    *,
    use_parametric_psd: bool = False,
) -> WienerAttitudeResult:
    """Non-causal Optimal Wiener Complementary Filter directly from static PSDs.

    Uses the exact ramp-free frequency-domain complementary formulation:
        Theta_hat(f) = G_W(f) * Z_a(f) + H_gyr(f) * Omega_x(f)
    where:
        D(f)     = (1 - exp(-j 2 pi f dt)) / dt   (exact discrete derivative)
        G_W(f)   = S_nw(f) / (S_nw(f) + |D(f)|^2 * S_na(f))
        H_gyr(f) = (1 - G_W(f)) / D(f) = D*(f) * S_na(f) / (S_nw(f) + |D(f)|^2 * S_na(f))

    Key properties:
        - G_W(0) = 1 and H_gyr(0) = 0 -> zero distortion on theta(t) & complete
          DC gyro bias rejection without ever integrating a linear drift ramp!
        - Symmetric (even) reflection on theta_acc and antisymmetric (odd)
          reflection on omega_x eliminate FFT periodic boundary artifacts.
    """
    za = np.asarray(theta_acc_cal, dtype=float)
    w = np.asarray(gx_cal, dtype=float)
    n = za.size
    fs = noise_model.sample_rate_hz
    dt = 1.0 / fs

    # Reflect boundaries: even for angle theta, odd for its derivative omega = d(theta)/dt
    # Remove mean of gyro before odd reflection so constant residual bias doesn't flip sign
    # into a square wave; instead add residual mean back into the unreflected path? Wait:
    # H_gyr(0) = 0 rejects constant bias completely! If we subtract w_mean = mean(w),
    # note that for zero-mean or periodic angle theta(T) ~ theta(0), mean(omega_true) ~ 0,
    # whereas if theta(T) != theta(0) then mean(omega_true) = (theta(T)-theta(0))/T != 0.
    # Even better: pad with linear ramp-matched even/odd reflection using only edge continuity!
    # Wait: even reflection of za and odd reflection of w preserves d(za_pad)/dt = w_pad
    # everywhere EXCEPT any constant bias b_g becomes a square wave of amplitude b_g.
    # Since static calibration already removes b_g0, remaining bias b_res is tiny (~1e-4 rad/s),
    # AND subtracting the constant bias estimate does NOT change H_gyr(0) = 0.
    pad = min(n, max(1024, n // 2))
    za_left = za[pad - 1 :: -1] if pad < n else za[::-1]
    za_right = za[-1 : -pad - 1 : -1] if pad < n else za[::-1]
    za_pad = np.concatenate([za_left, za, za_right])

    w_left = -w[pad - 1 :: -1] if pad < n else -w[::-1]
    w_right = -w[-1 : -pad - 1 : -1] if pad < n else -w[::-1]
    w_pad = np.concatenate([w_left, w, w_right])

    m = za_pad.size
    freqs = np.fft.rfftfreq(m, d=dt)

    if use_parametric_psd:
        # Smooth analytical PSD reconstructed purely from black-box Allan-fitted parameters
        # S_1s(f) = 2*N^2 + 2*S_B / (1/T_B^2 + (2*pi*f)^2) + 2*K^2 / (2*pi*f)^2
        gm = noise_model.gyro_rate_model
        am = noise_model.accel_angle_model
        w2 = (2.0 * math.pi * freqs) ** 2
        w2_safe = np.maximum(w2, 1e-18)
        sb_g = 2.0 * (gm.sigma_gm_hat ** 2) / max(gm.TB_av_hat, 1.0)
        sb_a = 2.0 * (am.sigma_gm_hat ** 2) / max(am.TB_av_hat, 1.0)
        s_nw = 2.0 * (gm.N_diff_hat ** 2) + 2.0 * sb_g / (1.0 / (gm.TB_av_hat ** 2) + w2) + 2.0 * (gm.K_av_hat ** 2) / w2_safe
        s_na = 2.0 * (am.N_diff_hat ** 2) + 2.0 * sb_a / (1.0 / (am.TB_av_hat ** 2) + w2) + 2.0 * (am.K_av_hat ** 2) / w2_safe
    else:
        # Directly interpolate empirical static Welch PSDs onto FFT grid
        # Extrapolate below lowest Welch bin using identified low-freq colored floor
        s_na = np.interp(freqs, noise_model.accel_angle_model.psd_freqs[1:], noise_model.accel_angle_model.psd_noise[1:])
        s_nw = np.interp(freqs, noise_model.gyro_rate_model.psd_freqs[1:], noise_model.gyro_rate_model.psd_noise[1:])

    # Exact discrete trapezoidal integration/differentiation frequency response:
    # Trapezoidal integrator: H_trap(f) = (dt/2) * (1 + e^{-jwdt}) / (1 - e^{-jwdt})
    # Inverse (trapezoidal differentiator): D(f) = (2/dt) * (1 - e^{-jwdt}) / (1 + e^{-jwdt})
    # Note: for backward Euler D(f) = (1 - e^{-jwdt}) / dt (no singularity at Nyquist!)
    # Use backward Euler D(f) = (1 - exp(-j 2 pi f dt)) / dt which is exact and well-conditioned everywhere
    ejw = np.exp(-1j * 2.0 * math.pi * freqs * dt)
    D_f = (1.0 - ejw) / dt
    d_sq = np.abs(D_f) ** 2

    denom = s_nw + d_sq * s_na + 1e-30
    G_w = s_nw / denom
    G_w[0] = 1.0

    H_gyr = np.conj(D_f) * s_na / denom
    H_gyr[0] = 0.0 + 0.0j

    Za_fft = np.fft.rfft(za_pad)
    W_fft = np.fft.rfft(w_pad)
    theta_pad_est = np.fft.irfft(G_w * Za_fft + H_gyr * W_fft, n=m)
    theta_est = theta_pad_est[pad : pad + n]

    return WienerAttitudeResult(
        theta_est_rad=theta_est,
        G_acc_grid=np.real(G_w),
        H_gyr_grid=H_gyr,
        freqs_hz=freqs,
        theo_rmse_rad=noise_model.wiener_theo_rmse_rad,
    )


def apply_wiener_complementary_causal(
    theta_acc_cal: np.ndarray,
    gx_cal: np.ndarray,
    noise_model: AttitudeNoiseModel,
) -> tuple[np.ndarray, float]:
    """Causal 1st-order Wiener Complementary Filter via spectral factorization.

    G_{W,causal}(s) = omega_c* / (s + omega_c*), where omega_c* = sqrt(S_omega / S_theta_a)
    is estimated purely from static sensor noise densities (zero hand-tuning!).
    Discretized as alpha* = 1 / (1 + omega_c* * dt).
    """
    dt = 1.0 / noise_model.sample_rate_hz
    omega_c = noise_model.wiener_causal_omega_c_radps
    alpha_opt = 1.0 / (1.0 + omega_c * dt)
    est = baseline_complementary_filter(theta_acc_cal, gx_cal, fs=noise_model.sample_rate_hz, alpha=alpha_opt)
    return est, alpha_opt


# ==============================================================================
# Method 2: 2-State Kalman Filter [theta, b_g]^T with AR(1) Gyro Bias Model
# ==============================================================================


@dataclass(frozen=True)
class AttitudeKalmanResult:
    """Outputs of the 2-State Roll-Angle Kalman Filter."""

    theta_est_rad: np.ndarray       # Posterior Roll angle estimate [rad]
    theta_std_rad: np.ndarray       # Posterior std sqrt(P_00) [rad]
    bias_est_radps: np.ndarray      # Posterior gyro bias estimate [rad/s]
    bias_std_radps: np.ndarray      # Posterior std sqrt(P_11) [rad/s]
    innovations_rad: np.ndarray     # Innovation nu[k] = theta_acc[k] - theta_pred[k]
    innovation_var: np.ndarray      # Innovation variance S[k]
    nis: np.ndarray                 # Normalized innovation squared nu[k]^2 / S[k]
    log_likelihood: float           # Marginal innovation log-likelihood
    R_used: float                   # Measurement variance R used


def run_attitude_kalman_2state(
    theta_acc_cal: np.ndarray,
    gx_meas: np.ndarray,
    noise_model: AttitudeNoiseModel,
    *,
    use_calibrated_gyro: bool = False,
    R_scale: float = 1.0,
    q_bias_scale: float = 1.0,
    init_bias_std_radps: float | None = None,
) -> AttitudeKalmanResult:
    """Run causal 2-state Kalman Filter x_k = [theta_k, b_{g,k}]^T.

    State dynamics (total gyro bias modeled as random walk with combined GM+RW diffusion):
        theta_{k} = theta_{k-1} + (omega_x[k] - b_{g,k-1}) * dt + w_{theta,k}
        b_{g,k}   = b_{g,k-1} + w_{b,k}

    Note: Using F[1,1] = 1.0 (rather than phi_B < 1) prevents biased shrinkage of
    non-zero turn-on / thermal bias shifts toward zero while keeping the 2-state
    system strictly full-rank observable (rank([H; H*F]) = 2).

    Measurement:
        y_k = theta_acc_cal[k] = [1, 0] * x_k + v_k,   v_k ~ N(0, R_theta * R_scale)

    All parameters (q_theta, q_b = q_B + q_K, R_theta) come strictly from black-box
    static identification (zero peeking at default.yaml!).
    """
    za = np.asarray(theta_acc_cal, dtype=float)
    gm = noise_model.gyro_rate_model
    am = noise_model.accel_angle_model
    w = np.asarray(gx_meas, dtype=float) - (gm.bias_hat if use_calibrated_gyro else 0.0)

    n = za.size
    dt = 1.0 / noise_model.sample_rate_hz

    F = np.array([[1.0, -dt], [0.0, 1.0]], dtype=float)
    H = np.array([1.0, 0.0], dtype=float)

    # Exact discrete process noise covariance from identified white + AR(1) + RW parameters
    q_theta = (gm.sigma_v_hat ** 2) * (dt * dt)  # = N_g^2 * dt
    q_b = (gm.q_b_hat + gm.q_k_hat) * q_bias_scale
    Q = np.array(
        [
            [q_theta + (q_b * dt * dt) / 3.0, -0.5 * q_b * dt],
            [-0.5 * q_b * dt, q_b],
        ],
        dtype=float,
    )
    R = float(am.R_hat) * R_scale

    # Prior state & covariance from black-box static estimates
    x = np.array([za[0], 0.0 if use_calibrated_gyro else gm.bias_hat], dtype=float)
    if init_bias_std_radps is not None:
        p_b0 = init_bias_std_radps ** 2
    else:
        # Accounts for GM steady-state std, calibration block SE, and modest turn-on drift tolerance
        p_b0 = gm.sigma_gm_hat ** 2 + gm.bias_se_block ** 2 + (0.0 if use_calibrated_gyro else (0.0025 ** 2))
    P = np.array([[R, 0.0], [0.0, p_b0]], dtype=float)

    theta_est = np.empty(n, dtype=float)
    theta_std = np.empty(n, dtype=float)
    bias_est = np.empty(n, dtype=float)
    bias_std = np.empty(n, dtype=float)
    innov = np.empty(n, dtype=float)
    s_var = np.empty(n, dtype=float)
    nis = np.empty(n, dtype=float)

    I2 = np.eye(2)
    log_lik = 0.0

    for k in range(n):
        if k > 0:
            x = F @ x + np.array([w[k] * dt, 0.0], dtype=float)
            P = F @ P @ F.T + Q

        nu = float(za[k]) - float(x[0])
        S = float(P[0, 0]) + R
        K = P[:, 0] / S

        x = x + K * nu
        # Joseph stabilized covariance update
        IKH = I2 - np.outer(K, H)
        P = IKH @ P @ IKH.T + R * np.outer(K, K)

        theta_est[k] = x[0]
        theta_std[k] = math.sqrt(max(P[0, 0], 0.0))
        bias_est[k] = x[1]
        bias_std[k] = math.sqrt(max(P[1, 1], 0.0))
        innov[k] = nu
        s_var[k] = S
        nis_k = (nu * nu) / S
        nis[k] = nis_k
        if k >= 20:
            log_lik += -0.5 * (math.log(2.0 * math.pi * S) + nis_k)

    return AttitudeKalmanResult(
        theta_est_rad=theta_est,
        theta_std_rad=theta_std,
        bias_est_radps=bias_est,
        bias_std_radps=bias_std,
        innovations_rad=innov,
        innovation_var=s_var,
        nis=nis,
        log_likelihood=log_lik,
        R_used=R,
    )


def tune_attitude_kalman_R_mle(
    theta_acc_cal: np.ndarray,
    gx_meas: np.ndarray,
    noise_model: AttitudeNoiseModel,
    *,
    use_calibrated_gyro: bool = False,
) -> tuple[float, AttitudeKalmanResult]:
    """Black-box MLE tuning of accelerometer variance multiplier R_scale (zero ground truth!).

    When the sensor experiences external linear vibration (a_lin != 0), empirical
    innovations detect excess accelerometer variance and automatically inflate R_scale > 1.
    """
    grid = np.linspace(0.0, 4.5, 19)  # R_scale in [1, 31622]
    best_ll = -float("inf")
    best_idx = 0
    for idx, lg in enumerate(grid):
        res = run_attitude_kalman_2state(
            theta_acc_cal,
            gx_meas,
            noise_model,
            use_calibrated_gyro=use_calibrated_gyro,
            R_scale=10.0 ** lg,
        )
        if res.log_likelihood > best_ll:
            best_ll = res.log_likelihood
            best_idx = idx

    lo = float(grid[max(0, best_idx - 1)])
    hi = float(grid[min(grid.size - 1, best_idx + 1)])
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc = run_attitude_kalman_2state(
        theta_acc_cal, gx_meas, noise_model, use_calibrated_gyro=use_calibrated_gyro, R_scale=10.0 ** c
    ).log_likelihood
    fd = run_attitude_kalman_2state(
        theta_acc_cal, gx_meas, noise_model, use_calibrated_gyro=use_calibrated_gyro, R_scale=10.0 ** d
    ).log_likelihood

    for _ in range(16):
        if fc < fd:
            lo = c
            c = d
            fc = fd
            d = lo + inv_phi * (hi - lo)
            fd = run_attitude_kalman_2state(
                theta_acc_cal, gx_meas, noise_model, use_calibrated_gyro=use_calibrated_gyro, R_scale=10.0 ** d
            ).log_likelihood
        else:
            hi = d
            d = c
            fd = fc
            c = hi - inv_phi * (hi - lo)
            fc = run_attitude_kalman_2state(
                theta_acc_cal, gx_meas, noise_model, use_calibrated_gyro=use_calibrated_gyro, R_scale=10.0 ** c
            ).log_likelihood

    r_scale_opt = 10.0 ** (0.5 * (lo + hi))
    best_res = run_attitude_kalman_2state(
        theta_acc_cal, gx_meas, noise_model, use_calibrated_gyro=use_calibrated_gyro, R_scale=r_scale_opt
    )
    return r_scale_opt, best_res


@dataclass(frozen=True)
class AttitudeKalman4StateResult:
    """Outputs of the 4-State Roll-Angle Kalman Filter x_k = [u_k, u_dot_k, u_ddot_k, b_k]^T."""

    theta_est_rad: np.ndarray       # Posterior Roll angle estimate u_hat[k|k] [rad]
    theta_std_rad: np.ndarray       # Posterior std sqrt(P_00) [rad]
    rate_est_radps: np.ndarray      # Posterior angular rate u_dot_hat[k|k] [rad/s]
    accel_est_radps2: np.ndarray    # Posterior angular accel u_ddot_hat[k|k] [rad/s^2]
    bias_est_radps: np.ndarray      # Posterior gyro bias b_hat[k|k] [rad/s]
    bias_std_radps: np.ndarray      # Posterior std sqrt(P_33) [rad/s]
    innovations_rad: np.ndarray     # Accel-angle innovation nu_a[k] [rad]
    innovations_rate: np.ndarray    # Gyro-rate innovation nu_g[k] [rad/s]
    innovation_var_angle: np.ndarray  # S_00[k] [rad^2]
    nis_dof: np.ndarray             # 2D NIS normalized by m=2: (nu^T S^{-1} nu) / 2 (ideal E = 1.0)
    log_likelihood: float           # 2D marginal innovation log-likelihood
    log_likelihood_gyro: float      # Gyro-channel innovation log-likelihood (immune to accel vibration!)
    q_u: float                      # Tuned angular-jerk process noise intensity


def run_attitude_kalman_4state(
    theta_acc_cal: np.ndarray,
    gx_meas: np.ndarray,
    noise_model: AttitudeNoiseModel,
    *,
    q_u: float = 100.0,
    R_angle_scale: float = 1.0,
    q_bias_scale: float = 1.0,
    init_bias_std_radps: float | None = None,
) -> AttitudeKalman4StateResult:
    """Run causal 4-state Kalman Filter x_k = [u_k, u_dot_k, u_ddot_k, b_k]^T.

    State vector:
        u_k      = Roll angle theta_k [rad]
        u_dot_k  = True angular velocity d(theta)/dt [rad/s]
        u_ddot_k = Angular acceleration d^2(theta)/dt^2 [rad/s^2]
        b_k      = Total gyroscope X-axis bias (b_0 + x_B + x_K) [rad/s]

    State transition (Constant Angular Acceleration CA + Random-Walk total bias):
        F = [[1, dt, 0.5*dt^2, 0],
             [0,  1,       dt, 0],
             [0,  0,        1, 0],
             [0,  0,        0, 1]]
        Q = blkdiag(q_u * Q_CA(dt), (q_B + q_K) * q_bias_scale)

    2D Sensor-Fusion Measurement Model:
        y_k = [theta_acc_cal[k], gx_meas[k]]^T = H * x_k + v_k
        H   = [[1, 0, 0, 0],
               [0, 1, 0, 1]]
        R   = diag(R_theta_hat * R_angle_scale, R_omega_hat)

    All sensor noise parameters (R_theta, R_omega, q_B, q_K, initial bias) come
    strictly from black-box static identification (zero peeking at default.yaml!).
    """
    za = np.asarray(theta_acc_cal, dtype=float)
    w = np.asarray(gx_meas, dtype=float)
    gm = noise_model.gyro_rate_model
    am = noise_model.accel_angle_model

    n = za.size
    dt = 1.0 / noise_model.sample_rate_hz
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt

    F = np.array(
        [
            [1.0, dt, 0.5 * dt2, 0.0],
            [0.0, 1.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    q_b = (gm.q_b_hat + gm.q_k_hat) * q_bias_scale
    Q = np.zeros((4, 4), dtype=float)
    Q[:3, :3] = q_u * np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=float,
    )
    Q[3, 3] = q_b

    r_a = float(am.R_hat) * R_angle_scale
    r_g = float(gm.R_hat)
    R = np.array([[r_a, 0.0], [0.0, r_g]], dtype=float)

    # Prior initialization strictly from static bench estimates
    x = np.array([za[0], w[0] - gm.bias_hat, 0.0, gm.bias_hat], dtype=float)
    if init_bias_std_radps is not None:
        p_b0 = init_bias_std_radps ** 2
    else:
        p_b0 = gm.sigma_gm_hat ** 2 + gm.bias_se_block ** 2 + (0.0025 ** 2)
    P = np.diag([r_a, 1.0, 10.0, p_b0]).astype(float)

    theta_est = np.empty(n, dtype=float)
    theta_std = np.empty(n, dtype=float)
    rate_est = np.empty(n, dtype=float)
    accel_est = np.empty(n, dtype=float)
    bias_est = np.empty(n, dtype=float)
    bias_std = np.empty(n, dtype=float)
    innov_a = np.empty(n, dtype=float)
    innov_g = np.empty(n, dtype=float)
    s_aa = np.empty(n, dtype=float)
    nis_dof = np.empty(n, dtype=float)

    I4 = np.eye(4)
    log_lik = 0.0
    log_lik_g = 0.0
    two_pi = 2.0 * math.pi
    two_pi_sq = two_pi * two_pi

    for k in range(n):
        if k > 0:
            x = F @ x
            P = F @ P @ F.T + Q

        # 2D Innovation: nu = [za - u, w - (u_dot + b)]
        nu0 = float(za[k]) - float(x[0])
        nu1 = float(w[k]) - (float(x[1]) + float(x[3]))

        # Closed-form 2x2 S = H P H^T + R
        PHt = P @ H.T  # shape (4, 2)
        s00 = float(PHt[0, 0]) + r_a
        s01 = float(PHt[0, 1])
        s10 = float(PHt[1, 0] + PHt[3, 0])
        s11 = float(PHt[1, 1] + PHt[3, 1]) + r_g
        det_s = s00 * s11 - s01 * s10
        inv_det = 1.0 / max(det_s, 1e-30)
        is00 = s11 * inv_det
        is01 = -s01 * inv_det
        is10 = -s10 * inv_det
        is11 = s00 * inv_det

        # Kalman gain K = P H^T S^{-1}, shape (4, 2)
        K0 = PHt[:, 0] * is00 + PHt[:, 1] * is10
        K1 = PHt[:, 0] * is01 + PHt[:, 1] * is11
        K = np.column_stack((K0, K1))

        x = x + K0 * nu0 + K1 * nu1
        # Joseph stabilized covariance update
        IKH = I4 - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T

        theta_est[k] = x[0]
        theta_std[k] = math.sqrt(max(P[0, 0], 0.0))
        rate_est[k] = x[1]
        accel_est[k] = x[2]
        bias_est[k] = x[3]
        bias_std[k] = math.sqrt(max(P[3, 3], 0.0))
        innov_a[k] = nu0
        innov_g[k] = nu1
        s_aa[k] = s00

        nis_2d = nu0 * (is00 * nu0 + is01 * nu1) + nu1 * (is10 * nu0 + is11 * nu1)
        nis_dof[k] = 0.5 * nis_2d  # Normalized per DOF (ideal E = 1.0)
        if k >= 20:
            log_lik += -0.5 * (math.log(max(two_pi_sq * det_s, 1e-300)) + nis_2d)
            log_lik_g += -0.5 * (math.log(max(two_pi * s11, 1e-300)) + (nu1 * nu1) / max(s11, 1e-30))

    return AttitudeKalman4StateResult(
        theta_est_rad=theta_est,
        theta_std_rad=theta_std,
        rate_est_radps=rate_est,
        accel_est_radps2=accel_est,
        bias_est_radps=bias_est,
        bias_std_radps=bias_std,
        innovations_rad=innov_a,
        innovations_rate=innov_g,
        innovation_var_angle=s_aa,
        nis_dof=nis_dof,
        log_likelihood=log_lik,
        log_likelihood_gyro=log_lik_g,
        q_u=q_u,
    )


def tune_attitude_kalman_4state_qu_mle(
    theta_acc_cal: np.ndarray,
    gx_meas: np.ndarray,
    noise_model: AttitudeNoiseModel,
    *,
    log10_bounds: tuple[float, float] = (-3.0, 6.0),
) -> tuple[float, float, AttitudeKalman4StateResult]:
    """Tune 4-state Kalman angular-jerk q_u via Gyro-channel Innovation MLE (zero ground truth).

    Maximizing the gyroscope-channel marginal log-likelihood fits q_u strictly to
    rotational kinematics and prevents external accelerometer lateral vibration
    from falsely inflating angular-jerk process noise.
    """
    grid = np.linspace(log10_bounds[0], log10_bounds[1], 19)
    best_ll = -float("inf")
    best_idx = 0
    for idx, lg in enumerate(grid):
        res = run_attitude_kalman_4state(theta_acc_cal, gx_meas, noise_model, q_u=10.0 ** lg)
        if res.log_likelihood_gyro > best_ll:
            best_ll = res.log_likelihood_gyro
            best_idx = idx

    lo = float(grid[max(0, best_idx - 1)])
    hi = float(grid[min(grid.size - 1, best_idx + 1)])
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc = run_attitude_kalman_4state(theta_acc_cal, gx_meas, noise_model, q_u=10.0 ** c).log_likelihood_gyro
    fd = run_attitude_kalman_4state(theta_acc_cal, gx_meas, noise_model, q_u=10.0 ** d).log_likelihood_gyro

    for _ in range(16):
        if fc < fd:
            lo = c
            c = d
            fc = fd
            d = lo + inv_phi * (hi - lo)
            fd = run_attitude_kalman_4state(theta_acc_cal, gx_meas, noise_model, q_u=10.0 ** d).log_likelihood_gyro
        else:
            hi = d
            d = c
            fd = fc
            c = hi - inv_phi * (hi - lo)
            fc = run_attitude_kalman_4state(theta_acc_cal, gx_meas, noise_model, q_u=10.0 ** c).log_likelihood_gyro

    lg_opt = 0.5 * (lo + hi)
    q_opt = 10.0 ** lg_opt
    best_res = run_attitude_kalman_4state(theta_acc_cal, gx_meas, noise_model, q_u=q_opt)

    h = 0.05
    ll_p = run_attitude_kalman_4state(theta_acc_cal, gx_meas, noise_model, q_u=10.0 ** (lg_opt + h)).log_likelihood_gyro
    ll_m = run_attitude_kalman_4state(theta_acc_cal, gx_meas, noise_model, q_u=10.0 ** (lg_opt - h)).log_likelihood_gyro
    d2 = (ll_p - 2.0 * best_res.log_likelihood_gyro + ll_m) / (h * h)
    lg_se = 1.0 / math.sqrt(-d2) if d2 < -1e-9 else float("nan")
    return q_opt, lg_se, best_res


def compute_attitude_innovation_diagnostics(
    kf_res: AttitudeKalmanResult | AttitudeKalman4StateResult,
    *,
    burn_in: int = 100,
    lags: int = 20,
) -> InnovationDiagnostics:
    """No-ground-truth consistency diagnostics for the 2-state or 4-state attitude Kalman filter."""
    nu = kf_res.innovations_rad[burn_in:]
    if isinstance(kf_res, AttitudeKalman4StateResult):
        nis_vals = kf_res.nis_dof[burn_in:]
        n_dof = 2 * nu.size
    else:
        nis_vals = kf_res.nis[burn_in:]
        n_dof = nu.size
    n = nu.size
    nis_mean = float(np.mean(nis_vals))
    nis_lo, nis_hi = chi2_mean_ci95(n_dof)
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
