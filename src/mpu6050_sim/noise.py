"""Discrete-time stochastic error processes for MEMS inertial sensors.

Implements the exact continuous-to-discrete state-space stochastic error equations
from Farrell, Silva, Rahman, & Wendel (2022), IEEE Control Systems Magazine,
Vol. 42, No. 6, pp. 40-66 (DOI: 10.1109/MCS.2022.3209059):

    z_i[k] = x_{B,i}[k] + x_{K,i}[k] + v_{N,i}[k]

where for each axis i in {x, y, z}:
    - v_{N,i}[k] ~ N(0, N_i^2 / dt)          (Broadband white noise: ARW / VRW)
    - x_{B,i}[k+1] = Phi_{B,i} x_{B,i}[k] + w_{B,d,i}[k]  (1st-order Gauss-Markov)
    - x_{K,i}[k+1] = x_{K,i}[k] + w_{K,d,i}[k]            (Rate / Accel Random Walk)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

# IEEE Std 952 / El-Sheimy (2008) flicker-floor Allan deviation constant:
# sigma_A,BI = sqrt(2 * ln(2) / pi) * B ~= 0.66428247 * B
BIAS_INSTABILITY_AV_FACTOR: float = math.sqrt(2.0 * math.log(2.0) / math.pi)

# Farrell et al. (2022) first-order Gauss-Markov Allan deviation peak location
# lambda_max = tau_max / T_B = 1.89 and corresponding peak coefficient c_GM ~= 0.436541
GAUSS_MARKOV_PEAK_LAMBDA: float = 1.89
GAUSS_MARKOV_PEAK_FACTOR: float = math.sqrt(
    (1.0 / GAUSS_MARKOV_PEAK_LAMBDA)
    - (3.0 - 4.0 * math.exp(-GAUSS_MARKOV_PEAK_LAMBDA) + math.exp(-2.0 * GAUSS_MARKOV_PEAK_LAMBDA))
    / (2.0 * GAUSS_MARKOV_PEAK_LAMBDA**2)
)


def bias_instability_to_gm_psd(
    bias_instability: np.ndarray, correlation_time_s: np.ndarray
) -> np.ndarray:
    """Convert Allan-variance Bias Instability B and correlation time T_B to GM driving PSD S_B.

    Matches the first-order Gauss-Markov Allan deviation peak
    (c_GM * sqrt(S_B * T_B) at tau = 1.89 * T_B) to the IEEE Std 952 flicker floor
    (c_BI * B), per Farrell et al. (2022), Eq. (8)-(10) in docs/mathematical_model.md.
    """
    ratio = BIAS_INSTABILITY_AV_FACTOR / GAUSS_MARKOV_PEAK_FACTOR
    return ((ratio * bias_instability) ** 2) / correlation_time_s


@dataclass(frozen=True)
class DiscreteStochasticParams:
    """Exact discrete-time state-space parameters for a 3-axis stochastic error process."""

    dt: float
    white_noise_std: np.ndarray        # sigma_{N,d} = N / sqrt(dt)
    gm_psd: np.ndarray                 # S_B (continuous driving PSD)
    gm_steady_state_std: np.ndarray    # sigma_gm = sqrt(S_B * T_B / 2)
    gm_phi: np.ndarray                 # Phi_B = exp(-dt / T_B)
    gm_driving_std: np.ndarray         # sqrt(q_{B,d}) = sqrt((S_B*T_B/2) * (1 - exp(-2*dt/T_B)))
    rw_driving_std: np.ndarray         # sqrt(q_{K,d}) = K * sqrt(dt)


class ThreeAxisStochasticError:
    """Exact discrete-time generator for 3-axis IMU stochastic errors.

    Generates mutually independent per-axis error components:
        1. Broadband white measurement noise (ARW / VRW)
        2. First-order Gauss-Markov in-run bias instability state x_B
        3. Random-walk bias drift state x_K
    """

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        noise_density: Sequence[float],
        bias_instability: Sequence[float],
        bias_correlation_time_s: Sequence[float],
        random_walk: Sequence[float],
        stationary_bias_init: bool = True,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")

        self.sample_rate_hz = float(sample_rate_hz)
        self.dt = 1.0 / self.sample_rate_hz
        self.noise_density = np.asarray(noise_density, dtype=float)
        self.bias_instability = np.asarray(bias_instability, dtype=float)
        self.bias_correlation_time_s = np.asarray(bias_correlation_time_s, dtype=float)
        self.random_walk = np.asarray(random_walk, dtype=float)
        self.stationary_bias_init = bool(stationary_bias_init)

        # Precompute exact discrete-time transition and noise std vectors (Eqs. 14-17)
        white_noise_std = self.noise_density / math.sqrt(self.dt)
        gm_psd = bias_instability_to_gm_psd(self.bias_instability, self.bias_correlation_time_s)
        gm_var_ss = 0.5 * gm_psd * self.bias_correlation_time_s
        gm_steady_state_std = np.sqrt(gm_var_ss)
        gm_phi = np.exp(-self.dt / self.bias_correlation_time_s)
        gm_driving_var = gm_var_ss * (-np.expm1(-2.0 * self.dt / self.bias_correlation_time_s))
        gm_driving_std = np.sqrt(np.maximum(gm_driving_var, 0.0))
        rw_driving_std = self.random_walk * math.sqrt(self.dt)

        self.params = DiscreteStochasticParams(
            dt=self.dt,
            white_noise_std=white_noise_std,
            gm_psd=gm_psd,
            gm_steady_state_std=gm_steady_state_std,
            gm_phi=gm_phi,
            gm_driving_std=gm_driving_std,
            rw_driving_std=rw_driving_std,
        )

        # Internal bias states x_B[k] and x_K[k]
        self._x_gm = np.zeros(3, dtype=float)
        self._x_rw = np.zeros(3, dtype=float)

    @property
    def gm_bias(self) -> np.ndarray:
        """Current first-order Gauss-Markov bias state x_B[k] (copy)."""
        return self._x_gm.copy()

    @property
    def rw_bias(self) -> np.ndarray:
        """Current random-walk bias drift state x_K[k] (copy)."""
        return self._x_rw.copy()

    @property
    def total_dynamic_bias(self) -> np.ndarray:
        """Current total time-varying stochastic bias x_B[k] + x_K[k]."""
        return self._x_gm + self._x_rw

    def reset(self, rng: np.random.Generator) -> None:
        """Reset stochastic bias states to initial conditions at k = 0."""
        if self.stationary_bias_init:
            self._x_gm = rng.normal(0.0, self.params.gm_steady_state_std)
        else:
            self._x_gm = np.zeros(3, dtype=float)
        self._x_rw = np.zeros(3, dtype=float)

    def step(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Emit total stochastic error z[k] at current step and advance states to k+1.

        Returns:
            (z_k, dynamic_bias_k) where:
                - dynamic_bias_k = x_B[k] + x_K[k]
                - z_k = x_B[k] + x_K[k] + v_N[k]
        """
        dynamic_bias_k = self._x_gm + self._x_rw
        v_n = rng.normal(0.0, self.params.white_noise_std)
        z_k = dynamic_bias_k + v_n

        # Exact state propagation to k + 1 (Eqs. 14-16)
        w_gm = rng.normal(0.0, self.params.gm_driving_std)
        w_rw = rng.normal(0.0, self.params.rw_driving_std)
        self._x_gm = self.params.gm_phi * self._x_gm + w_gm
        self._x_rw = self._x_rw + w_rw

        return z_k, dynamic_bias_k

    def generate(self, n_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Generate a sequence of n_samples stochastic error vectors from current state.

        Produces results statistically and algebraically identical to calling step()
        n_samples times, advancing the internal states to k + n_samples.

        Returns:
            (z_seq, dynamic_bias_seq) each of shape (n_samples, 3).
        """
        if n_samples < 0:
            raise ValueError(f"n_samples must be non-negative, got {n_samples}")
        if n_samples == 0:
            empty = np.empty((0, 3), dtype=float)
            return empty, empty.copy()

        # Draw all innovations upfront for speed while preserving exact recurrence
        v_n = rng.normal(0.0, self.params.white_noise_std, size=(n_samples, 3))
        w_gm = rng.normal(0.0, self.params.gm_driving_std, size=(n_samples, 3))
        w_rw = rng.normal(0.0, self.params.rw_driving_std, size=(n_samples, 3))

        # Random walk state sequence: x_K[0] is current state, x_K[k+1] = x_K[k] + w_rw[k]
        x_rw_seq = np.empty((n_samples, 3), dtype=float)
        x_rw_seq[0] = self._x_rw
        if n_samples > 1:
            x_rw_seq[1:] = self._x_rw + np.cumsum(w_rw[:-1], axis=0)
        self._x_rw = x_rw_seq[-1] + w_rw[-1]

        # First-order Gauss-Markov sequence: x_B[k+1] = phi * x_B[k] + w_gm[k]
        x_gm_seq = np.empty((n_samples, 3), dtype=float)
        phi = self.params.gm_phi
        state = self._x_gm.copy()
        for k in range(n_samples):
            x_gm_seq[k] = state
            state = phi * state + w_gm[k]
        self._x_gm = state

        dynamic_bias_seq = x_gm_seq + x_rw_seq
        z_seq = dynamic_bias_seq + v_n
        return z_seq, dynamic_bias_seq


def theoretical_allan_variance(
    tau: np.ndarray,
    *,
    noise_density: float,
    bias_instability: float,
    bias_correlation_time_s: float,
    random_walk: float,
    quantization_step: float = 0.0,
    sample_rate_hz: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute exact analytical Allan variance and components (Farrell 2022; El-Sheimy 2008).

    Evaluates Eq. (22) of docs/mathematical_model.md for a single rate/accel axis:
        sigma_A^2(tau) = (Delta_LSB^2 * dt) / (12 * tau)
                         + N^2 / tau
                         + S_B * T_B * (1/lam - (3 - 4*exp(-lam) + exp(-2*lam)) / (2*lam^2))
                         + K^2 * tau / 3

    Returns a dictionary containing 'total', 'quantization', 'white_noise',
    'gauss_markov', and 'random_walk' Allan variances (same shape as tau).
    """
    tau_arr = np.asarray(tau, dtype=float)
    if np.any(tau_arr <= 0.0):
        raise ValueError("All cluster times tau must be strictly positive")

    # Direct rate/acceleration ADC quantization contributes white variance Delta_LSB^2 / 12 per sample
    if quantization_step > 0.0:
        if sample_rate_hz is None or sample_rate_hz <= 0.0:
            raise ValueError(
                "sample_rate_hz must be provided and positive when quantization_step > 0"
            )
        dt = 1.0 / float(sample_rate_hz)
        var_q = ((quantization_step**2) * dt) / (12.0 * tau_arr)
    else:
        var_q = np.zeros_like(tau_arr)

    # Broadband white noise (ARW / VRW)
    var_n = (noise_density**2) / tau_arr

    # First-order Gauss-Markov bias instability (Farrell et al., 2022)
    if bias_instability > 0.0:
        s_b = float(
            bias_instability_to_gm_psd(
                np.array([bias_instability]), np.array([bias_correlation_time_s])
            )[0]
        )
        lam = tau_arr / bias_correlation_time_s
        # Numerically stable evaluation for both small and large lambda
        term = np.empty_like(lam)
        small = lam < 1e-4
        if np.any(small):
            # Taylor expansion: 1/lam - (3 - 4e^-lam + e^-2lam)/(2 lam^2) = lam/3 - lam^2/4 + ...
            ls = lam[small]
            term[small] = (ls / 3.0) - (ls**2) / 4.0 + (7.0 / 60.0) * (ls**3)
        if np.any(~small):
            ll = lam[~small]
            term[~small] = (1.0 / ll) - (
                3.0 - 4.0 * np.exp(-ll) + np.exp(-2.0 * ll)
            ) / (2.0 * ll**2)
        var_gm = s_b * bias_correlation_time_s * np.maximum(term, 0.0)
    else:
        var_gm = np.zeros_like(tau_arr)

    # Rate / Acceleration Random Walk
    var_k = (random_walk**2) * tau_arr / 3.0

    var_total = var_q + var_n + var_gm + var_k
    return {
        "total": var_total,
        "quantization": var_q,
        "white_noise": var_n,
        "gauss_markov": var_gm,
        "random_walk": var_k,
    }


def overlapping_allan_variance(
    data: np.ndarray,
    sample_rate_hz: float,
    max_clusters: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute overlapping Allan variance per El-Sheimy et al. (2008), Eq. (5).

    Args:
        data: 1-D stationary measurement sequence of length M.
        sample_rate_hz: Sampling frequency f_s [Hz].
        max_clusters: Number of logarithmically spaced cluster sizes m to evaluate.

    Returns:
        (tau, avar) arrays where tau = m * dt [s] and avar = sigma_A^2(tau).
    """
    y = np.asarray(data, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {y.shape}")
    n_pts = y.size
    if n_pts < 6:
        raise ValueError("At least 6 samples are required to compute Allan variance")

    dt = 1.0 / float(sample_rate_hz)
    max_m = max(1, (n_pts - 1) // 3)
    m_candidates = np.unique(
        np.floor(np.logspace(0.0, math.log10(max_m), num=max_clusters)).astype(int)
    )
    m_values = m_candidates[(m_candidates >= 1) & (2 * m_candidates < n_pts)]

    # Integrated phase sequence theta_k = dt * sum_{j=0}^{k-1} y_j, with theta_0 = 0
    theta = np.empty(n_pts + 1, dtype=float)
    theta[0] = 0.0
    np.cumsum(y * dt, out=theta[1:])

    tau_out = np.empty(m_values.size, dtype=float)
    avar_out = np.empty(m_values.size, dtype=float)
    for idx, m in enumerate(m_values):
        tau = m * dt
        diff = theta[2 * m :] - 2.0 * theta[m:-m] + theta[: -2 * m]
        tau_out[idx] = tau
        avar_out[idx] = float(np.mean(diff**2) / (2.0 * tau**2))

    return tau_out, avar_out
