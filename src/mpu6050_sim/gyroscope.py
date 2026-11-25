"""MPU-6050 3-axis gyroscope behavioral model.

Implements the deterministic + stochastic + ADC output equations documented in
docs/mathematical_model.md (Farrell et al., IEEE CSM 2022; InvenSense
PS-MPU-6000A-00 Rev 3.4):

    w_analog[k] = (I_3 + S_g + M_g) w_true[k] + b_{0,g} + z_g[k]
    w_sat[k]    = clip(w_analog[k], -w_FSR, +w_FSR)
    c_g[k]      = clip(round(w_sat[k] * S_nom,g), -32768, 32767)
    w_meas[k]   = c_g[k] / S_nom,g
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .configuration import ADC_MAX_COUNT, ADC_MIN_COUNT, GyroscopeConfig
from .noise import ThreeAxisStochasticError


@dataclass(frozen=True)
class GyroscopeMeasurement:
    """Single-step output record from the MPU-6050 gyroscope."""

    output: np.ndarray            # Measured angular velocity [rad/s], shape (3,)
    raw_counts: np.ndarray        # 16-bit signed ADC counts (int16), shape (3,)
    ideal_input: np.ndarray       # Ground-truth angular velocity [rad/s], shape (3,)
    dynamic_bias: np.ndarray      # Stochastic bias x_B[k] + x_K[k] [rad/s], shape (3,)
    stochastic_error: np.ndarray  # Total stochastic error z_g[k] [rad/s], shape (3,)


class Gyroscope:
    """Behavioral model of the InvenSense MPU-6050 3-axis gyroscope."""

    def __init__(
        self,
        config: GyroscopeConfig,
        *,
        sample_rate_hz: float,
        stationary_bias_init: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.sample_rate_hz = float(sample_rate_hz)
        self.stationary_bias_init = bool(stationary_bias_init)
        self._rng = rng if rng is not None else np.random.default_rng()

        # Deterministic error matrices (Eqs. 1-2)
        self.scale_factor_matrix = np.diag(np.asarray(config.scale_factor_error, dtype=float))
        self.misalignment_matrix = np.asarray(config.misalignment, dtype=float)
        self.gain_matrix = np.eye(3, dtype=float) + self.scale_factor_matrix + self.misalignment_matrix
        self.static_bias = np.asarray(config.static_bias_radps, dtype=float)

        # Stochastic error generator (Eqs. 3-18)
        self.stochastic = ThreeAxisStochasticError(
            sample_rate_hz=self.sample_rate_hz,
            noise_density=config.noise_density_radps_per_sqrthz,
            bias_instability=config.bias_instability_radps,
            bias_correlation_time_s=config.bias_correlation_time_s,
            random_walk=config.random_walk_radps_per_sqrts,
            stationary_bias_init=self.stationary_bias_init,
        )
        self.stochastic.reset(self._rng)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset stochastic bias states (and optionally replace the RNG)."""
        if rng is not None:
            self._rng = rng
        self.stochastic.reset(self._rng)

    def _apply_output_stage(self, analog_output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply optional FSR saturation and 16-bit ADC quantization (Eqs. 19-21)."""
        value = analog_output
        if self.config.enable_saturation:
            limit = self.config.fsr_radps
            value = np.clip(value, -limit, limit)

        sensitivity = self.config.sensitivity_lsb_per_radps
        counts = np.clip(np.rint(value * sensitivity), ADC_MIN_COUNT, ADC_MAX_COUNT).astype(np.int16)

        if self.config.enable_quantization:
            measured = counts.astype(float) / sensitivity
        else:
            measured = value
        return measured, counts

    def step(self, angular_velocity: Sequence[float] | np.ndarray) -> GyroscopeMeasurement:
        """Generate a single 3-axis gyroscope measurement [rad/s]."""
        u_true = np.asarray(angular_velocity, dtype=float)
        if u_true.shape != (3,):
            raise ValueError(f"angular_velocity must have shape (3,), got {u_true.shape}")

        z_k, dyn_bias_k = self.stochastic.step(self._rng)
        analog = self.gain_matrix @ u_true + self.static_bias + z_k
        measured, counts = self._apply_output_stage(analog)
        return GyroscopeMeasurement(
            output=measured,
            raw_counts=counts,
            ideal_input=u_true.copy(),
            dynamic_bias=dyn_bias_k,
            stochastic_error=z_k,
        )

    def simulate(
        self, angular_velocities: Sequence[Sequence[float]] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a batch of N gyroscope samples.

        Args:
            angular_velocities: Array of ideal angular-velocity vectors [rad/s], shape (N, 3).

        Returns:
            (measured, raw_counts) with shapes (N, 3) float64 and (N, 3) int16.
        """
        u_seq = np.asarray(angular_velocities, dtype=float)
        if u_seq.ndim != 2 or u_seq.shape[1] != 3:
            raise ValueError(f"angular_velocities must have shape (N, 3), got {u_seq.shape}")

        z_seq, _ = self.stochastic.generate(u_seq.shape[0], self._rng)
        analog = u_seq @ self.gain_matrix.T + self.static_bias + z_seq
        return self._apply_output_stage(analog)
