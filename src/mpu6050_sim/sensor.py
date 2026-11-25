"""Integrated 6-axis MPU-6050 behavioral simulator and data export interface."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .accelerometer import Accelerometer
from .configuration import MPU6050Config
from .gyroscope import Gyroscope


@dataclass(frozen=True)
class IMUMeasurement:
    """Single-step measurement produced by the 6-axis MPU-6050 simulator."""

    timestamp_s: float
    accelerometer: np.ndarray           # Measured specific force [m/s^2], shape (3,)
    gyroscope: np.ndarray               # Measured angular velocity [rad/s], shape (3,)
    raw_accelerometer: np.ndarray       # 16-bit signed ADC counts (int16), shape (3,)
    raw_gyroscope: np.ndarray           # 16-bit signed ADC counts (int16), shape (3,)
    ideal_acceleration: np.ndarray      # Ground-truth specific force [m/s^2], shape (3,)
    ideal_angular_velocity: np.ndarray  # Ground-truth angular velocity [rad/s], shape (3,)


@dataclass(frozen=True)
class SimulationResult:
    """Batch simulation output retaining both ground-truth inputs and sensor outputs."""

    time_s: np.ndarray                  # Shape (N,)
    ideal_acceleration: np.ndarray      # Shape (N, 3) [m/s^2]
    ideal_angular_velocity: np.ndarray  # Shape (N, 3) [rad/s]
    accelerometer: np.ndarray           # Shape (N, 3) [m/s^2]
    gyroscope: np.ndarray               # Shape (N, 3) [rad/s]
    raw_accelerometer: np.ndarray       # Shape (N, 3) int16 [LSB]
    raw_gyroscope: np.ndarray           # Shape (N, 3) int16 [LSB]
    sample_rate_hz: float
    config: MPU6050Config

    @property
    def num_samples(self) -> int:
        return int(self.time_s.shape[0])

    def to_csv(self, path: str | Path) -> Path:
        """Export simulation time series (ideal inputs + simulated outputs) to CSV."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "time_s",
            "ideal_ax_mps2",
            "ideal_ay_mps2",
            "ideal_az_mps2",
            "ideal_gx_radps",
            "ideal_gy_radps",
            "ideal_gz_radps",
            "meas_ax_mps2",
            "meas_ay_mps2",
            "meas_az_mps2",
            "meas_gx_radps",
            "meas_gy_radps",
            "meas_gz_radps",
            "raw_ax_lsb",
            "raw_ay_lsb",
            "raw_az_lsb",
            "raw_gx_lsb",
            "raw_gy_lsb",
            "raw_gz_lsb",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for k in range(self.num_samples):
                writer.writerow(
                    [
                        f"{self.time_s[k]:.9f}",
                        f"{self.ideal_acceleration[k, 0]:.9f}",
                        f"{self.ideal_acceleration[k, 1]:.9f}",
                        f"{self.ideal_acceleration[k, 2]:.9f}",
                        f"{self.ideal_angular_velocity[k, 0]:.9f}",
                        f"{self.ideal_angular_velocity[k, 1]:.9f}",
                        f"{self.ideal_angular_velocity[k, 2]:.9f}",
                        f"{self.accelerometer[k, 0]:.9f}",
                        f"{self.accelerometer[k, 1]:.9f}",
                        f"{self.accelerometer[k, 2]:.9f}",
                        f"{self.gyroscope[k, 0]:.9f}",
                        f"{self.gyroscope[k, 1]:.9f}",
                        f"{self.gyroscope[k, 2]:.9f}",
                        int(self.raw_accelerometer[k, 0]),
                        int(self.raw_accelerometer[k, 1]),
                        int(self.raw_accelerometer[k, 2]),
                        int(self.raw_gyroscope[k, 0]),
                        int(self.raw_gyroscope[k, 1]),
                        int(self.raw_gyroscope[k, 2]),
                    ]
                )
        return out_path

    def to_dataframe(self) -> Any:
        """Convert simulation result to a pandas.DataFrame (requires optional pandas)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(); use to_csv() or install pandas."
            ) from exc
        return pd.DataFrame(
            {
                "time_s": self.time_s,
                "ideal_ax_mps2": self.ideal_acceleration[:, 0],
                "ideal_ay_mps2": self.ideal_acceleration[:, 1],
                "ideal_az_mps2": self.ideal_acceleration[:, 2],
                "ideal_gx_radps": self.ideal_angular_velocity[:, 0],
                "ideal_gy_radps": self.ideal_angular_velocity[:, 1],
                "ideal_gz_radps": self.ideal_angular_velocity[:, 2],
                "meas_ax_mps2": self.accelerometer[:, 0],
                "meas_ay_mps2": self.accelerometer[:, 1],
                "meas_az_mps2": self.accelerometer[:, 2],
                "meas_gx_radps": self.gyroscope[:, 0],
                "meas_gy_radps": self.gyroscope[:, 1],
                "meas_gz_radps": self.gyroscope[:, 2],
                "raw_ax_lsb": self.raw_accelerometer[:, 0],
                "raw_ay_lsb": self.raw_accelerometer[:, 1],
                "raw_az_lsb": self.raw_accelerometer[:, 2],
                "raw_gx_lsb": self.raw_gyroscope[:, 0],
                "raw_gy_lsb": self.raw_gyroscope[:, 1],
                "raw_gz_lsb": self.raw_gyroscope[:, 2],
            }
        )


class MPU6050:
    """Configurable behavioral simulator of the InvenSense MPU-6050 6-axis IMU.

    Combines an independent 3-axis Accelerometer and 3-axis Gyroscope with
    independent seeded RNG streams spawned from a master SeedSequence so that
    modifying one subsystem never perturbs the other's random draws.
    """

    def __init__(self, config: MPU6050Config | None = None) -> None:
        self._config = config if config is not None else MPU6050Config()
        self._step_index = 0
        self._build_subsystems(seed=self._config.simulation.random_seed)

    @classmethod
    def from_config(cls, path: str | Path) -> MPU6050:
        """Create an MPU6050 simulator from a YAML configuration file."""
        return cls(MPU6050Config.from_yaml(path))

    @property
    def config(self) -> MPU6050Config:
        """Active immutable MPU6050Config."""
        return self._config

    @property
    def sample_rate_hz(self) -> float:
        return self._config.simulation.sample_rate_hz

    @property
    def dt(self) -> float:
        return self._config.simulation.dt

    def _spawn_rngs(self, seed: int | None) -> tuple[np.random.Generator, np.random.Generator]:
        ss = np.random.SeedSequence(seed)
        accel_ss, gyro_ss = ss.spawn(2)
        return np.random.default_rng(accel_ss), np.random.default_rng(gyro_ss)

    def _build_subsystems(self, seed: int | None) -> None:
        accel_rng, gyro_rng = self._spawn_rngs(seed)
        self.accelerometer = Accelerometer(
            self._config.accelerometer,
            sample_rate_hz=self._config.simulation.sample_rate_hz,
            stationary_bias_init=self._config.simulation.stationary_bias_init,
            rng=accel_rng,
        )
        self.gyroscope = Gyroscope(
            self._config.gyroscope,
            sample_rate_hz=self._config.simulation.sample_rate_hz,
            stationary_bias_init=self._config.simulation.stationary_bias_init,
            rng=gyro_rng,
        )
        self._step_index = 0

    def reset(self, seed: int | None = None) -> None:
        """Reset internal stochastic bias states and RNG streams to t = 0."""
        effective_seed = self._config.simulation.random_seed if seed is None else seed
        accel_rng, gyro_rng = self._spawn_rngs(effective_seed)
        self.accelerometer.reset(accel_rng)
        self.gyroscope.reset(gyro_rng)
        self._step_index = 0

    def update_parameters(
        self,
        *,
        simulation: Mapping[str, Any] | None = None,
        accelerometer: Mapping[str, Any] | None = None,
        gyroscope: Mapping[str, Any] | None = None,
    ) -> None:
        """Update configurable parameters and rebuild subsystems deterministically."""
        self._config = self._config.with_updates(
            simulation=simulation,
            accelerometer=accelerometer,
            gyroscope=gyroscope,
        )
        self._build_subsystems(seed=self._config.simulation.random_seed)

    def time_vector(self, duration_s: float | None = None) -> np.ndarray:
        """Generate uniform discrete timestamps [s] for the configured sample rate."""
        dur = self._config.simulation.duration_s if duration_s is None else float(duration_s)
        n = int(round(dur * self.sample_rate_hz))
        return np.arange(n, dtype=float) * self.dt

    def step(
        self,
        acceleration: Sequence[float] | np.ndarray,
        angular_velocity: Sequence[float] | np.ndarray,
    ) -> IMUMeasurement:
        """Generate one simulated 6-axis MPU-6050 measurement at the next sample step."""
        timestamp = self._step_index * self.dt
        acc_meas = self.accelerometer.step(acceleration)
        gyr_meas = self.gyroscope.step(angular_velocity)
        self._step_index += 1
        return IMUMeasurement(
            timestamp_s=timestamp,
            accelerometer=acc_meas.output,
            gyroscope=gyr_meas.output,
            raw_accelerometer=acc_meas.raw_counts,
            raw_gyroscope=gyr_meas.raw_counts,
            ideal_acceleration=acc_meas.ideal_input,
            ideal_angular_velocity=gyr_meas.ideal_input,
        )

    def simulate(
        self,
        accelerations: Sequence[Sequence[float]] | np.ndarray,
        angular_velocities: Sequence[Sequence[float]] | np.ndarray,
        *,
        start_time_s: float | None = None,
    ) -> SimulationResult:
        """Simulate a synchronized sequence of N 6-axis MPU-6050 measurements."""
        acc_in = np.asarray(accelerations, dtype=float)
        gyr_in = np.asarray(angular_velocities, dtype=float)
        if acc_in.ndim != 2 or acc_in.shape[1] != 3:
            raise ValueError(f"accelerations must have shape (N, 3), got {acc_in.shape}")
        if gyr_in.ndim != 2 or gyr_in.shape[1] != 3:
            raise ValueError(f"angular_velocities must have shape (N, 3), got {gyr_in.shape}")
        if acc_in.shape[0] != gyr_in.shape[0]:
            raise ValueError(
                f"accelerations ({acc_in.shape[0]}) and angular_velocities ({gyr_in.shape[0]}) "
                "must have the same number of samples"
            )

        n_samples = acc_in.shape[0]
        t0 = self._step_index * self.dt if start_time_s is None else float(start_time_s)
        time_s = t0 + np.arange(n_samples, dtype=float) * self.dt

        acc_out, acc_raw = self.accelerometer.simulate(acc_in)
        gyr_out, gyr_raw = self.gyroscope.simulate(gyr_in)
        self._step_index += n_samples

        return SimulationResult(
            time_s=time_s,
            ideal_acceleration=acc_in.copy(),
            ideal_angular_velocity=gyr_in.copy(),
            accelerometer=acc_out,
            gyroscope=gyr_out,
            raw_accelerometer=acc_raw,
            raw_gyroscope=gyr_raw,
            sample_rate_hz=self.sample_rate_hz,
            config=self._config,
        )
