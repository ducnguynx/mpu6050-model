"""Configuration schemas, hardware lookup tables, and YAML loader for MPU-6050.

Hardware specifications follow InvenSense MPU-6000/MPU-6050 Product Specification
(PS-MPU-6000A-00, Rev 3.4) and Register Map (RM-MPU-6000A-00, Rev 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import numpy as np
import yaml

# CGPM standard gravity [m/s^2]
STANDARD_GRAVITY: float = 9.80665

# 16-bit signed two's-complement ADC limits (PS-MPU-6000A-00 §6.1-6.2)
ADC_MIN_COUNT: int = -32768
ADC_MAX_COUNT: int = 32767

# Accelerometer FSR [g] -> nominal sensitivity [LSB/g] (PS-MPU-6000A-00 Table 6.2)
ACCEL_SENSITIVITY_LSB_PER_G: dict[int, float] = {
    2: 16384.0,
    4: 8192.0,
    8: 4096.0,
    16: 2048.0,
}

# Gyroscope FSR [deg/s] -> nominal sensitivity [LSB/(deg/s)] (PS-MPU-6000A-00 Table 6.1)
GYRO_SENSITIVITY_LSB_PER_DPS: dict[int, float] = {
    250: 131.0,
    500: 65.5,
    1000: 32.8,
    2000: 16.4,
}


def _to_vec3(value: float | Sequence[float] | np.ndarray, name: str) -> tuple[float, float, float]:
    """Validate and convert a scalar or 3-element sequence into a 3-tuple of floats."""
    if isinstance(value, (int, float)):
        val = float(value)
        return (val, val, val)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a scalar or 3-element vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite numbers")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _to_misalignment_matrix(
    value: Sequence[Sequence[float]] | np.ndarray, name: str
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Validate a 3x3 cross-axis misalignment matrix with zero diagonal."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite numbers")
    if not np.allclose(np.diag(arr), 0.0, atol=1e-15):
        raise ValueError(
            f"{name} must have zero diagonal entries (scale-factor errors belong in scale_factor_error)"
        )
    return (
        (float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2])),
        (float(arr[1, 0]), float(arr[1, 1]), float(arr[1, 2])),
        (float(arr[2, 0]), float(arr[2, 1]), float(arr[2, 2])),
    )


def _zero_matrix3() -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]:
    return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@dataclass(frozen=True)
class SimulationConfig:
    """Global discrete-time simulation configuration."""

    sample_rate_hz: float = 200.0
    duration_s: float = 10.0
    random_seed: int | None = 42
    stationary_bias_init: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError(f"sample_rate_hz must be positive, got {self.sample_rate_hz}")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise ValueError(f"random_seed must be an int or None, got {type(self.random_seed)}")

    @property
    def dt(self) -> float:
        """Sampling interval Delta t = 1 / f_s [s]."""
        return 1.0 / self.sample_rate_hz

    @property
    def num_samples(self) -> int:
        """Number of discrete samples for duration_s at sample_rate_hz."""
        return int(round(self.duration_s * self.sample_rate_hz))


@dataclass(frozen=True)
class AccelerometerConfig:
    """Configuration for the 3-axis MPU-6050 accelerometer (all internal units SI: m/s^2, s)."""

    full_scale_range_g: int = 2
    scale_factor_error: tuple[float, float, float] = (0.005, -0.004, 0.006)
    misalignment: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ] = field(default_factory=_zero_matrix3)
    static_bias_mps2: tuple[float, float, float] = (0.15, -0.12, 0.25)
    # Derived from PS-MPU-6000A-00: 400 ug/sqrt(Hz) / sqrt(2) in (m/s^2)/sqrt(Hz)
    noise_density_mps2_per_sqrthz: tuple[float, float, float] = (
        0.00277374,
        0.00277374,
        0.00277374,
    )
    bias_instability_mps2: tuple[float, float, float] = (0.0015, 0.0015, 0.0020)
    bias_correlation_time_s: tuple[float, float, float] = (100.0, 100.0, 100.0)
    random_walk_mps2_per_sqrts: tuple[float, float, float] = (0.0001, 0.0001, 0.00015)
    enable_saturation: bool = True
    enable_quantization: bool = True

    def __post_init__(self) -> None:
        fsr = int(self.full_scale_range_g)
        if fsr not in ACCEL_SENSITIVITY_LSB_PER_G:
            raise ValueError(
                f"full_scale_range_g must be one of {sorted(ACCEL_SENSITIVITY_LSB_PER_G)}, got {fsr}"
            )
        object.__setattr__(self, "full_scale_range_g", fsr)
        object.__setattr__(
            self, "scale_factor_error", _to_vec3(self.scale_factor_error, "scale_factor_error")
        )
        object.__setattr__(
            self, "misalignment", _to_misalignment_matrix(self.misalignment, "misalignment")
        )
        object.__setattr__(
            self, "static_bias_mps2", _to_vec3(self.static_bias_mps2, "static_bias_mps2")
        )
        noise_d = _to_vec3(self.noise_density_mps2_per_sqrthz, "noise_density_mps2_per_sqrthz")
        bias_inst = _to_vec3(self.bias_instability_mps2, "bias_instability_mps2")
        corr_t = _to_vec3(self.bias_correlation_time_s, "bias_correlation_time_s")
        rw = _to_vec3(self.random_walk_mps2_per_sqrts, "random_walk_mps2_per_sqrts")

        if any(v < 0.0 for v in noise_d):
            raise ValueError("noise_density_mps2_per_sqrthz must be non-negative")
        if any(v < 0.0 for v in bias_inst):
            raise ValueError("bias_instability_mps2 must be non-negative")
        if any(v <= 0.0 for v in corr_t):
            raise ValueError("bias_correlation_time_s must be strictly positive")
        if any(v < 0.0 for v in rw):
            raise ValueError("random_walk_mps2_per_sqrts must be non-negative")

        object.__setattr__(self, "noise_density_mps2_per_sqrthz", noise_d)
        object.__setattr__(self, "bias_instability_mps2", bias_inst)
        object.__setattr__(self, "bias_correlation_time_s", corr_t)
        object.__setattr__(self, "random_walk_mps2_per_sqrts", rw)

    @property
    def fsr_mps2(self) -> float:
        """Full-scale measurement limit in m/s^2."""
        return self.full_scale_range_g * STANDARD_GRAVITY

    @property
    def sensitivity_lsb_per_mps2(self) -> float:
        """Nominal ADC sensitivity scale factor in LSB / (m/s^2)."""
        return ACCEL_SENSITIVITY_LSB_PER_G[self.full_scale_range_g] / STANDARD_GRAVITY

    @property
    def quantization_step_mps2(self) -> float:
        """1-LSB quantization step size Delta_LSB in m/s^2."""
        return 1.0 / self.sensitivity_lsb_per_mps2


@dataclass(frozen=True)
class GyroscopeConfig:
    """Configuration for the 3-axis MPU-6050 gyroscope (all internal units SI: rad/s, s)."""

    full_scale_range_dps: int = 250
    scale_factor_error: tuple[float, float, float] = (0.008, -0.006, 0.007)
    misalignment: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ] = field(default_factory=_zero_matrix3)
    static_bias_radps: tuple[float, float, float] = (0.015, -0.010, 0.018)
    # Derived from PS-MPU-6000A-00: 0.005 (deg/s)/sqrt(Hz) / sqrt(2) in (rad/s)/sqrt(Hz)
    noise_density_radps_per_sqrthz: tuple[float, float, float] = (
        6.170671e-5,
        6.170671e-5,
        6.170671e-5,
    )
    bias_instability_radps: tuple[float, float, float] = (1.2e-4, 6.0e-5, 1.2e-4)
    bias_correlation_time_s: tuple[float, float, float] = (100.0, 100.0, 100.0)
    random_walk_radps_per_sqrts: tuple[float, float, float] = (1.0e-5, 1.0e-5, 1.0e-5)
    enable_saturation: bool = True
    enable_quantization: bool = True

    def __post_init__(self) -> None:
        fsr = int(self.full_scale_range_dps)
        if fsr not in GYRO_SENSITIVITY_LSB_PER_DPS:
            raise ValueError(
                f"full_scale_range_dps must be one of {sorted(GYRO_SENSITIVITY_LSB_PER_DPS)}, got {fsr}"
            )
        object.__setattr__(self, "full_scale_range_dps", fsr)
        object.__setattr__(
            self, "scale_factor_error", _to_vec3(self.scale_factor_error, "scale_factor_error")
        )
        object.__setattr__(
            self, "misalignment", _to_misalignment_matrix(self.misalignment, "misalignment")
        )
        object.__setattr__(
            self, "static_bias_radps", _to_vec3(self.static_bias_radps, "static_bias_radps")
        )
        noise_d = _to_vec3(
            self.noise_density_radps_per_sqrthz, "noise_density_radps_per_sqrthz"
        )
        bias_inst = _to_vec3(self.bias_instability_radps, "bias_instability_radps")
        corr_t = _to_vec3(self.bias_correlation_time_s, "bias_correlation_time_s")
        rw = _to_vec3(self.random_walk_radps_per_sqrts, "random_walk_radps_per_sqrts")

        if any(v < 0.0 for v in noise_d):
            raise ValueError("noise_density_radps_per_sqrthz must be non-negative")
        if any(v < 0.0 for v in bias_inst):
            raise ValueError("bias_instability_radps must be non-negative")
        if any(v <= 0.0 for v in corr_t):
            raise ValueError("bias_correlation_time_s must be strictly positive")
        if any(v < 0.0 for v in rw):
            raise ValueError("random_walk_radps_per_sqrts must be non-negative")

        object.__setattr__(self, "noise_density_radps_per_sqrthz", noise_d)
        object.__setattr__(self, "bias_instability_radps", bias_inst)
        object.__setattr__(self, "bias_correlation_time_s", corr_t)
        object.__setattr__(self, "random_walk_radps_per_sqrts", rw)

    @property
    def fsr_radps(self) -> float:
        """Full-scale measurement limit in rad/s."""
        return math.radians(self.full_scale_range_dps)

    @property
    def sensitivity_lsb_per_radps(self) -> float:
        """Nominal ADC sensitivity scale factor in LSB / (rad/s)."""
        return GYRO_SENSITIVITY_LSB_PER_DPS[self.full_scale_range_dps] * (180.0 / math.pi)

    @property
    def quantization_step_radps(self) -> float:
        """1-LSB quantization step size Delta_LSB in rad/s."""
        return 1.0 / self.sensitivity_lsb_per_radps


@dataclass(frozen=True)
class MPU6050Config:
    """Complete configuration container for the 6-axis MPU-6050 behavioral simulator."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    accelerometer: AccelerometerConfig = field(default_factory=AccelerometerConfig)
    gyroscope: GyroscopeConfig = field(default_factory=GyroscopeConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MPU6050Config:
        """Build an MPU6050Config from a nested dictionary."""
        sim_data = dict(data.get("simulation", {}))
        accel_data = dict(data.get("accelerometer", {}))
        gyro_data = dict(data.get("gyroscope", {}))
        return cls(
            simulation=SimulationConfig(**sim_data),
            accelerometer=AccelerometerConfig(**accel_data),
            gyroscope=GyroscopeConfig(**gyro_data),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> MPU6050Config:
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"YAML configuration root must be a mapping, got {type(raw)}")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain Python dictionary suitable for YAML export."""
        return {
            "simulation": {
                "sample_rate_hz": self.simulation.sample_rate_hz,
                "duration_s": self.simulation.duration_s,
                "random_seed": self.simulation.random_seed,
                "stationary_bias_init": self.simulation.stationary_bias_init,
            },
            "accelerometer": {
                "full_scale_range_g": self.accelerometer.full_scale_range_g,
                "scale_factor_error": list(self.accelerometer.scale_factor_error),
                "misalignment": [list(row) for row in self.accelerometer.misalignment],
                "static_bias_mps2": list(self.accelerometer.static_bias_mps2),
                "noise_density_mps2_per_sqrthz": list(
                    self.accelerometer.noise_density_mps2_per_sqrthz
                ),
                "bias_instability_mps2": list(self.accelerometer.bias_instability_mps2),
                "bias_correlation_time_s": list(self.accelerometer.bias_correlation_time_s),
                "random_walk_mps2_per_sqrts": list(self.accelerometer.random_walk_mps2_per_sqrts),
                "enable_saturation": self.accelerometer.enable_saturation,
                "enable_quantization": self.accelerometer.enable_quantization,
            },
            "gyroscope": {
                "full_scale_range_dps": self.gyroscope.full_scale_range_dps,
                "scale_factor_error": list(self.gyroscope.scale_factor_error),
                "misalignment": [list(row) for row in self.gyroscope.misalignment],
                "static_bias_radps": list(self.gyroscope.static_bias_radps),
                "noise_density_radps_per_sqrthz": list(
                    self.gyroscope.noise_density_radps_per_sqrthz
                ),
                "bias_instability_radps": list(self.gyroscope.bias_instability_radps),
                "bias_correlation_time_s": list(self.gyroscope.bias_correlation_time_s),
                "random_walk_radps_per_sqrts": list(self.gyroscope.random_walk_radps_per_sqrts),
                "enable_saturation": self.gyroscope.enable_saturation,
                "enable_quantization": self.gyroscope.enable_quantization,
            },
        }

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def with_updates(
        self,
        *,
        simulation: Mapping[str, Any] | None = None,
        accelerometer: Mapping[str, Any] | None = None,
        gyroscope: Mapping[str, Any] | None = None,
    ) -> MPU6050Config:
        """Return a new MPU6050Config with selected fields updated."""
        new_sim = replace(self.simulation, **simulation) if simulation else self.simulation
        new_acc = (
            replace(self.accelerometer, **accelerometer) if accelerometer else self.accelerometer
        )
        new_gyr = replace(self.gyroscope, **gyroscope) if gyroscope else self.gyroscope
        return MPU6050Config(simulation=new_sim, accelerometer=new_acc, gyroscope=new_gyr)
