"""Gyroscope-focused behavioral simulations and Scenario 4 parameter sweeps.

Demonstrates:
    - 3-axis gyroscope response to multi-frequency angular velocity input
    - Scenario 4C: Varying gyroscope white noise density (ARW N_g)
    - Scenario 4D: Varying gyroscope static bias and rate random walk drift (b_{0,g}, K_g)
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpu6050_sim import MPU6050, STANDARD_GRAVITY
from mpu6050_sim.visualization import plot_parameter_sweep, plot_sensor_axes

CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    sensor = MPU6050.from_config(CONFIG_PATH)
    t = sensor.time_vector(duration_s=5.0)
    ideal_acc = np.tile(np.array([0.0, 0.0, STANDARD_GRAVITY]), (t.size, 1))
    ideal_gyr = np.column_stack(
        [
            0.7 * np.sin(2.0 * np.pi * 0.5 * t) + 0.25 * np.sin(2.0 * np.pi * 3.5 * t),
            0.5 * np.cos(2.0 * np.pi * 0.8 * t),
            0.9 * np.sin(2.0 * np.pi * 0.35 * t) + 0.2 * np.cos(2.0 * np.pi * 5.0 * t),
        ]
    )

    # Baseline 3-axis gyroscope plot
    base_result = sensor.simulate(ideal_acc, ideal_gyr)
    plot_sensor_axes(
        t,
        ideal_gyr,
        base_result.gyroscope,
        sensor_name="Gyroscope",
        unit="rad/s",
        title="MPU-6050 Gyroscope: Ideal vs. Simulated Output (3 Axes)",
        config_note=(
            f"fs={sensor.sample_rate_hz:.0f} Hz | FSR=±{sensor.config.gyroscope.full_scale_range_dps} °/s | "
            f"N_g={sensor.config.gyroscope.noise_density_radps_per_sqrthz[0]:.3e} (rad/s)/√Hz"
        ),
        save_path=OUTPUT_DIR / "gyro_3axis_multifreq.png",
    )

    # Scenario 4C: Gyroscope white noise density sweep
    base_nd = sensor.config.gyroscope.noise_density_radps_per_sqrthz[0]
    noise_sweeps = []
    for mult, desc in [(0.0, "No noise (0×)"), (1.0, "Datasheet ARW (1×)"), (10.0, "Elevated noise (10×)"), (30.0, "High noise (30×)")]:
        s = MPU6050.from_config(CONFIG_PATH)
        s.update_parameters(
            gyroscope={
                "static_bias_radps": [0.0, 0.0, 0.0],
                "scale_factor_error": [0.0, 0.0, 0.0],
                "misalignment": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "bias_instability_radps": [0.0, 0.0, 0.0],
                "random_walk_radps_per_sqrts": [0.0, 0.0, 0.0],
                "noise_density_radps_per_sqrthz": [base_nd * mult] * 3,
            }
        )
        res = s.simulate(ideal_acc, ideal_gyr)
        noise_sweeps.append((f"{desc}: N_g={base_nd * mult:.2e}", res.gyroscope[:, 0]))

    plot_parameter_sweep(
        t,
        ideal_gyr[:, 0],
        noise_sweeps,
        ylabel="X-axis Angular Velocity [rad/s]",
        title="Scenario 4C: Influence of Gyroscope White Noise Density (ARW)",
        config_note="Deterministic errors & bias disabled to isolate broadband noise N_g (fs=200 Hz, seed=42)",
        save_path=OUTPUT_DIR / "scenario4_gyro_noise_sweep.png",
    )

    # Scenario 4D: Gyroscope static bias & random-walk drift sweep (60 s horizon)
    t_long = sensor.time_vector(duration_s=60.0)
    acc_long = np.tile(np.array([0.0, 0.0, STANDARD_GRAVITY]), (t_long.size, 1))
    gyr_long = np.zeros((t_long.size, 3), dtype=float)
    bias_sweeps = []
    configs = [
        ("Zero bias, K_g=0", 0.0, 0.0),
        ("Static bias +0.02 rad/s, K_g=0", 0.02, 0.0),
        ("Static bias +0.02 rad/s, K_g=0.002 rad/s^(3/2)", 0.02, 0.002),
        ("Static bias -0.03 rad/s, K_g=0.004 rad/s^(3/2)", -0.03, 0.004),
    ]
    for label, b_val, k_val in configs:
        s = MPU6050.from_config(CONFIG_PATH)
        s.update_parameters(
            simulation={"stationary_bias_init": False},
            gyroscope={
                "static_bias_radps": [0.0, 0.0, b_val],
                "scale_factor_error": [0.0, 0.0, 0.0],
                "misalignment": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "bias_instability_radps": [0.0, 0.0, 0.0],
                "random_walk_radps_per_sqrts": [0.0, 0.0, k_val],
            },
        )
        res = s.simulate(acc_long, gyr_long)
        bias_sweeps.append((label, res.gyroscope[:, 2]))

    plot_parameter_sweep(
        t_long,
        gyr_long[:, 2],
        bias_sweeps,
        ylabel="Z-axis Angular Velocity [rad/s]",
        title="Scenario 4D: Influence of Gyroscope Static Bias and Rate Random Walk Drift",
        config_note="Stationary input (omega_true = 0 rad/s) over 60 s (fs=200 Hz, seed=42)",
        save_path=OUTPUT_DIR / "scenario4_gyro_bias_drift_sweep.png",
    )

    print(f"Gyroscope simulation and Scenario 4 sweep plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
