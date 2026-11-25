"""Accelerometer-focused behavioral simulations and Scenario 4 parameter sweeps.

Demonstrates:
    - 3-axis accelerometer response to multi-frequency input
    - Scenario 4A: Varying accelerometer white noise density (VRW N_a)
    - Scenario 4B: Varying accelerometer static bias (b_{0,a})
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
    ideal_acc = np.column_stack(
        [
            2.0 * np.sin(2.0 * np.pi * 0.6 * t) + 0.6 * np.sin(2.0 * np.pi * 4.0 * t),
            1.5 * np.cos(2.0 * np.pi * 0.9 * t),
            STANDARD_GRAVITY + 1.0 * np.sin(2.0 * np.pi * 1.5 * t),
        ]
    )
    ideal_gyr = np.zeros_like(ideal_acc)

    # Baseline 3-axis accelerometer plot
    base_result = sensor.simulate(ideal_acc, ideal_gyr)
    plot_sensor_axes(
        t,
        ideal_acc,
        base_result.accelerometer,
        sensor_name="Accelerometer",
        unit="m/s²",
        title="MPU-6050 Accelerometer: Ideal vs. Simulated Output (3 Axes)",
        config_note=(
            f"fs={sensor.sample_rate_hz:.0f} Hz | FSR=±{sensor.config.accelerometer.full_scale_range_g}g | "
            f"N_a={sensor.config.accelerometer.noise_density_mps2_per_sqrthz[0]:.6f} (m/s²)/√Hz"
        ),
        save_path=OUTPUT_DIR / "accel_3axis_multifreq.png",
    )

    # Scenario 4A: Noise density variation on X-axis (keeping bias & deterministic errors fixed)
    base_nd = sensor.config.accelerometer.noise_density_mps2_per_sqrthz[0]
    noise_sweeps = []
    for mult, desc in [(0.0, "No noise (0×)"), (1.0, "Datasheet VRW (1×)"), (5.0, "High noise (5×)"), (15.0, "Very high noise (15×)")]:
        s = MPU6050.from_config(CONFIG_PATH)
        s.update_parameters(
            accelerometer={
                "static_bias_mps2": [0.0, 0.0, 0.0],
                "scale_factor_error": [0.0, 0.0, 0.0],
                "misalignment": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "bias_instability_mps2": [0.0, 0.0, 0.0],
                "random_walk_mps2_per_sqrts": [0.0, 0.0, 0.0],
                "noise_density_mps2_per_sqrthz": [base_nd * mult] * 3,
            }
        )
        res = s.simulate(ideal_acc, ideal_gyr)
        noise_sweeps.append((f"{desc}: N_a={base_nd * mult:.5f}", res.accelerometer[:, 0]))

    plot_parameter_sweep(
        t,
        ideal_acc[:, 0],
        noise_sweeps,
        ylabel="X-axis Acceleration [m/s²]",
        title="Scenario 4A: Influence of Accelerometer White Noise Density (VRW)",
        config_note="Deterministic errors & bias disabled to isolate broadband noise N_a (fs=200 Hz, seed=42)",
        save_path=OUTPUT_DIR / "scenario4_accel_noise_sweep.png",
    )

    # Scenario 4B: Static bias variation on Z-axis (within MPU-6050 +/-80 mg spec and beyond)
    bias_sweeps = []
    for b_val in [0.0, 0.25, 0.785, -0.50]:
        s = MPU6050.from_config(CONFIG_PATH)
        s.update_parameters(
            accelerometer={
                "static_bias_mps2": [0.0, 0.0, b_val],
                "scale_factor_error": [0.0, 0.0, 0.0],
                "misalignment": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            }
        )
        res = s.simulate(ideal_acc, ideal_gyr)
        bias_sweeps.append((f"b_0,z = {b_val:+.3f} m/s² ({b_val / STANDARD_GRAVITY * 1e3:+.1f} mg)", res.accelerometer[:, 2]))

    plot_parameter_sweep(
        t,
        ideal_acc[:, 2],
        bias_sweeps,
        ylabel="Z-axis Acceleration [m/s²]",
        title="Scenario 4B: Influence of Accelerometer Static Bias",
        config_note="MPU-6050 datasheet Z-axis initial zero-g tolerance is ±80 mg (±0.785 m/s²)",
        save_path=OUTPUT_DIR / "scenario4_accel_bias_sweep.png",
    )

    print(f"Accelerometer simulation and Scenario 4 sweep plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
