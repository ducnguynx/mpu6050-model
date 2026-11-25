"""Basic 6-axis MPU-6050 behavioral simulation scenarios and CSV export.

Demonstrates:
    - Single-step Python API (MPU6050.from_config + sensor.step)
    - Scenario 1: Static input (constant gravity on Z, zero angular velocity)
    - Scenario 2: Sinusoidal input (3-axis sinusoidal acceleration and angular velocity)
    - Scenario 3: Multi-frequency input (superposed frequency components)
    - CSV export preserving ideal ground-truth inputs alongside simulated outputs
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpu6050_sim import MPU6050, STANDARD_GRAVITY
from mpu6050_sim.visualization import plot_imu_comparison

CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def run_single_step_demo() -> None:
    """Illustrate the step-by-step API matching a parameterized MATLAB loop."""
    sensor = MPU6050.from_config(CONFIG_PATH)
    acceleration = [0.0, 0.0, STANDARD_GRAVITY]
    angular_velocity = [0.0, 0.0, 0.0]

    measurement = sensor.step(
        acceleration=acceleration,
        angular_velocity=angular_velocity,
    )
    print("=== Single-Step API Demo ===")
    print(f"Timestamp [s]:   {measurement.timestamp_s:.4f}")
    print(f"Accelerometer:   {measurement.accelerometer} m/s^2 (raw LSB: {measurement.raw_accelerometer})")
    print(f"Gyroscope:       {measurement.gyroscope} rad/s (raw LSB: {measurement.raw_gyroscope})")


def run_static_scenario() -> None:
    """Scenario 1: Constant gravity [0, 0, g0] and zero angular rate."""
    sensor = MPU6050.from_config(CONFIG_PATH)
    t = sensor.time_vector(duration_s=10.0)
    n = t.size

    ideal_acc = np.tile(np.array([0.0, 0.0, STANDARD_GRAVITY]), (n, 1))
    ideal_gyr = np.zeros((n, 3), dtype=float)

    result = sensor.simulate(ideal_acc, ideal_gyr)
    csv_path = result.to_csv(OUTPUT_DIR / "scenario1_static.csv")
    plot_imu_comparison(
        result,
        title="Scenario 1: MPU-6050 Static Input (Constant Gravity, Zero Angular Velocity)",
        save_path=OUTPUT_DIR / "scenario1_static.png",
    )
    print(f"\n=== Scenario 1 (Static) saved to {csv_path.name} ===")
    print("Mean accel error [m/s^2]:", np.mean(result.accelerometer - ideal_acc, axis=0))
    print("Std  accel error [m/s^2]:", np.std(result.accelerometer - ideal_acc, axis=0))
    print("Mean gyro  error [rad/s]:", np.mean(result.gyroscope - ideal_gyr, axis=0))
    print("Std  gyro  error [rad/s]:", np.std(result.gyroscope - ideal_gyr, axis=0))


def run_sinusoidal_scenario() -> None:
    """Scenario 2: Pure sinusoidal acceleration and angular velocity inputs."""
    sensor = MPU6050.from_config(CONFIG_PATH)
    t = sensor.time_vector(duration_s=5.0)

    ideal_acc = np.column_stack(
        [
            2.5 * np.sin(2.0 * np.pi * 0.5 * t),
            1.8 * np.cos(2.0 * np.pi * 0.8 * t),
            STANDARD_GRAVITY + 1.2 * np.sin(2.0 * np.pi * 1.2 * t),
        ]
    )
    ideal_gyr = np.column_stack(
        [
            0.6 * np.sin(2.0 * np.pi * 0.6 * t),
            0.4 * np.cos(2.0 * np.pi * 1.0 * t),
            0.8 * np.sin(2.0 * np.pi * 0.4 * t),
        ]
    )

    result = sensor.simulate(ideal_acc, ideal_gyr)
    result.to_csv(OUTPUT_DIR / "scenario2_sinusoidal.csv")
    plot_imu_comparison(
        result,
        title="Scenario 2: MPU-6050 Sinusoidal Input",
        save_path=OUTPUT_DIR / "scenario2_sinusoidal.png",
    )
    print("\n=== Scenario 2 (Sinusoidal) complete ===")
    print("RMSE accel [m/s^2]:", np.sqrt(np.mean((result.accelerometer - ideal_acc) ** 2, axis=0)))
    print("RMSE gyro  [rad/s]:", np.sqrt(np.mean((result.gyroscope - ideal_gyr) ** 2, axis=0)))


def run_multifrequency_scenario() -> None:
    """Scenario 3: Multi-frequency composite acceleration and angular velocity inputs."""
    sensor = MPU6050.from_config(CONFIG_PATH)
    t = sensor.time_vector(duration_s=5.0)

    ideal_acc = np.column_stack(
        [
            2.0 * np.sin(2.0 * np.pi * 0.5 * t) + 0.8 * np.sin(2.0 * np.pi * 3.5 * t),
            1.5 * np.cos(2.0 * np.pi * 0.8 * t) + 0.6 * np.sin(2.0 * np.pi * 5.0 * t),
            STANDARD_GRAVITY
            + 1.0 * np.sin(2.0 * np.pi * 1.0 * t)
            + 0.5 * np.cos(2.0 * np.pi * 7.5 * t),
        ]
    )
    ideal_gyr = np.column_stack(
        [
            0.5 * np.sin(2.0 * np.pi * 0.5 * t) + 0.2 * np.sin(2.0 * np.pi * 4.0 * t),
            0.4 * np.cos(2.0 * np.pi * 1.2 * t) + 0.15 * np.sin(2.0 * np.pi * 6.0 * t),
            0.7 * np.sin(2.0 * np.pi * 0.3 * t) + 0.25 * np.cos(2.0 * np.pi * 8.0 * t),
        ]
    )

    result = sensor.simulate(ideal_acc, ideal_gyr)
    result.to_csv(OUTPUT_DIR / "scenario3_multifrequency.csv")
    plot_imu_comparison(
        result,
        title="Scenario 3: MPU-6050 Multi-Frequency Input",
        save_path=OUTPUT_DIR / "scenario3_multifrequency.png",
    )
    print("\n=== Scenario 3 (Multi-Frequency) complete ===")
    print("RMSE accel [m/s^2]:", np.sqrt(np.mean((result.accelerometer - ideal_acc) ** 2, axis=0)))
    print("RMSE gyro  [rad/s]:", np.sqrt(np.mean((result.gyroscope - ideal_gyr) ** 2, axis=0)))


if __name__ == "__main__":
    run_single_step_demo()
    run_static_scenario()
    run_sinusoidal_scenario()
    run_multifrequency_scenario()
    print(f"\nAll basic simulation plots and CSVs written to {OUTPUT_DIR}")
