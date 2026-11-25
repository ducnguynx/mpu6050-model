"""Visualization utilities for MPU-6050 behavioral simulation and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

# Use headless Agg backend when no interactive display is available
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .sensor import SimulationResult

AXIS_NAMES = ("X", "Y", "Z")
AXIS_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")


def plot_sensor_axes(
    time_s: np.ndarray,
    ideal: np.ndarray,
    simulated: np.ndarray,
    *,
    sensor_name: str,
    unit: str,
    title: str,
    config_note: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot 3 individual axes comparing ideal input against simulated sensor output."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(
            time_s,
            simulated[:, idx],
            color=AXIS_COLORS[idx],
            alpha=0.75,
            linewidth=0.9,
            label=f"Simulated {AXIS_NAMES[idx]}",
        )
        ax.plot(
            time_s,
            ideal[:, idx],
            color="#111111",
            linestyle="--",
            linewidth=1.5,
            label=f"Ideal {AXIS_NAMES[idx]}",
        )
        ax.set_ylabel(f"{AXIS_NAMES[idx]}-axis [{unit}]")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time [s]")
    full_title = title if not config_note else f"{title}\n{config_note}"
    fig.suptitle(full_title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def plot_imu_comparison(
    result: SimulationResult,
    *,
    title: str = "MPU-6050 Simulated vs. Ideal Measurements",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot side-by-side 3-axis accelerometer and gyroscope comparisons."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    for idx in range(3):
        # Accelerometer column
        ax_a = axes[idx, 0]
        ax_a.plot(
            result.time_s,
            result.accelerometer[:, idx],
            color=AXIS_COLORS[idx],
            alpha=0.75,
            linewidth=0.85,
            label=f"Simulated Accel {AXIS_NAMES[idx]}",
        )
        ax_a.plot(
            result.time_s,
            result.ideal_acceleration[:, idx],
            color="#111111",
            linestyle="--",
            linewidth=1.4,
            label=f"Ideal Accel {AXIS_NAMES[idx]}",
        )
        ax_a.set_ylabel(f"{AXIS_NAMES[idx]} [m/s²]")
        ax_a.grid(True, linestyle=":", alpha=0.6)
        ax_a.legend(loc="upper right", fontsize=8)

        # Gyroscope column
        ax_g = axes[idx, 1]
        ax_g.plot(
            result.time_s,
            result.gyroscope[:, idx],
            color=AXIS_COLORS[idx],
            alpha=0.75,
            linewidth=0.85,
            label=f"Simulated Gyro {AXIS_NAMES[idx]}",
        )
        ax_g.plot(
            result.time_s,
            result.ideal_angular_velocity[:, idx],
            color="#111111",
            linestyle="--",
            linewidth=1.4,
            label=f"Ideal Gyro {AXIS_NAMES[idx]}",
        )
        ax_g.set_ylabel(f"{AXIS_NAMES[idx]} [rad/s]")
        ax_g.grid(True, linestyle=":", alpha=0.6)
        ax_g.legend(loc="upper right", fontsize=8)

    axes[0, 0].set_title("Accelerometer (Specific Force)", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("Gyroscope (Angular Velocity)", fontsize=10, fontweight="bold")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")

    cfg = result.config
    subtitle = (
        f"fs = {cfg.simulation.sample_rate_hz:.0f} Hz | "
        f"Accel FSR = ±{cfg.accelerometer.full_scale_range_g} g | "
        f"Gyro FSR = ±{cfg.gyroscope.full_scale_range_dps} °/s | "
        f"Seed = {cfg.simulation.random_seed}"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def plot_parameter_sweep(
    time_s: np.ndarray,
    ideal_axis: np.ndarray,
    sweeps: Sequence[tuple[str, np.ndarray]],
    *,
    ylabel: str,
    title: str,
    config_note: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot multiple parameter configurations against ground truth on a single axis."""
    fig, (ax_sig, ax_err) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (label, series) in enumerate(sweeps):
        color = colors[idx % len(colors)]
        ax_sig.plot(time_s, series, color=color, alpha=0.8, linewidth=0.95, label=label)
        ax_err.plot(
            time_s,
            series - ideal_axis,
            color=color,
            alpha=0.8,
            linewidth=0.9,
            label=f"Error: {label}",
        )

    ax_sig.plot(
        time_s,
        ideal_axis,
        color="#111111",
        linestyle="--",
        linewidth=1.5,
        label="Ideal input",
    )
    ax_sig.set_ylabel(ylabel)
    ax_sig.grid(True, linestyle=":", alpha=0.6)
    ax_sig.legend(loc="upper right", fontsize=8.5)

    ax_err.axhline(0.0, color="#111111", linestyle="--", linewidth=1.0)
    ax_err.set_ylabel(f"Output - Ideal [{ylabel.split('[')[-1]}" if "[" in ylabel else "Error")
    ax_err.set_xlabel("Time [s]")
    ax_err.grid(True, linestyle=":", alpha=0.6)
    ax_err.legend(loc="upper right", fontsize=8.5)

    full_title = title if not config_note else f"{title}\n{config_note}"
    fig.suptitle(full_title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig


def plot_allan_deviation(
    tau: np.ndarray,
    empirical_adev: np.ndarray,
    theoretical_components: Mapping[str, np.ndarray],
    *,
    unit: str,
    title: str,
    config_note: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot log-log empirical vs. analytical Allan deviation (Farrell 2022; El-Sheimy 2008)."""
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.loglog(
        tau,
        empirical_adev,
        color="#1f77b4",
        linewidth=2.0,
        label="Simulated (Empirical Overlapping ADEV)",
    )
    ax.loglog(
        tau,
        np.sqrt(theoretical_components["total"]),
        color="#111111",
        linestyle="--",
        linewidth=1.8,
        label="Analytical Total (Farrell 2022 / El-Sheimy 2008)",
    )

    component_styles = [
        ("white_noise", "White Noise (ARW/VRW, slope -1/2)", "#2ca02c", ":"),
        ("gauss_markov", "Gauss-Markov Bias Instability", "#ff7f0e", "-."),
        ("random_walk", "Bias Random Walk (slope +1/2)", "#d62728", ":"),
        ("quantization", "16-bit ADC Quantization (rate-white)", "#9467bd", ":"),
    ]
    for key, label, color, style in component_styles:
        vals = np.sqrt(theoretical_components[key])
        if np.any(vals > 0.0):
            ax.loglog(tau, vals, color=color, linestyle=style, linewidth=1.2, alpha=0.85, label=label)

    ax.set_xlabel(r"Averaging Cluster Time $\tau$ [s]")
    ax.set_ylabel(rf"Allan Deviation $\sigma_A(\tau)$ [{unit}]")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8.5)

    full_title = title if not config_note else f"{title}\n{config_note}"
    ax.set_title(full_title, fontsize=10.5, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
    return fig
