# Model Assumptions and Limitations

This document explicitly lists the mathematical assumptions, engineering simplifications, and scope boundaries of the MPU-6050 behavioral simulator derived from **Farrell et al. (2022)** (`10.1109/MCS.2022.3209059`) and **InvenSense `PS-MPU-6000A-00` (Rev 3.4)**.

---

## 1. Mathematical Assumptions Preserved from Farrell et al. (2022)

1. **Additive Linear Error Superposition:**
   Sensor imperfections combine linearly as $\tilde{\mathbf{u}} = (\mathbf{I}_3 + \mathbf{S} + \mathbf{M})\mathbf{u} + \mathbf{b}_0 + \mathbf{z}$ prior to ADC saturation and quantization. Higher-order nonlinearities (e.g., quadratic $0.2\%$ nonlinearity, $g$-sensitivity of the gyroscope, and $g^2$ vibration rectification) are omitted unless explicitly required, matching the linear state-space error framework of Farrell et al. (2022).
2. **Independence of Stochastic Error Processes:**
   The broadband white noise ($v_N$), first-order Gauss–Markov bias instability ($x_B$), and random-walk bias drift ($x_K$) are statistically independent zero-mean Gaussian processes across both error types and across the three sensor axes ($X, Y, Z$).
3. **First-Order Gauss–Markov Approximation to Flicker Noise (Bias Instability):**
   True $1/f$ flicker noise cannot be represented exactly by a finite-dimensional linear time-invariant (LTI) state-space model. Following Farrell et al. (2022), bias instability is approximated by a first-order Gauss–Markov process whose Allan deviation peak ($0.43654\sqrt{S_B T_B}$ at $\tau = 1.89\,T_B$) matches the target flicker floor ($0.66428\,B$). This approximation is exact in its own Allan variance curve (Eq. 8 of `mathematical_model.md`) and accurate around $\tau \approx 1.89\,T_B$, while rolling off with slope $-1/2$ at $\tau \gg T_B$ and $+1/2$ at $\tau \ll T_B$.
4. **Constant-Rate Uniform Sampling:**
   The sensor is sampled uniformly with constant step $\Delta t = 1/f_s$, with zero clock jitter and zero timestamp skew between the accelerometer and gyroscope channels.
5. **Direct Input Injection in Body Frame:**
   The simulator accepts ideal specific force $\mathbf{f}_k$ ($\text{m/s}^2$) and ideal angular velocity $\boldsymbol{\omega}_k$ ($\text{rad/s}$) directly in the MPU-6050 sensor body frame. No rigid-body kinematics or external trajectory dynamics are coupled into the sensor model.

---

## 2. MPU-6050 Hardware Simplifications and Omissions

1. **Constant Ambient Temperature ($T_A = 25^\circ\text{C}$):**
   Thermal sensitivity drift (e.g., datasheet gyroscope ZRO variation over temperature $\pm 20\,^\circ/\text{s}$ across $-40^\circ\text{C}$ to $+85^\circ\text{C}$, and accelerometer zero-$g$ temperature coefficient) is not dynamically simulated. Static bias and scale-factor parameters represent fixed operating conditions at nominal temperature ($25^\circ\text{C}$).
2. **No On-Chip Digital Low-Pass Filter (DLPF) Transfer Function:**
   In accordance with project scope limitations ("Do NOT implement Kalman filters, Wiener filters, moving average filters, or other signal-processing algorithms"), the MPU-6050's internal configurable DLPF (`CONFIG` register `DLPF_CFG`) is not implemented as a digital filter stage. Instead, broadband noise density is mapped directly to discrete white noise at the configured sampling frequency $f_s$ (with Nyquist bandwidth $f_s / 2$).
3. **No Internal MEMS Mechanical Structure or Dynamics:**
   Proof-mass spring-damper resonance, Coriolis vibrating-structure dynamics, and package mechanical vibration modes are excluded. The sensor is treated purely at the behavioral signal level.
4. **No I²C Bus, Register State Machine, FIFO, or DMP Simulation:**
   This is a mathematical behavioral simulator, not a hardware-level digital twin. I²C communication timing, register read/write transactions, FIFO overflow, interrupts, and the on-chip Digital Motion Processor (DMP) are outside scope.
5. **Unmodeled Non-Linearity ($0.2\%$) and Gyroscope $g$-Sensitivity:**
   The MPU-6050 datasheet lists a typical gyroscope nonlinearity of $0.2\%$ (best-fit straight line). Consistent with Farrell et al. (2022)'s linear error formulation, nonlinear distortion is not included in the default equations.

---

## 3. Validation Boundaries

- **Analytical and Simulation Validation Only:** Validation verifies the Python implementation against the exact analytical stochastic equations and Allan variance signatures published in **Farrell et al. (2022)** and **El-Sheimy et al. (2008)**, as well as the numerical specifications in **InvenSense `PS-MPU-6000A-00` Rev 3.4**.
- **No Physical Hardware Validation Claimed:** Because no physical MPU-6050 hardware logs are collected in this repository, no experimental hardware validation is claimed. Real physical MPU-6050 chips exhibit unit-to-unit variations in static bias, scale factor, temperature sensitivity, and axis-dependent Allan deviation parameters (as documented in Gonzalez & Dabove 2019 and Rudyk et al. 2020), all of which can be entered directly into YAML configuration files when hardware data are available.
