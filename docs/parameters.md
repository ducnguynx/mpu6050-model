# Traceable Parameter Database

This document catalogs every configurable parameter in `config/default.yaml`, including its symbol, SI unit, default value, valid range, and provenance. Parameters obtained from the InvenSense MPU-6050 Product Specification (`PS-MPU-6000A-00`, Rev 3.4) or Register Map (`RM-MPU-6000A-00`, Rev 4.2) are explicitly distinguished from **configurable assumptions** (parameters not specified in the manufacturer datasheet that must be set from experimental characterization or user requirements).

Standard gravity constant used throughout: $g_0 = 9.80665\,\text{m/s}^2$ (CGPM standard gravity).

---

## 1. Simulation Configuration (`simulation`)

| YAML Key | Symbol | Unit | Default | Provenance | Description |
|---|---|---|---|---|---|
| `sample_rate_hz` | $f_s$ | $\text{Hz}$ | `200.0` | **Configurable assumption** (within datasheet ODR range $4\text{–}1000\,\text{Hz}$ accel / $4\text{–}8000\,\text{Hz}$ gyro, `PS-MPU-6000A-00` §6.1–6.2) | Discrete sampling frequency. Chosen as $200\,\text{Hz}$ because its Nyquist bandwidth ($100\,\text{Hz}$) matches the MPU-6050 `DLPF_CFG = 2` bandwidth where the datasheet specifies gyroscope Total RMS Noise ($0.05\,^\circ/\text{s-rms}$). |
| `duration_s` | $T_{\mathrm{sim}}$ | $\text{s}$ | `10.0` | **Configurable assumption** | Default simulation duration when generating time vectors. |
| `random_seed` | — | integer | `42` | **Configurable assumption** | Seed for NumPy `default_rng` to guarantee exact bit-for-bit reproducibility. |
| `stationary_bias_init` | — | boolean | `true` | **Farrell et al. (2022)** | If `true`, initializes the first-order Gauss–Markov bias state from its stationary distribution $\mathcal{N}(0, \sigma_{\mathrm{gm}}^2)$; if `false`, initializes at `0.0`. |

---

## 2. MPU-6050 Hardware Lookup Tables (Datasheet-Sourced)

All values below are directly from **InvenSense `PS-MPU-6000A-00` Rev 3.4, Tables 6.1 & 6.2** and **`RM-MPU-6000A-00` Rev 4.2**:

### 2.1 Accelerometer Full-Scale Ranges (`AFS_SEL`)

| `AFS_SEL` | `full_scale_range_g` | FSR in SI ($\text{m/s}^2$) | Datasheet Sensitivity ($\text{LSB}/g$) | Sensitivity in SI ($\text{LSB}/(\text{m/s}^2)$) | Quantization Step $\Delta_{\mathrm{LSB}}$ ($\text{m/s}^2$) |
|---|---|---|---|---|---|
| `0` | `2` ($\pm 2g$) | $\pm 19.61330$ | `16384` | `1670.69896` | $5.98552 \times 10^{-4}$ |
| `1` | `4` ($\pm 4g$) | $\pm 39.22660$ | `8192` | `835.34948` | $1.19710 \times 10^{-3}$ |
| `2` | `8` ($\pm 8g$) | $\pm 78.45320$ | `4096` | `417.67474` | $2.39421 \times 10^{-3}$ |
| `3` | `16` ($\pm 16g$) | $\pm 156.90640$ | `2048` | `208.83737` | $4.78841 \times 10^{-3}$ |

### 2.2 Gyroscope Full-Scale Ranges (`FS_SEL`)

| `FS_SEL` | `full_scale_range_dps` | FSR in SI ($\text{rad/s}$) | Datasheet Sensitivity ($\text{LSB}/(^\circ/\text{s})$) | Sensitivity in SI ($\text{LSB}/(\text{rad/s})$) | Quantization Step $\Delta_{\mathrm{LSB}}$ ($\text{rad/s}$) |
|---|---|---|---|---|---|
| `0` | `250` ($\pm 250\,^\circ/\text{s}$) | $\pm 4.36332$ | `131.0` | `7505.74713` | $1.33231 \times 10^{-4}$ |
| `1` | `500` ($\pm 500\,^\circ/\text{s}$) | $\pm 8.72665$ | `65.5` | `3752.87357` | $2.66462 \times 10^{-4}$ |
| `2` | `1000` ($\pm 1000\,^\circ/\text{s}$) | $\pm 17.45329$ | `32.8` | `1879.30164` | $5.32113 \times 10^{-4}$ |
| `3` | `2000` ($\pm 2000\,^\circ/\text{s}$) | $\pm 34.90659$ | `16.4` | `939.65082` | $1.06423 \times 10^{-3}$ |

- **ADC Resolution:** 16-bit signed two's complement (`[-32768, 32767]`), `PS-MPU-6000A-00` §6.1–6.2.

---

## 3. Accelerometer Parameters (`accelerometer`)

| YAML Key | Symbol | Unit | Default | Provenance & Datasheet Bound | Description |
|---|---|---|---|---|---|
| `full_scale_range_g` | $f_{\mathrm{FSR}}$ | $g$ | `2` | **Datasheet-sourced** (`PS-MPU-6000A-00` §6.2: $\pm 2, \pm 4, \pm 8, \pm 16\,g$) | Programmable accelerometer full-scale range. |
| `scale_factor_error` | $[s_{a,x}, s_{a,y}, s_{a,z}]$ | dimensionless | `[0.005, -0.004, 0.006]` | **Configurable assumption** within **Datasheet-sourced** tolerance bound $\pm 3\%$ ($\pm 0.03$) at $25^\circ\text{C}$ (`PS-MPU-6000A-00` §6.2) | Per-axis fractional scale-factor error (typical sample unit within $\pm 3\%$ manufacturing tolerance). |
| `misalignment` | $\mathbf{M}_a$ | dimensionless ($3\times 3$) | `[[0, 0.005, -0.003], [-0.004, 0, 0.005], [0.003, -0.005, 0]]` | **Configurable assumption** within **Datasheet-sourced** cross-axis sensitivity bound $\pm 2\%$ ($\pm 0.02$) (`PS-MPU-6000A-00` §6.2) | Off-diagonal cross-axis sensitivity / non-orthogonality matrix ($M_{ii} = 0$). |
| `static_bias_mps2` | $\mathbf{b}_{0,a}$ | $\text{m/s}^2$ | `[0.15, -0.12, 0.25]` | **Configurable assumption** within **Datasheet-sourced** initial zero-$g$ tolerance $\pm 50\,\text{mg}$ ($\pm 0.490\,\text{m/s}^2$) X/Y and $\pm 80\,\text{mg}$ ($\pm 0.785\,\text{m/s}^2$) Z (`PS-MPU-6000A-00` §6.2) | Per-axis constant turn-on bias. |
| `noise_density_mps2_per_sqrthz` | $[N_{a,x}, N_{a,y}, N_{a,z}]$ | $(\text{m/s}^2)/\sqrt{\text{Hz}}$ (two-sided VRW) | `[0.00277374, 0.00277374, 0.00277374]` | **Derived directly from Datasheet** (`PS-MPU-6000A-00` §6.2): $\mathrm{NSD}_{1\mathrm{s}} = 400\,\mu g/\sqrt{\text{Hz}} = 3.92266 \times 10^{-3}\,(\text{m/s}^2)/\sqrt{\text{Hz}}$, converted via Eq. (5): $N_a = \mathrm{NSD}_{1\mathrm{s}}/\sqrt{2} = 2.77374 \times 10^{-3}$ | Velocity Random Walk (VRW) coefficient governing broadband white noise ($\sigma_{d,a} = N_a \sqrt{f_s}$). |
| `bias_instability_mps2` | $[B_{a,x}, B_{a,y}, B_{a,z}]$ | $\text{m/s}^2$ | `[0.0015, 0.0015, 0.0020]` | **Configurable assumption** (order of magnitude consistent with low-cost MPU-6000/6050 Allan variance characterization in Gonzalez & Dabove 2019; not published in datasheet) | Bias instability coefficient $B_a$ mapped to first-order Gauss–Markov driving PSD $S_B$ via Eq. (10). |
| `bias_correlation_time_s` | $[T_{B,a,x}, T_{B,a,y}, T_{B,a,z}]$ | $\text{s}$ | `[100.0, 100.0, 100.0]` | **Configurable assumption** (typical consumer MEMS correlation timescale $10\text{–}300\,\text{s}$ per Farrell et al. 2022 & Gonzalez & Dabove 2019; not published in datasheet) | First-order Gauss–Markov correlation time constant $T_{B,a}$. |
| `random_walk_mps2_per_sqrts` | $[K_{a,x}, K_{a,y}, K_{a,z}]$ | $(\text{m/s}^2)/\sqrt{\text{s}}$ | `[0.0001, 0.0001, 0.00015]` | **Configurable assumption** (not published in datasheet; can be set to `0.0` to disable long-term random walk drift) | Acceleration Random Walk (bias drift) coefficient $K_a$. |
| `enable_saturation` | — | boolean | `true` | **Datasheet-sourced** hardware behavior | Enables $\pm f_{\mathrm{FSR}}$ clipping. |
| `enable_quantization` | — | boolean | `true` | **Datasheet-sourced** hardware behavior (16-bit ADC) | Enables 16-bit signed integer LSB quantization. |

---

## 4. Gyroscope Parameters (`gyroscope`)

| YAML Key | Symbol | Unit | Default | Provenance & Datasheet Bound | Description |
|---|---|---|---|---|---|
| `full_scale_range_dps` | $\omega_{\mathrm{FSR}}$ | $^\circ/\text{s}$ | `250` | **Datasheet-sourced** (`PS-MPU-6000A-00` §6.1: $\pm 250, \pm 500, \pm 1000, \pm 2000\,^\circ/\text{s}$) | Programmable gyroscope full-scale range. |
| `scale_factor_error` | $[s_{g,x}, s_{g,y}, s_{g,z}]$ | dimensionless | `[0.008, -0.006, 0.007]` | **Configurable assumption** within **Datasheet-sourced** tolerance bound $\pm 3\%$ ($\pm 0.03$) at $25^\circ\text{C}$ (`PS-MPU-6000A-00` §6.1) | Per-axis fractional scale-factor error. |
| `misalignment` | $\mathbf{M}_g$ | dimensionless ($3\times 3$) | `[[0, 0.004, -0.003], [-0.005, 0, 0.004], [0.003, -0.004, 0]]` | **Configurable assumption** within **Datasheet-sourced** cross-axis sensitivity bound $\pm 2\%$ ($\pm 0.02$) (`PS-MPU-6000A-00` §6.1) | Off-diagonal cross-axis sensitivity / non-orthogonality matrix ($M_{ii} = 0$). |
| `static_bias_radps` | $\mathbf{b}_{0,g}$ | $\text{rad/s}$ | `[0.015, -0.010, 0.018]` | **Configurable assumption** within **Datasheet-sourced** initial zero-rate output tolerance $\pm 20\,^\circ/\text{s}$ ($\pm 0.349\,\text{rad/s}$) at $25^\circ\text{C}$ (`PS-MPU-6000A-00` §6.1) | Per-axis constant turn-on bias (default $\approx 0.6\text{–}1.0\,^\circ/\text{s}$, representative of an uncalibrated room-temperature MPU-6050). |
| `noise_density_radps_per_sqrthz` | $[N_{g,x}, N_{g,y}, N_{g,z}]$ | $(\text{rad/s})/\sqrt{\text{Hz}}$ (two-sided ARW) | `[6.170671e-5, 6.170671e-5, 6.170671e-5]` | **Derived directly from Datasheet** (`PS-MPU-6000A-00` §6.1): $\mathrm{NSD}_{1\mathrm{s}} = 0.005\,(^\circ/\text{s})/\sqrt{\text{Hz}} = 8.726646 \times 10^{-5}\,(\text{rad/s})/\sqrt{\text{Hz}}$, converted via Eq. (5): $N_g = \mathrm{NSD}_{1\mathrm{s}}/\sqrt{2} = 6.170671 \times 10^{-5}$ | Angle Random Walk (ARW) coefficient governing broadband white noise ($\sigma_{d,g} = N_g \sqrt{f_s}$). At $f_s = 200\,\text{Hz}$, produces exactly $0.05\,^\circ/\text{s-rms}$ matching datasheet Total RMS Noise. |
| `bias_instability_radps` | $[B_{g,x}, B_{g,y}, B_{g,z}]$ | $\text{rad/s}$ | `[1.2e-4, 6.0e-5, 1.2e-4]` | **Configurable assumption** (order of magnitude consistent with MPU-6050/6000 Allan variance tests; reflects Rudyk et al. 2020 observation that X and Z axes exhibit roughly $2\times$ the bias instability of the Y axis; not published in datasheet) | Bias instability coefficient $B_g$ mapped to first-order Gauss–Markov driving PSD $S_B$ via Eq. (10). |
| `bias_correlation_time_s` | $[T_{B,g,x}, T_{B,g,y}, T_{B,g,z}]$ | $\text{s}$ | `[100.0, 100.0, 100.0]` | **Configurable assumption** (typical MEMS correlation time per Farrell et al. 2022 & Gonzalez & Dabove 2019; not published in datasheet) | First-order Gauss–Markov correlation time constant $T_{B,g}$. |
| `random_walk_radps_per_sqrts` | $[K_{g,x}, K_{g,y}, K_{g,z}]$ | $(\text{rad/s})/\sqrt{\text{s}}$ | `[1.0e-5, 1.0e-5, 1.0e-5]` | **Configurable assumption** (not published in datasheet; Rudyk et al. 2020 noted negligible RRW over 1 hr static tests, so users may set to `0.0` to study white + GM only) | Rate Random Walk (RRW) bias drift coefficient $K_g$. |
| `enable_saturation` | — | boolean | `true` | **Datasheet-sourced** hardware behavior | Enables $\pm \omega_{\mathrm{FSR}}$ clipping. |
| `enable_quantization` | — | boolean | `true` | **Datasheet-sourced** hardware behavior (16-bit ADC) | Enables 16-bit signed integer LSB quantization. |

---

## 5. Summary of Provenance Separation

- **Strictly Datasheet-Sourced (Zero Assumption):**
  - Accelerometer FSRs ($\pm 2/4/8/16\,g$) and nominal sensitivities ($16384/8192/4096/2048\,\text{LSB}/g$).
  - Gyroscope FSRs ($\pm 250/500/1000/2000\,^\circ/\text{s}$) and nominal sensitivities ($131/65.5/32.8/16.4\,\text{LSB}/(^\circ/\text{s})$).
  - 16-bit signed two's-complement ADC range (`[-32768, 32767]`).
  - Accelerometer white noise VRW derived directly from $400\,\mu g/\sqrt{\text{Hz}}$.
  - Gyroscope white noise ARW derived directly from $0.005\,(^\circ/\text{s})/\sqrt{\text{Hz}}$ (and cross-checked against $0.05\,^\circ/\text{s-rms}$ at $100\,\text{Hz}$ BW).
  - Upper tolerance bounds for scale-factor error ($\pm 3\%$), cross-axis sensitivity ($\pm 2\%$), zero-$g$ bias ($\pm 50/80\,\text{mg}$), and zero-rate bias ($\pm 20\,^\circ/\text{s}$).
- **Configurable Assumptions (Explicitly Labeled):**
  - Specific per-axis realization values of scale-factor error, cross-axis misalignment, and static turn-on bias (chosen well inside datasheet manufacturing bounds).
  - Low-frequency stochastic parameters not published in the InvenSense datasheet: bias instability ($B_a, B_g$), Gauss–Markov correlation time ($T_{B,a}, T_{B,g}$), and rate/acceleration random walk ($K_a, K_g$).
  - Sampling frequency ($200\,\text{Hz}$), simulation duration ($10\,\text{s}$), and random seed (`42`).
