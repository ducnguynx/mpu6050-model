# Validation Report

This report documents the quantitative verification of the Python MPU-6050 behavioral simulator (`mpu6050_sim`) against:
1. **Farrell, Silva, Rahman, & Wendel (2022)**, *"Inertial Measurement Unit Error Modeling Tutorial: Inertial Navigation System State Estimation with Real-Time Sensor Calibration,"* *IEEE Control Systems Magazine*, Vol. 42, No. 6, pp. 40–66 (`10.1109/MCS.2022.3209059`).
2. **El-Sheimy, Hou, & Niu (2008)**, *"Analysis and Modeling of Inertial Sensors Using Allan Variance,"* *IEEE Transactions on Instrumentation and Measurement*, Vol. 57, No. 1, pp. 140–149 (`10.1109/TIM.2007.908635`).
3. **InvenSense MPU-6000 and MPU-6050 Product Specification** (`PS-MPU-6000A-00`, Revision 3.4) and **Register Map** (`RM-MPU-6000A-00`, Revision 4.2).

> **Important Scope Statement:** Validation is performed strictly against published analytical stochastic equations, Allan variance theory, and manufacturer specifications. **No experimental validation against physical MPU-6050 hardware logs is claimed.**

---

## 1. Automated Verification Summary

All 20 automated unit and statistical validation tests (`python3 -m unittest discover -s tests -v`) pass in ~1.1 s:

| Test Suite | Tests | Scope Verified | Status |
|---|---|---|---|
| `tests/test_accelerometer.py` | 5 | Deterministic $(\mathbf{I}+\mathbf{S}+\mathbf{M})\mathbf{u}+\mathbf{b}_0$, FSR saturation clamping ($\pm 2/4/8/16\,g$), 16-bit ADC LSB counts (`16384/8192/4096/2048 LSB/g`), discrete white-noise variance, invalid configuration rejection | PASS |
| `tests/test_gyroscope.py` | 5 | Deterministic $(\mathbf{I}+\mathbf{S}+\mathbf{M})\mathbf{u}+\mathbf{b}_0$, FSR saturation clamping ($\pm 250/500/1000/2000\,^\circ/\text{s}$), 16-bit ADC LSB counts (`131/65.5/32.8/16.4 LSB/(°/s)`), $0.05\,^\circ/\text{s-rms}$ datasheet RMS noise at $200\,\text{Hz}$, invalid configuration rejection | PASS |
| `tests/test_sensor.py` | 5 | YAML loading & round-trip serialization, bit-for-bit seed reproducibility, `reset()` restoration, independent accel/gyro RNG streams via `SeedSequence.spawn(2)`, CSV export (19 columns preserving ideal + measured + raw LSB), input shape validation | PASS |
| `tests/test_validation.py` | 5 | Farrell et al. (2022) Gauss–Markov peak constant ($0.43654$ at $\lambda = 1.89$), discrete GM stationary variance preservation, pure white-noise ADEV ($-1/2$ slope), pure random-walk ADEV ($+1/2$ slope), combined full MPU-6050 accelerometer & gyroscope ADEV curves | PASS |

---

## 2. Quantitative Comparison Against Published Specifications & Equations

All metrics below are reproduced directly by running `python3 examples/validate_against_paper.py`.

### 2.1 InvenSense MPU-6050 Datasheet Broadband RMS Noise (`PS-MPU-6000A-00` Rev 3.4)
At sampling rate $f_s = 200\,\text{Hz}$ (Nyquist noise bandwidth $B = f_s / 2 = 100\,\text{Hz}$, matching the MPU-6050 `DLPF_CFG = 2` bandwidth where the datasheet specifies gyroscope Total RMS Noise), simulated over $N = 200{,}000$ stationary samples ($1000\,\text{s}$, seed `123`, white noise isolated):

| Sensor | Datasheet Reference (100 Hz BW) | Simulated Empirical RMS | Relative Error |
|---|---|---|---|
| **Gyroscope** | $0.05000\,^\circ/\text{s-rms}$ ($0.005\,(^\circ/\text{s})/\sqrt{\text{Hz}} \times \sqrt{100}$) | $0.05006\,^\circ/\text{s-rms}$ | **0.119%** |
| **Accelerometer** | $4.00000\,\text{mg-rms}$ ($400\,\mu g/\sqrt{\text{Hz}} \times \sqrt{100}$) | $4.00492\,\text{mg-rms}$ | **0.123%** |

### 2.2 Full MPU-6050 Overlapping Allan Deviation vs. Farrell et al. (2022) Eq. (22)
Stationary 1-hour simulation ($T = 3600\,\text{s}$, $f_s = 50\,\text{Hz}$, $180{,}000$ samples, seed `2022`) using `config/default.yaml` with white noise, first-order Gauss–Markov bias instability, random-walk bias drift, saturation, and 16-bit ADC quantization all enabled simultaneously:

| Axis | MAPE vs. Analytical ($\tau \le 100\,\text{s}$) | Empirical $\hat{\sigma}_A(1\,\text{s})$ | Analytical $\sigma_A(1\,\text{s})$ | Validation Plot |
|---|---|---|---|---|
| **Accelerometer X** | **2.05%** | $2.723 \times 10^{-3}\,\text{m/s}^2$ | $2.698 \times 10^{-3}\,\text{m/s}^2$ | `outputs/validation_accel_allan_deviation.png` |
| **Gyroscope X** | **1.25%** | $6.062 \times 10^{-5}\,\text{rad/s}$ | $6.142 \times 10^{-5}\,\text{rad/s}$ | `outputs/validation_gyro_allan_deviation.png` |

**Note on ADC Quantization Term Verified During Validation:**
Initial inspection revealed an instructive physical distinction: the classic IEEE Std 952 quantization term $\sigma_A^2(\tau) = 3Q^2/\tau^2$ (slope $-1$) applies to sensors outputting *quantized integrated angle/velocity increments* (e.g., optical gyros). Because the MPU-6050's 16-bit ADC quantizes instantaneous *rate and specific-force samples* directly, per-sample quantization behaves as rate-white noise of variance $\Delta_{\mathrm{LSB}}^2/12$, contributing $\sigma_{A,\mathrm{ADC}}^2(\tau) = \frac{\Delta_{\mathrm{LSB}}^2 \Delta t}{12\tau}$. Incorporating this exact rate-ADC relation reduced the gyroscope analytical-vs-empirical discrepancy from 27.5% to **1.25%**.

### 2.3 Farrell et al. (2022) First-Order Gauss–Markov Analytical Peak Check
Pure first-order Gauss–Markov bias process ($T_B = 5.0\,\text{s}$, $B = 1.0 \times 10^{-3}\,\text{rad/s}$, $T = 10{,}000\,\text{s}$ at $f_s = 20\,\text{Hz}$, 3-axis mean Allan variance, seed `777`):

| Metric | Analytical (Farrell et al., 2022) | Simulated Empirical | Relative Error |
|---|---|---|---|
| Peak Allan deviation $\sigma_{A,\max}$ | $0.66428\,B = 6.6428 \times 10^{-4}\,\text{rad/s}$ | $6.7006 \times 10^{-4}\,\text{rad/s}$ | **0.87%** |
| Peak cluster time $\tau_{\max}$ | $1.89\,T_B = 9.45\,\text{s}$ | $8.60\,\text{s}$ (flat hump region) | Within single log-grid bin |

Plot saved at `outputs/validation_gauss_markov_peak.png`.

---

## 3. MATLAB vs. Python Implementation Comparison

Because `jaffarrell/AV-Matlab-SW` is implemented in MATLAB using discrete state-space propagation (`x_k = A_d * x_k + B_d * w_k`, `z_k = C_d * x_k + v_k`) driven by `randn`:
- **Algebraic Equivalence:** Both MATLAB and this Python implementation execute the exact same discrete-time transition matrix $\Phi_B = \exp(-\Delta t / T_B)$, exact discrete process-noise covariance $q_{B,d} = \frac{S_B T_B}{2}(1 - \exp(-2\Delta t / T_B))$, random walk covariance $q_{K,d} = K^2\Delta t$, and white-noise variance $\sigma_{N,d}^2 = N^2 / \Delta t$.
- **Expected Differences:** Sample-by-sample pseudo-random sequences differ between MATLAB's Mersenne Twister (`rng`) and NumPy's PCG64 (`np.random.default_rng`), while all ensemble statistics (mean, variance, power spectral density, and overlapping Allan deviation) agree within standard finite-sample Monte Carlo confidence bounds ($\sim 1\text{–}3\%$).
