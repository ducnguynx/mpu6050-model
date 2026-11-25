# Literature Review and References

This document records the academic papers, IEEE standards, and official manufacturer documentation reviewed to select and parameterize the mathematical behavioral model of the InvenSense MPU-6050 6-axis Inertial Measurement Unit (IMU).

---

## 1. Selected Primary IEEE Reference

### Farrell, Silva, Rahman, & Wendel (2022)
- **Title:** *Inertial Measurement Unit Error Modeling Tutorial: Inertial Navigation System State Estimation with Real-Time Sensor Calibration*
- **Authors:** Jay A. Farrell, Felipe O. Silva, Farzana Rahman, and Jan Wendel
- **Publication:** *IEEE Control Systems Magazine*, Vol. 42, No. 6, pp. 40–66, December 2022
- **DOI:** `10.1109/MCS.2022.3209059`
- **URL:** https://doi.org/10.1109/MCS.2022.3209059
- **Public MATLAB Companion Software:** https://github.com/jaffarrell/AV-Matlab-SW (`AV_Matlab_SW_IEEE_CSM/`)
- **Model Scope:**
  - Unified continuous-time and exact discrete-time behavioral error model for both 3-axis accelerometers and 3-axis gyroscopes.
  - Explicit deterministic error terms: scale-factor error, cross-axis / non-orthogonality misalignment, and constant turn-on bias.
  - Explicit stochastic error terms: broadband white measurement noise (ARW/VRW), first-order Gauss–Markov process for time-correlated bias instability ($T_B, S_B$), and rate/acceleration random walk ($K$).
  - Exact continuous-to-discrete state-space conversion ($\mathbf{A}_d = \exp(\mathbf{F}\Delta t)$, $\mathbf{Q}_d = \int_0^{\Delta t} \exp(\mathbf{F}\tau)\mathbf{G}\mathbf{Q}_c\mathbf{G}^T\exp(\mathbf{F}^T\tau)\,d\tau$).
  - Exact analytical Allan Variance (AV) / Allan Standard Deviation (ASD) formulas linking time-series stochastic simulation outputs back to model parameters.
- **MATLAB Implementation Availability:** Yes. The authors maintain an open-source MATLAB repository (`jaffarrell/AV-Matlab-SW`) implementing continuous-to-discrete state-space conversion, discrete-time stochastic error sequence generation ($x_{k+1} = A_d x_k + w_k$, $z_k = C_d x_k + v_k$), and Allan variance verification routines.

---

## 2. Supporting IEEE Literature and Standards

### El-Sheimy, Hou, & Niu (2008)
- **Title:** *Analysis and Modeling of Inertial Sensors Using Allan Variance*
- **Authors:** Naser El-Sheimy, Haiying Hou, and Xiaoji Niu
- **Publication:** *IEEE Transactions on Instrumentation and Measurement*, Vol. 57, No. 1, pp. 140–149, January 2008
- **DOI:** `10.1109/TIM.2007.908635`
- **URL:** https://doi.org/10.1109/TIM.2007.908635
- **Role in Project:** Foundational IEEE benchmark establishing the overlapping Allan variance estimator and analytical log-log slope signatures for Quantization Noise ($Q$, slope $-1$), Angle/Velocity Random Walk ($N$, slope $-1/2$), Bias Instability ($B$, slope $0$, $\sigma_A \approx 0.664 B$), Rate/Acceleration Random Walk ($K$, slope $+1/2$), and Gauss–Markov exponentially correlated noise (peak at $\tau = 1.89\,T_c$). Used for validation of the simulated stochastic processes.
- **MATLAB Implementation Availability:** Standard overlapping Allan variance algorithms are widely implemented in MATLAB (e.g., `allanvar` in Navigation Toolbox) and replicated in Farrell et al. (2022).

### IEEE Std 952-1997
- **Title:** *IEEE Standard Specification Format Guide and Test Procedure for Single-Axis Interferometric Fiber Optic Gyros*
- **Organization:** Institute of Electrical and Electronics Engineers (IEEE)
- **Publication Year:** 1998
- **DOI:** `10.1109/IEEESTD.1998.86153`
- **URL:** https://doi.org/10.1109/IEEESTD.1998.86153
- **Role in Project:** Defines the canonical Annex C Allan variance spectral-to-time-domain integral $\sigma^2(\tau) = 4 \int_0^\infty S_\omega(f) \frac{\sin^4(\pi f \tau)}{(\pi f \tau)^2}\,df$ and standardized definitions of random walk ($N$), bias instability ($B$), rate random walk ($K$), and quantization noise ($Q$).

### Li, Yu, Zheng, & Mourikis (2014)
- **Title:** *High-Fidelity Sensor Modeling and Self-Calibration in Vision-Aided Inertial Navigation*
- **Authors:** Mingyang Li, Hongsheng Yu, Xing Zheng, and Anastasios I. Mourikis
- **Publication:** *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)*, pp. 409–416, May 2014
- **DOI:** `10.1109/ICRA.2014.6906889`
- **URL:** https://doi.org/10.1109/ICRA.2014.6906889
- **Role in Project:** Reviewed as a candidate IEEE paper applying an IMU measurement model (scale factor, misalignment, bias random walk, white noise) to an InvenSense MPU-6050. Not chosen as primary because its focus is online vision-aided EKF self-calibration rather than standalone IMU behavioral simulation with MATLAB source.

---

## 3. Supporting MEMS / MPU-6050 Characterization Studies

### Gonzalez, Giribet, & Patiño (2015) / Gonzalez & Dabove (2019)
- **Paper 1:** R. Gonzalez, J. I. Giribet, and H. D. Patiño, *"NaveGo: A Simulation Framework for Low-Cost Integrated Navigation Systems,"* *Journal of Control Engineering and Applied Informatics*, Vol. 17, No. 2, pp. 110–120, 2015.
- **Paper 2:** R. Gonzalez and P. Dabove, *"Performance Assessment of an Ultra Low-Cost Inertial Measurement Unit for Ground Vehicle Navigation,"* *Sensors* (MDPI), Vol. 19, No. 18, Art. no. 3865, 2019.
- **DOI (2019):** `10.3390/s19183865`
- **URL:** https://doi.org/10.3390/s19183865
- **Public MATLAB Software:** NaveGo (`https://github.com/rodralez/navego`)
- **Role in Project:** Characterizes the InvenSense MPU-6000 (which shares the identical MEMS die and inertial specifications as the MPU-6050, differing only in SPI bus support) using Allan variance and a first-order Gauss–Markov + white noise + deterministic bias/scale-factor model consistent with Farrell et al. (2022).

### Rudyk, Semenov, Kryvinska, Semenova, Kvasnikov, & Safonyk (2020)
- **Title:** *Strapdown Inertial Navigation Systems for Positioning Mobile Robots—MEMS Gyroscopes Random Errors Analysis Using Allan Variance Method*
- **Authors:** Andrii V. Rudyk, Andriy O. Semenov, Natalia Kryvinska, Olena O. Semenova, Volodymyr P. Kvasnikov, and Andrii P. Safonyk
- **Publication:** *Sensors* (MDPI), Vol. 20, No. 17, Art. no. 4841, 2020
- **DOI:** `10.3390/s20174841`
- **URL:** https://doi.org/10.3390/s20174841
- **Role in Project:** Direct experimental Allan variance study of the InvenSense MPU-6050 3-axis gyroscope confirming white noise (ARW) and bias instability as dominant stochastic regimes, with observable axis-to-axis variation (X and Z axes exhibiting roughly twice the bias instability of the Y axis).

---

## 4. Official InvenSense / TDK Hardware Documentation

### MPU-6000 and MPU-6050 Product Specification (Rev 3.4)
- **Title:** *MPU-6000 and MPU-6050 Product Specification*
- **Document Number:** `PS-MPU-6000A-00`, Revision 3.4
- **Publisher:** InvenSense Inc. (now TDK InvenSense)
- **Release Date:** August 19, 2013
- **Role in Project:** Primary authoritative source for MPU-6050 hardware parameters: programmable full-scale ranges (FSR), 16-bit ADC word length, nominal sensitivity scale factors ($\text{LSB}/g$ and $\text{LSB}/(^\circ/\text{s})$), initial sensitivity tolerance ($\pm 3\%$), cross-axis sensitivity ($\pm 2\%$), nonlinearity ($0.2\%$), initial zero-$g$ tolerance ($\pm 50\,\text{mg}$ X/Y, $\pm 80\,\text{mg}$ Z), initial zero-rate tolerance ($\pm 20\,^\circ/\text{s}$), accelerometer noise PSD ($400\,\mu g/\sqrt{\text{Hz}}$), gyroscope rate noise spectral density ($0.005\,(^\circ/\text{s})/\sqrt{\text{Hz}}$), gyroscope total RMS noise ($0.05\,^\circ/\text{s-rms}$ at $100\,\text{Hz}$ BW), and output data rate ranges.

### MPU-6000 and MPU-6050 Register Map and Descriptions (Rev 4.2)
- **Title:** *MPU-6000 and MPU-6050 Register Map and Descriptions*
- **Document Number:** `RM-MPU-6000A-00`, Revision 4.2
- **Publisher:** InvenSense Inc. (now TDK InvenSense)
- **Release Date:** August 19, 2013
- **Role in Project:** Authoritative reference for `SMPLRT_DIV` (Register 25), `CONFIG` / `DLPF_CFG` (Register 26), `GYRO_CONFIG` / `FS_SEL` (Register 27), `ACCEL_CONFIG` / `AFS_SEL` (Register 28), and the 16-bit signed two's-complement ADC output format (`[-32768, 32767]`).

---

## 5. Comparison of Candidate Models and Selection Justification

| Criterion | Farrell et al. (2022) [IEEE CSM] | El-Sheimy et al. (2008) [IEEE TIM] | Li et al. (2014) [IEEE ICRA] | Gonzalez & Dabove (2019) [Sensors] |
|---|---|---|---|---|
| **IEEE Peer-Reviewed** | Yes (IEEE Control Systems Magazine) | Yes (IEEE Trans. Instrum. Meas.) | Yes (IEEE ICRA) | No (MDPI Sensors) |
| **Accelerometer & Gyroscope Models** | Both, unified formulation | Both (analysis focus) | Both (coupled with camera EKF) | Both (MPU-6000 via NaveGo) |
| **Explicit Discrete-Time Simulation Equations** | Yes (exact $A_d, Q_d$ state-space) | Partial (focuses on AV extraction) | Partial (filter measurement model) | Yes (NaveGo MATLAB toolbox) |
| **Deterministic + Stochastic Errors** | Yes (scale factor, misalignment, bias, white noise, Gauss–Markov, random walk) | Stochastic only (AV decomposition) | Yes (calibration focus) | Yes |
| **Public MATLAB Source Code** | Yes (`jaffarrell/AV-Matlab-SW`) | No official repo | No | Yes (`rodralez/navego`) |
| **Decoupled from State Estimation / Filtering** | Yes (standalone forward sensor model + AV verification) | Yes | No (tightly coupled to VINS) | Coupled to 15-state INS/GNSS EKF |

### Selection Decision
**Farrell et al. (2022)** (`10.1109/MCS.2022.3209059`) is selected as the **primary mathematical behavioral model**, parameterized with official **InvenSense MPU-6050 Product Specification Rev 3.4** hardware parameters and cross-checked with **El-Sheimy et al. (2008)** (`10.1109/TIM.2007.908635`) and **IEEE Std 952-1997** analytical Allan variance equations.

This combination satisfies every project requirement:
1. It uses an authoritative, peer-reviewed **IEEE** paper with explicit continuous-time and exact discrete-time behavioral equations for both accelerometer and gyroscope.
2. It provides a publicly documented **MATLAB simulation architecture** (`jaffarrell/AV-Matlab-SW`) that maps directly to modular Python classes.
3. It requires **zero Kalman filters or signal-processing algorithms** inside the simulator itself.
4. Its outputs can be quantitatively validated against closed-form analytical Allan variance curves and InvenSense MPU-6050 hardware specifications.
