# Mathematical Behavioral Model of the MPU-6050

This document specifies the exact mathematical equations implemented in `mpu6050_sim`. The behavioral error formulation follows **Farrell, Silva, Rahman, & Wendel (2022)** (*IEEE Control Systems Magazine*, DOI: `10.1109/MCS.2022.3209059`), augmented with the hardware output stage (full-scale range saturation and 16-bit signed ADC quantization) from the **InvenSense MPU-6000/MPU-6050 Product Specification (Rev 3.4, `PS-MPU-6000A-00`)** and **Register Map (Rev 4.2, `RM-MPU-6000A-00`)**.

All internal equations use consistent **SI units**:
- Specific force (accelerometer): $\text{m/s}^2$, with standard gravity $g_0 = 9.80665\,\text{m/s}^2$.
- Angular velocity (gyroscope): $\text{rad/s}$.
- Time and sampling interval: $\text{s}$; sampling frequency $f_s = 1 / \Delta t$ in $\text{Hz}$.

---

## 1. Unified 3-Axis Sensor Measurement Equation

Let $\mathbf{u}_k \in \mathbb{R}^3$ denote the ideal physical input vector in the sensor body frame at discrete sample time $t_k = k \Delta t$:
- For the **accelerometer**, $\mathbf{u}_k = \mathbf{f}_k = [f_{x,k}, f_{y,k}, f_{z,k}]^T$ is the ideal specific force ($\text{m/s}^2$). For a stationary sensor aligned with $z$-up, $\mathbf{f}_k = [0, 0, +g_0]^T$.
- For the **gyroscope**, $\mathbf{u}_k = \boldsymbol{\omega}_k = [\omega_{x,k}, \omega_{y,k}, \omega_{z,k}]^T$ is the ideal angular velocity ($\text{rad/s}$).

Following Farrell et al. (2022), the pre-ADC (analog) corrupted measurement vector $\tilde{\mathbf{u}}_k^{\mathrm{analog}} \in \mathbb{R}^3$ is:

$$\tilde{\mathbf{u}}_k^{\mathrm{analog}} = \left(\mathbf{I}_3 + \mathbf{S} + \mathbf{M}\right)\mathbf{u}_k + \mathbf{b}_0 + \mathbf{z}_k \tag{1}$$

where:
- $\mathbf{I}_3 \in \mathbb{R}^{3\times 3}$ is the $3\times 3$ identity matrix.
- $\mathbf{S} = \operatorname{diag}(s_x, s_y, s_z) \in \mathbb{R}^{3\times 3}$ is the dimensionless diagonal **scale-factor error matrix** (fractional deviation from nominal sensitivity; e.g., $s_i = 0.01$ represents $+1\%$ scale-factor error).
- $\mathbf{M} \in \mathbb{R}^{3\times 3}$ is the dimensionless **cross-axis sensitivity / non-orthogonality misalignment matrix** with zero diagonal entries ($M_{ii} = 0$):
  $$\mathbf{M} = \begin{bmatrix} 0 & m_{xy} & m_{xz} \\ m_{yx} & 0 & m_{yz} \\ m_{zx} & m_{zy} & 0 \end{bmatrix} \tag{2}$$
- $\mathbf{b}_0 = [b_{0,x}, b_{0,y}, b_{0,z}]^T \in \mathbb{R}^3$ is the constant **turn-on / static bias** vector ($\text{m/s}^2$ for accelerometer; $\text{rad/s}$ for gyroscope).
- $\mathbf{z}_k = [z_{x,k}, z_{y,k}, z_{z,k}]^T \in \mathbb{R}^3$ is the **cumulative stochastic error** vector at step $k$.

**Python mapping:**
- Implemented independently in `Accelerometer.step()` (`src/mpu6050_sim/accelerometer.py`) and `Gyroscope.step()` (`src/mpu6050_sim/gyroscope.py`).
- Effective deterministic gain matrix $\mathbf{G} = \mathbf{I}_3 + \mathbf{S} + \mathbf{M}$ is precomputed at initialization.

---

## 2. Continuous-Time Stochastic Error Model

Following Farrell et al. (2022), the stochastic error on each axis $i \in \{x, y, z\}$ is modeled as the sum of three mutually independent stochastic processes:

$$z_i(t) = x_{B,i}(t) + x_{K,i}(t) + v_{N,i}(t) \tag{3}$$

### 2.1 Broadband White Measurement Noise ($v_{N,i}$ — ARW / VRW)
Continuous-time zero-mean white Gaussian noise representing thermal and electronic sensor noise, characterized by its **Angle Random Walk (ARW)** coefficient $N_{g,i}$ (gyroscope, $(\text{rad/s})\cdot\sqrt{\text{s}} = \text{rad}/\sqrt{\text{s}}$) or **Velocity Random Walk (VRW)** coefficient $N_{a,i}$ (accelerometer, $(\text{m/s}^2)\cdot\sqrt{\text{s}} = (\text{m/s})/\sqrt{\text{s}}$):

$$\mathbb{E}\!\left[v_{N,i}(t)\,v_{N,i}(t+\tau)\right] = S_{N,i}\,\delta(\tau), \qquad S_{N,i} = N_i^2 \tag{4}$$

**Relationship to InvenSense single-sided Noise Spectral Density ($\mathrm{NSD}_{1\mathrm{s}}$):**
Under IEEE Std 952-1997 and El-Sheimy et al. (2008), the two-sided power spectral density $S_{N,i} = N_i^2$ satisfies $\sigma_A^2(\tau) = 4\int_0^\infty S_{N,i} \frac{\sin^4(\pi f\tau)}{(\pi f\tau)^2}\,df = \frac{N_i^2}{\tau}$. Because a single-sided spectral density satisfies $S_{1\mathrm{s}} = \mathrm{NSD}_{1\mathrm{s}}^2 = 2 S_{N,i}$, the equivalent ARW/VRW parameter is:

$$N_i = \frac{\mathrm{NSD}_{1\mathrm{s},i}}{\sqrt{2}} \tag{5}$$

Sanity check with the MPU-6050 gyroscope datasheet (`PS-MPU-6000A-00` Rev 3.4):
- Datasheet single-sided rate noise spectral density: $\mathrm{NSD}_{1\mathrm{s},g} = 0.005\,(^\circ/\text{s})/\sqrt{\text{Hz}} = 8.726646 \times 10^{-5}\,(\text{rad/s})/\sqrt{\text{Hz}}$.
- Corresponding two-sided ARW: $N_g = \mathrm{NSD}_{1\mathrm{s},g} / \sqrt{2} = 6.170671 \times 10^{-5}\,(\text{rad/s})\cdot\sqrt{\text{s}}$.
- At $f_s = 200\,\text{Hz}$ (Nyquist bandwidth $B = f_s / 2 = 100\,\text{Hz}$, matching MPU-6050 `DLPF_CFG = 2`):
  $$\sigma_{d,g} = N_g \sqrt{f_s} = \frac{\mathrm{NSD}_{1\mathrm{s},g}}{\sqrt{2}} \sqrt{200} = 10\,\mathrm{NSD}_{1\mathrm{s},g} = 0.05\,^\circ/\text{s-rms}$$
  which matches the datasheet Total RMS Noise specification ($0.05\,^\circ/\text{s-rms}$ at $100\,\text{Hz}$ bandwidth) identically.

### 2.2 Time-Correlated Bias Instability ($x_{B,i}$ — First-Order Gauss–Markov Process)
As detailed in Farrell et al. (2022), finite-dimensional state-space modeling approximates bounded in-run bias instability using a continuous-time **first-order Gauss–Markov (GM) process** with correlation time $T_{B,i} > 0$ ($\text{s}$) and white driving noise $w_{B,i}(t)$ of spectral density $S_{B,i}$:

$$\dot{x}_{B,i}(t) = -\frac{1}{T_{B,i}}\, x_{B,i}(t) + w_{B,i}(t), \qquad \mathbb{E}\!\left[w_{B,i}(t)\,w_{B,i}(t+\tau)\right] = S_{B,i}\,\delta(\tau) \tag{6}$$

The steady-state variance of $x_{B,i}(t)$ is:

$$\sigma_{\mathrm{gm},i}^2 = \frac{S_{B,i}\,T_{B,i}}{2} \tag{7}$$

**Mapping from Allan Variance Bias Instability coefficient $B_i$:**
In Farrell et al. (2022), the exact Allan variance of a first-order Gauss–Markov process is:

$$\sigma_{A,\mathrm{GM}}^2(\tau) = S_{B,i} T_{B,i} \left(\frac{1}{\lambda} - \frac{1}{2\lambda^2}\left[3 - 4e^{-\lambda} + e^{-2\lambda}\right]\right), \qquad \lambda = \frac{\tau}{T_{B,i}} \tag{8}$$

This curve attains its peak at $\lambda_{\max} \approx 1.89$ ($\tau_{\max} \approx 1.89\,T_{B,i}$), where:

$$\sigma_{A,\mathrm{GM}}(1.89\,T_{B,i}) \approx 0.43654\,\sqrt{S_{B,i} T_{B,i}} \tag{9}$$

Matching this peak to the standard IEEE Std 952 / El-Sheimy (2008) flicker-floor Allan deviation $\sigma_{A,\mathrm{BI}} = \sqrt{\frac{2\ln 2}{\pi}}\,B_i \approx 0.66428\,B_i$ yields the driving PSD $S_{B,i}$ directly from the user-configurable $(B_i, T_{B,i})$:

$$S_{B,i} = \frac{1}{T_{B,i}} \left(\frac{c_{\mathrm{BI}}\,B_i}{c_{\mathrm{GM}}}\right)^2, \qquad c_{\mathrm{BI}} = \sqrt{\frac{2\ln 2}{\pi}} \approx 0.664282, \qquad c_{\mathrm{GM}} \approx 0.436541 \tag{10}$$

Equivalently, $\sigma_{\mathrm{gm},i} = \sqrt{S_{B,i} T_{B,i} / 2} \approx 1.07601\,B_i$.

### 2.3 Bias Drift / Rate or Acceleration Random Walk ($x_{K,i}$)
Long-term unbounded bias drift is modeled as a continuous-time **random walk** (integrated white noise) with coefficient $K_i$ ($\text{rad/s}^{3/2}$ for gyroscope RRW; $\text{m/s}^{5/2}$ for accelerometer ARW-drift):

$$\dot{x}_{K,i}(t) = w_{K,i}(t), \qquad \mathbb{E}\!\left[w_{K,i}(t)\,w_{K,i}(t+\tau)\right] = K_i^2\,\delta(\tau) \tag{11}$$

---

## 3. Exact Continuous-to-Discrete State-Space Discretization

In Farrell et al. (2022), a continuous-time linear stochastic system $\dot{\mathbf{x}}(t) = \mathbf{F}\mathbf{x}(t) + \mathbf{G}\mathbf{w}(t)$ sampled at interval $\Delta t = 1/f_s$ has the exact discrete-time representation:

$$\mathbf{x}[k+1] = \mathbf{A}_d\,\mathbf{x}[k] + \mathbf{w}_d[k], \qquad \mathbf{A}_d = \exp(\mathbf{F}\Delta t) \tag{12}$$

$$\mathbf{Q}_d = \mathbb{E}\!\left[\mathbf{w}_d[k]\mathbf{w}_d[k]^T\right] = \int_0^{\Delta t} \exp(\mathbf{F}\tau)\,\mathbf{G}\mathbf{Q}_c\mathbf{G}^T\,\exp(\mathbf{F}^T\tau)\,d\tau \tag{13}$$

Because the per-axis stochastic states $\mathbf{x}_i[k] = [x_{B,i}[k],\; x_{K,i}[k]]^T$ are decoupled ($\mathbf{F}_i = \operatorname{diag}(-1/T_{B,i},\, 0)$, $\mathbf{G}_i = \mathbf{I}_2$, $\mathbf{Q}_{c,i} = \operatorname{diag}(S_{B,i},\, K_i^2)$), Equations (12)–(13) admit exact closed-form discrete updates:

1. **Gauss–Markov Bias Instability State:**
   $$x_{B,i}[k+1] = \Phi_{B,i}\, x_{B,i}[k] + w_{B,d,i}[k] \tag{14}$$
   $$\Phi_{B,i} = \exp\!\left(-\frac{\Delta t}{T_{B,i}}\right), \qquad q_{B,d,i} = \frac{S_{B,i}\,T_{B,i}}{2}\left(1 - \exp\!\left(-\frac{2\Delta t}{T_{B,i}}\right)\right) \tag{15}$$
   where $w_{B,d,i}[k] \sim \mathcal{N}(0, q_{B,d,i})$. At $k = 0$, $x_{B,i}[0]$ is initialized either from its stationary distribution $\mathcal{N}(0, \sigma_{\mathrm{gm},i}^2)$ (default, ensuring wide-sense stationarity from $t=0$) or at $0$ (configurable).

2. **Random Walk Drift State:**
   $$x_{K,i}[k+1] = x_{K,i}[k] + w_{K,d,i}[k], \qquad q_{K,d,i} = K_i^2\,\Delta t \tag{16}$$
   where $w_{K,d,i}[k] \sim \mathcal{N}(0, q_{K,d,i})$ and $x_{K,i}[0] = 0$.

3. **Discrete White Measurement Noise:**
   $$v_{N,i}[k] \sim \mathcal{N}\!\left(0, \sigma_{N,d,i}^2\right), \qquad \sigma_{N,d,i}^2 = \frac{N_i^2}{\Delta t} = N_i^2\,f_s \tag{17}$$

4. **Total Discrete Stochastic Output at Sample $k$:**
   $$z_i[k] = x_{B,i}[k] + x_{K,i}[k] + v_{N,i}[k] \tag{18}$$

**Python mapping:**
- Implemented in `ThreeAxisStochasticError` inside `src/mpu6050_sim/noise.py`.
- Both scalar `step()` and vectorized `generate(n_samples)` use the exact recurrence equations (14)–(18) with a dedicated NumPy `np.random.Generator` seeded for deterministic reproducibility.

---

## 4. MPU-6050 Hardware Output Stage (Saturation and 16-Bit ADC Quantization)

From the InvenSense `PS-MPU-6000A-00` (Rev 3.4) and `RM-MPU-6000A-00` (Rev 4.2) specifications, the continuous-valued physical measurement $\tilde{\mathbf{u}}_k^{\mathrm{analog}}$ passes through full-scale range limiting and a 16-bit signed two's-complement ADC:

### 4.1 Full-Scale Range Saturation
When saturation is enabled (`enable_saturation: true`):

$$\tilde{u}_{i,k}^{\mathrm{sat}} = \operatorname{clip}\!\left(\tilde{u}_{i,k}^{\mathrm{analog}},\; -u_{\mathrm{FSR}},\; +u_{\mathrm{FSR}}\right) \tag{19}$$

where:
- Accelerometer $u_{\mathrm{FSR}} \in \{2, 4, 8, 16\} \times g_0$ ($\text{m/s}^2$).
- Gyroscope $u_{\mathrm{FSR}} \in \{250, 500, 1000, 2000\} \times \frac{\pi}{180}$ ($\text{rad/s}$).

### 4.2 16-Bit Signed ADC Quantization
When quantization is enabled (`enable_quantization: true`), the sensor converts the saturated physical value into 16-bit signed integer counts $c_{i,k} \in \{-32768, \dots, 32767\}$ using the nominal datasheet sensitivity $S_{\mathrm{nom}}$ (in $\text{LSB}/(\text{m/s}^2)$ or $\text{LSB}/(\text{rad/s})$), and scales back to SI physical units:

$$c_{i,k} = \operatorname{clip}\!\left(\operatorname{round}\!\left(\tilde{u}_{i,k}^{\mathrm{sat}}\,S_{\mathrm{nom}}\right),\; -32768,\; 32767\right) \tag{20}$$

$$\tilde{u}_{i,k} = \frac{c_{i,k}}{S_{\mathrm{nom}}} \tag{21}$$

The quantization step size is $\Delta_{\mathrm{LSB}} = 1 / S_{\mathrm{nom}}$. Because the MPU-6050's 16-bit ADC quantizes instantaneous **specific-force / angular-rate samples** directly (rather than integrated velocity or angle increments as in ring-laser/fiber-optic gyros), under the standard Bennett white quantization model (valid whenever broadband noise $\sigma_{N,d} \gtrsim \Delta_{\mathrm{LSB}}/2$), per-sample ADC quantization adds uncorrelated error of variance $\Delta_{\mathrm{LSB}}^2 / 12$, contributing $\frac{\Delta_{\mathrm{LSB}}^2\,\Delta t}{12\,\tau}$ to the rate Allan variance.

---

## 5. Analytical Allan Variance Reference Equation (Validation)

By independence of the error processes (El-Sheimy et al., 2008; Farrell et al., 2022; IEEE Std 952-1997), the total theoretical Allan variance of a stationary simulated MPU-6050 axis output (constant input $\mathbf{u}_k = \mathbf{u}_0$) as a function of cluster averaging time $\tau = m\Delta t$ is:

$$\sigma_{A,\mathrm{total}}^2(\tau) = \frac{\Delta_{\mathrm{LSB}}^2\,\Delta t}{12\,\tau} + \frac{N^2}{\tau} + S_B T_B \left(\frac{1}{\lambda} - \frac{3 - 4e^{-\lambda} + e^{-2\lambda}}{2\lambda^2}\right) + \frac{K^2 \tau}{3}, \qquad \lambda = \frac{\tau}{T_B} \tag{22}$$

*(Note: When ADC quantization is disabled, $\Delta_{\mathrm{LSB}} = 0$. The classic IEEE Std 952 $3Q^2/\tau^2$ term with slope $-1$ applies only to sensors outputting quantized integrated angle/velocity increments, whereas direct rate/acceleration ADC quantization contributes a $-1/2$ slope term $\frac{\Delta_{\mathrm{LSB}}^2 \Delta t}{12\tau}$.)*

The empirical overlapping Allan variance computed from a simulated time series $\{y_k\}_{k=0}^{M-1}$ via integrated phase $\theta_k = \Delta t \sum_{j=0}^{k-1} y_j$ is (El-Sheimy et al., 2008, Eq. 5):

$$\hat{\sigma}_A^2(m\Delta t) = \frac{1}{2(m\Delta t)^2 (M - 2m)} \sum_{k=0}^{M - 2m - 1} \left(\theta_{k+2m} - 2\theta_{k+m} + \theta_k\right)^2 \tag{23}$$

Equation (22) vs. Equation (23) forms the core quantitative validation check of our Python implementation against Farrell et al. (2022) and El-Sheimy et al. (2008).

---

## 6. Summary Mapping: Equations to Python Modules

| Model Component | Equations | Python Module & Class/Function |
|---|---|---|
| Configuration parsing, unit tables, FSR/sensitivity lookup | §4 | `src/mpu6050_sim/configuration.py` (`MPU6050Config`, `AccelerometerConfig`, `GyroscopeConfig`, `SimulationConfig`) |
| Exact discrete-time stochastic processes ($v_N, x_B, x_K$) | (3)–(18) | `src/mpu6050_sim/noise.py` (`ThreeAxisStochasticError`) |
| 3-axis accelerometer deterministic + stochastic + ADC | (1)–(2), (19)–(21) | `src/mpu6050_sim/accelerometer.py` (`Accelerometer`) |
| 3-axis gyroscope deterministic + stochastic + ADC | (1)–(2), (19)–(21) | `src/mpu6050_sim/gyroscope.py` (`Gyroscope`) |
| Integrated 6-axis MPU-6050 simulator, sequence runner, CSV export | (1)–(21) | `src/mpu6050_sim/sensor.py` (`MPU6050`, `IMUMeasurement`, `SimulationResult`) |
| Plotting utilities (time-domain, parameter sweeps, Allan deviation) | (22)–(23) | `src/mpu6050_sim/visualization.py` |
| Analytical vs. empirical Allan variance validation | (22)–(23) | `examples/validate_against_paper.py`, `tests/test_validation.py` |
