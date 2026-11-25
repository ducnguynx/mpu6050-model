# Mô phỏng Cảm biến MPU-6050 bằng Python

## 1. Mô hình được xây dựng như thế nào?

### 1.1. Cơ sở lý thuyết và nguồn tham số
Mô hình kết hợp 2 nguồn chuẩn:
- **Mô hình toán học:** Theo bài báo IEEE của **Farrell, Silva, Rahman & Wendel (2022)**, *"Inertial Measurement Unit Error Modeling Tutorial"*, *IEEE Control Systems Magazine*, 42(6):40–66 (`DOI: 10.1109/MCS.2022.3209059`, mã nguồn MATLAB tham chiếu `jaffarrell/AV-Matlab-SW`).
- **Thông số phần cứng MPU-6050:** Lấy từ tài liệu chính thức của hãng **InvenSense `PS-MPU-6000A-00` (Rev 3.4)** và **Register Map `RM-MPU-6000A-00` (Rev 4.2)**.

### 1.2. Ý tưởng 
Đưa vào **tín hiệu chuyển động lý tưởng** $\mathbf{u}_k$ (gia tốc riêng tính bằng $\text{m/s}^2$, hoặc vận tốc góc tính bằng $\text{rad/s}$). Bộ mô phỏng sẽ cộng thêm đúng các sai số thực tế của MPU-6050 theo phương trình:

$$\tilde{\mathbf{u}}_k^{\text{analog}} = \underbrace{(\mathbf{I}_3 + \mathbf{S} + \mathbf{M})\,\mathbf{u}_k + \mathbf{b}_0}_{\text{Sai số tất định (cố định)}} + \underbrace{\mathbf{v}_{N,k} + \mathbf{x}_{B,k} + \mathbf{x}_{K,k}}_{\text{Nhiễu ngẫu nhiên \& trôi theo thời gian}}$$

Sau đó tín hiệu đi qua **tầng phần cứng của MPU-6050**:
1. **Bão hòa (Saturation):** Cắt ngọn tín hiệu nếu vượt quá dải đo (Full-Scale Range).
2. **Lượng tử hóa ADC 16-bit (Quantization):** Làm tròn sang số nguyên 16-bit có dấu (`-32768` đến `32767`) theo độ nhạy LSB của datasheet.

### 1.3. Ý nghĩa từng tham số và nguồn gốc (xem file `config/default.yaml`)

Mọi tính toán bên trong đều dùng **hệ đơn vị SI** ($\text{m/s}^2$, $\text{rad/s}$, $\text{s}$, $\text{Hz}$).

| Thành phần / Tham số | Ký hiệu | Ý nghĩa dễ hiểu | Nguồn gốc |
|---|---|---|---|
| **Dải đo & Độ nhạy ADC** (`full_scale_range_*`) | $\text{FSR}, S_{\text{nom}}$ | Giới hạn đo ($\pm 2/4/8/16\,g$; $\pm 250/500/1000/2000\,s$) và bước lượng tử 16-bit (`16384 LSB/g`, `131 LSB/(°/s)`...). | **Datasheet MPU-6050** |
| **Nhiễu trắng băng rộng** (`noise_density_*`) | $N$ ($\mathbf{v}_N$) | Độ rung ngẫu nhiên nhanh ở từng mẫu đo (gây ARW / VRW). Sinh từ phân phối chuẩn $\mathcal{N}(0, N^2 f_s)$, độc lập giữa các mẫu. | **Suy ra trực tiếp từ Datasheet** ($400\,\mu g/\sqrt{\text{Hz}}$ và $0.005\,(^\circ/\text{s})/\sqrt{\text{Hz}}$ chia $\sqrt{2}$) |
| **Sai số hệ số tỉ lệ** (`scale_factor_error`) | $\mathbf{S}$ | "Thước đo" của từng trục bị co/giãn nhẹ (vd $+0.5\%$). | **Giả định cấu hình được**, nằm trong dung sai Datasheet ($\pm 3\%$) |
| **Lệch trục / Xuyên trục** (`misalignment`) | $\mathbf{M}$ | 3 trục X, Y, Z không vuông góc tuyệt đối $90^\circ$ nên chuyển động trục này rò nhẹ sang trục khác. | **Giả định cấu hình được**, nằm trong dung sai Datasheet ($\pm 2\%$) |
| **Độ lệch tĩnh (Bias)** (`static_bias_*`) | $\mathbf{b}_0$ | Sai số lệch điểm 0 cố định ngay cả khi cảm biến đứng yên. | **Giả định cấu hình được**, nằm trong dung sai Datasheet ($\pm 50/80\,\text{mg}$; $\pm 20\,s$) |
| **Bất ổn định bias** (`bias_instability_*`, `bias_correlation_time_s`) | $B, T_B$ ($\mathbf{x}_B$) | Điểm 0 dao động chậm theo thời gian nhưng có giới hạn. Mô hình hóa bằng quá trình **Gauss–Markov bậc 1**: $x_B[k+1] = e^{-\Delta t/T_B} x_B[k] + w_B[k]$. | **Giả định cấu hình được** (Datasheet không công bố; tham khảo nghiên cứu Allan Variance trên MPU-6050) |
| **Trôi ngẫu nhiên dài hạn** (`random_walk_*`) | $K$ ($\mathbf{x}_K$) | Độ trôi cộng dồn chậm theo thời gian (**Random Walk**): $x_K[k+1] = x_K[k] + w_K[k]$. Có thể đặt bằng `0` nếu không muốn trôi vô hạn. | **Giả định cấu hình được** (Datasheet không công bố) |
| **Hạt giống ngẫu nhiên** (`random_seed`) | — | Số nguyên (vd `42`) giúp tái lập **chính xác cùng một chuỗi nhiễu** mỗi lần chạy để so sánh công bằng các bộ lọc. | **Giả định cấu hình được** |

### 1.4. Kiểm chứng mô hình
Mô hình đã được kiểm chứng định lượng (chi tiết xem [docs/validation_report.md](docs/validation_report.md)):
- **Nhiễu RMS băng rộng tại $f_s = 200\,\text{Hz}$ (băng thông Nyquist $100\,\text{Hz}$):** Con quay đạt $0.05006\,s-rms$ so với $0.05000\,s-rms$ trong datasheet (sai lệch **0.12%**); gia tốc kế đạt $4.005\,\text{mg-rms}$ so với $4.000\,\text{mg-rms}$ (sai lệch **0.12%**).
- **Đường cong Allan Deviation (chạy tĩnh 1 giờ):** Khớp công thức lý thuyết của Farrell (2022) & El-Sheimy (2008) với sai số trung bình **1.25%** (gyro) và **2.05%** (accel).

---

## 2. Cài đặt, Chạy thử và Kết nối với Script Xử lý Tín hiệu

### 2.1. Cài đặt
Yêu cầu **Python $\ge 3.10$** cùng 3 thư viện lõi: `numpy`, `matplotlib`, `PyYAML`.

```bash
# (Tùy chọn) Cài package ở chế độ editable để import mpu6050_sim từ bất kỳ đâu
pip install -e . --no-build-isolation
```

### 2.2. Chạy kiểm thử và các kịch bản mẫu
Tất cả thông số cảm biến đều chỉnh được trong [config/default.yaml](config/default.yaml) mà **không cần sửa mã nguồn**.

```bash
# 1. Chạy toàn bộ 20 unit test & validation test
python3 -m unittest discover -s tests -v

# 2. Chạy Kịch bản 1-3 (tĩnh, sin, đa tần số) -> xuất hình PNG & file CSV vào outputs/
python3 examples/basic_simulation.py

# 3. Chạy Kịch bản 4 (thay đổi mức nhiễu và bias trên gia tốc kế & con quay)
python3 examples/accelerometer_simulation.py
python3 examples/gyroscope_simulation.py

# 4. Chạy kiểm chứng RMS datasheet & Allan Deviation theo bài báo IEEE
python3 examples/validate_against_paper.py
```

### 2.3. Cách kết nối vào Pipeline Xử lý Tín hiệu

Có **3 cách** lấy dữ liệu từ bộ mô phỏng:

#### Cách 1: Lấy từng mẫu theo thời gian thực
Phù hợp cho **bộ lọc đệ quy / online** (Kalman Filter, Complementary Filter, bộ lọc thích nghi...).

```python
import sys
sys.path.insert(0, "src")
from mpu6050_sim import MPU6050

sensor = MPU6050.from_config("config/default.yaml")
dt = sensor.dt  # 0.005 s (tại fs = 200 Hz)

for k in range(2000):
    true_acc = [0.0, 0.0, 9.80665]  # Tín hiệu lý tưởng [m/s^2]
    true_gyr = [0.0, 0.0, 0.0]      # Tín hiệu lý tưởng [rad/s]

    # Lấy 1 mẫu đo nhiễu giống như đọc từ chip MPU-6050 thật
    meas = sensor.step(acceleration=true_acc, angular_velocity=true_gyr)

    # Đưa thẳng vào bộ lọc của bạn:
    # meas.accelerometer : mảng (3,) [m/s^2]
    # meas.gyroscope     : mảng (3,) [rad/s]
    # meas.raw_accelerometer / meas.raw_gyroscope : số nguyên 16-bit [LSB]
    est = my_filter.update(meas.accelerometer, meas.gyroscope, dt)
```

#### Cách 2: Lấy nguyên chuỗi mảng NumPy 
Phù hợp cho **xử lý theo khối / offline** (Wiener Filter, Moving Average, FFT/PSD, FIR/IIR...) và tiện tính RMSE/SNR vì có sẵn tín hiệu gốc `ideal_*`.

```python
import sys
sys.path.insert(0, "src")
import numpy as np
from mpu6050_sim import MPU6050

sensor = MPU6050.from_config("config/default.yaml")
t = sensor.time_vector(duration_s=10.0)

# Tạo tín hiệu lý tưởng đầu vào, kích thước (N, 3)
ideal_acc = np.column_stack([
    2.0 * np.sin(2 * np.pi * 1.0 * t),
    np.zeros_like(t),
    np.full_like(t, 9.80665),
])
ideal_gyr = np.zeros((t.size, 3))

# Mô phỏng toàn bộ chuỗi
result = sensor.simulate(ideal_acc, ideal_gyr)

noisy_ax = result.accelerometer[:, 0]       # Tín hiệu đo có nhiễu
clean_ax = result.ideal_acceleration[:, 0]  # Ground truth để đánh giá bộ lọc

filtered_ax = my_filter(noisy_ax, fs=result.sample_rate_hz)
rmse_before = np.sqrt(np.mean((noisy_ax - clean_ax) ** 2))
rmse_after  = np.sqrt(np.mean((filtered_ax - clean_ax) ** 2))
```

*(Mẹo thử nghiệm: dùng `sensor.update_parameters(gyroscope={"static_bias_radps": [0.05, 0.0, 0.0]})` và `sensor.reset()` để thay đổi mức nhiễu/bias hoặc lặp lại cùng chuỗi nhiễu khi so sánh nhiều bộ lọc.)*

#### Cách 3: Xuất ra file CSV rồi đọc ở script khác
Phù hợp khi muốn xuất ra file CSV rồi xử lý sau

```python
# Ở script sinh dữ liệu:
result.to_csv("outputs/dataset.csv")

# Ở script xử lý tín hiệu riêng biệt:
import numpy as np
data = np.genfromtxt("outputs/dataset.csv", delimiter=",", names=True)
t        = data["time_s"]
noisy_ax = data["meas_ax_mps2"]
clean_ax = data["ideal_ax_mps2"]
noisy_gx = data["meas_gx_radps"]
clean_gx = data["ideal_gx_radps"]
```

---

## 3. Cấu trúc thư mục & Tài liệu chi tiết

```text
.
├── config/default.yaml          # File cấu hình YAML toàn bộ tham số MPU-6050
├── src/mpu6050_sim/             # Mã nguồn mô phỏng (accelerometer, gyroscope, noise, sensor...)
├── examples/                    # Các kịch bản mẫu (static, sin, đa tần, sweep) & kiểm chứng
├── tests/                       # 20 bài kiểm thử tự động (unittest / pytest)
├── outputs/                     # Biểu đồ PNG và file CSV sinh ra từ examples/
└── docs/
    ├── references.md            # Tổng quan bài báo IEEE và lý do chọn mô hình
    ├── mathematical_model.md    # Toàn bộ công thức toán học & ánh xạ sang mã Python
    ├── parameters.md            # Bảng tra cứu nguồn gốc từng tham số (datasheet vs giả định)
    ├── model_assumptions.md     # Các giả định và giới hạn của mô hình
    └── validation_report.md     # Báo cáo kiểm chứng định lượng với bài báo & datasheet
```
