Dựa trên mô hình cảm biến MPU-6050 trong repo của bạn ([mathematical_model.md](file:///home/duc/Workspace/master/random-signal-processing/docs/mathematical_model.md) và [default.yaml](file:///home/duc/Workspace/master/random-signal-processing/config/default.yaml)), đặc trưng cốt lõi của tín hiệu đo là:
$$\tilde{\mathbf{u}}_k = (\mathbf{I}_3 + \mathbf{S} + \mathbf{M})\mathbf{u}_k + \mathbf{b}_0 + \underbrace{\mathbf{v}_{N,k}}_{\text{nhiễu trắng}} + \underbrace{\mathbf{x}_{B,k}}_{\text{Gauss-Markov (nhiễu màu)}} + \underbrace{\mathbf{x}_{K,k}}_{\text{Random Walk (không dừng)}}$$

Điểm quan trọng nhất của mô hình này là **nhiễu không chỉ là nhiễu trắng dừng**, mà có cả **nhiễu màu tương quan thời gian** ($T_B \approx 100\text{ s}$) và **trôi không dừng** ($x_K$), đồng thời cảm biến có **6 trục** (gia tốc kế + con quay). Điều này quyết định trực tiếp phương án lọc nào hợp lý.

---

## 1. Giải thích nhanh 4 gợi ý của thầy bạn

Vì bạn đang chưa rõ ý nghĩa của 4 cặp thầy đưa ra, hiểu nôm na mục tiêu so sánh của từng cặp như sau:

1. **Wiener và Kalman**: So sánh 2 trường phái lọc tối ưu MMSE kinh điển: *Wiener* (giả định tín hiệu/nhiễu dừng, dùng tương quan/phổ tần số, hệ số lọc cố định) với *Kalman* (không gian trạng thái, xử lý đệ quy từng mẫu, làm việc được với quá trình không dừng/trôi).
2. **Kalman với hai cấu trúc mô hình khác nhau**: Cùng là Kalman, nhưng so sánh *mô hình trạng thái đơn giản* (ví dụ chỉ coi nhiễu là nhiễu trắng) với *mô hình trạng thái mở rộng* (bổ sung trạng thái ẩn biểu diễn nhiễu màu/trôi bias).
3. **Kalman cố định và Kalman có tham số ước lượng lại bằng RLS**: So sánh Kalman dùng bộ tham số nhận dạng 1 lần rồi giữ cố định, với Kalman kết hợp RLS chạy online để liên tục cập nhật lại tham số mô hình khi đặc tính tín hiệu/cảm biến thay đổi theo thời gian.
4. **LMS và RLS**: So sánh 2 thuật toán lọc thích nghi dạng FIR (khác nhau chủ yếu ở tốc độ hội tụ và độ phức tạp tính toán), đòi hỏi tín hiệu tham chiếu $d[k]$ hoặc kênh nhiễu tham chiếu.

---

## 2. Đề xuất 2 phương án phù hợp nhất & Lý do chọn

### Phương án 1 (Khuyên chọn số 1): **Wiener và Kalman**
- **Lý do phù hợp:**
  - **Đúng trọng tâm môn Xử lý tín hiệu ngẫu nhiên:** Thể hiện rõ bước chuyển lý thuyết từ bộ lọc dừng theo miền phổ/tương quan (**Wiener**) sang bộ lọc không gian trạng thái đệ quy xử lý được tính không dừng (**Kalman**).
  - **Phơi bày rõ bản chất MPU-6050:** Trong mô hình của bạn, nhiễu gồm cả phần *dừng* ($v_N$, $x_B$ xác lập) lẫn phần *không dừng* ($x_K$ trôi theo thời gian) và tín hiệu chuyển động thay đổi chế độ ([basic_simulation.py](file:///home/duc/Workspace/master/random-signal-processing/examples/basic_simulation.py)). Khi chạy thử:
    - Ở đoạn ngắn, tín hiệu gần dừng: Wiener và Kalman cho kết quả xấp xỉ nhau (vì Kalman ở chế độ xác lập tương đương bộ lọc Wiener nhân quả).
    - Ở đoạn dài (trôi bias xuất hiện), hoặc tín hiệu biến thiên đột ngột / cần lọc thời gian thực ngay từ $t=0$: Kalman thể hiện ưu thế rõ rệt.
  - **Phù hợp với điều kiện "không biết mô hình cảm biến":** Wiener là cách tiếp cận *phi tham số* (ước lượng trực tiếp phổ PSD / tự tương quan từ dữ liệu), còn Kalman là cách tiếp cận *có tham số* (nhận dạng mô hình không gian trạng thái từ dữ liệu). Hai hướng tiếp cận bổ sung cho nhau rất chặt chẽ.

### Phương án 2 (Khuyên chọn số 2): **Kalman với hai cấu trúc mô hình khác nhau**
- **Lý do phù hợp:**
  - **Đánh thẳng vào giá trị học thuật của mô hình bạn đã xây dựng:** Repo của bạn xây theo bài báo IEEE Farrell et al. (2022) với cấu trúc nhiễu 3 tầng ($v_N + x_B + x_K$). Phương án này trả lời trực tiếp câu hỏi: *"Khi không biết cấu trúc thật của cảm biến, nếu giả sử đơn giản chỉ có nhiễu trắng (Cấu trúc 1) so với mô hình hóa thêm trạng thái bias/nhiễu màu (Cấu trúc 2), sai số và tính nhất quán bộ lọc khác nhau thế nào?"*
  - **Có thể triển khai ở 2 mức độ đều rất đẹp:**
    - *Mức 1 (lọc 1 trục gia tốc/gyro):* Cấu trúc 1 chỉ có trạng thái tín hiệu $[u_k, \dot{u}_k]^T$ + nhiễu đo trắng; Cấu trúc 2 mở rộng thêm trạng thái bias màu $[u_k, \dot{u}_k, b_k]^T$.
    - *Mức 2 (khuyên dùng nếu muốn điểm tối đa — hợp nhất 6 trục Accel + Gyro ước lượng góc nghiêng Roll/Pitch):* Cấu trúc 1 chỉ ước lượng góc $\theta_k$ (bỏ qua bias gyro $\Rightarrow$ góc bị lệch tĩnh mãi mãi); Cấu trúc 2 ước lượng đồng thời $[\theta_k, b_{g,k}]^T$ (hệ hoàn toàn quan sát được $\Rightarrow$ tách và khử sạch cả bias tĩnh $b_0$, Gauss-Markov $x_B$ lẫn trôi $x_K$).
  - **An toàn về mặt toán học:** Dễ kiểm chứng chặt chẽ bằng kiểm định tính trắng của phần dư (Innovation) và tính nhất quán (NIS), không bị các bẫy mất ổn định số.

---

## 3. Tại sao KHÔNG nên dùng 2 phương án còn lại?

### Tại sao không chọn **LMS và RLS**?
1. **Bế tắc về tín hiệu tham chiếu (Reference Signal Problem):** LMS/RLS là bộ lọc thích nghi cập nhật trọng số theo sai số $e[k] = d[k] - \mathbf{w}^T[k]\mathbf{x}[k]$.
   - Trong thực tế đo MPU-6050, bạn **không có** tín hiệu sạch $d[k]$ (nếu lấy `ideal_acc`/`ideal_gyr` từ bộ mô phỏng làm $d[k]$ thì là "gian lận", vì ngoài đời cảm biến không có ground truth).
   - Bạn cũng **không có** kênh cảm biến thứ hai chỉ đo riêng nhiễu nội bộ $r[k]$ để làm khử nhiễu thích nghi (ANC).
2. **Dùng biến thể tự dự đoán trễ (ALE) sẽ thất bại với bias/drift:** Nếu lấy chính $y[k-\Delta]$ làm đầu vào để tránh cần $d[k]$, bộ lọc chỉ tách được sóng tuần hoàn khỏi nhiễu trắng $v_N$, nhưng **nhiễu Gauss–Markov ($T_B = 100\text{ s}$) và Random Walk ($x_K$) có độ tự tương quan cực cao** $\Rightarrow$ LMS/RLS sẽ nhầm độ trôi là tín hiệu có ích và **giữ nguyên toàn bộ sai số trôi**.
3. **Bản chất so sánh hẹp:** LMS và RLS chỉ khác nhau ở thuật toán tối ưu hóa (tốc độ hội tụ vs. độ phức tạp) trên cùng một cấu trúc lọc FIR, không khai thác được cấu trúc trạng thái hay hợp nhất cảm biến của MPU-6050.

### Tại sao không ưu tiên **Kalman cố định vs. Kalman có tham số ước lượng lại bằng RLS**?
1. **Sai khớp với đặc tính mô hình hiện tại:** Trong [default.yaml](file:///home/duc/Workspace/master/random-signal-processing/config/default.yaml), các **tham số thống kê** của cảm biến ($N, B, T_B, K$) là **bất biến theo thời gian** trong suốt một phiên đo (chỉ giá trị nhiễu $x_B, x_K$ biến đổi theo thời gian). Dùng RLS ước lượng lại tham số online khi tham số thật vốn không đổi sẽ làm tăng phương sai ước lượng (kém hơn cả Kalman cố định đã nhận dạng tốt), trừ khi bạn cố tình mở rộng kịch bản thành *tham số cảm biến biến đổi đột ngột theo nhiệt độ* hoặc *tín hiệu động thay đổi phổ liên tục*.
2. **Bẫy toán học rất dễ làm sai:**
   - RLS dùng để ước lượng **hệ số tuyến tính** (như hệ số mô hình AR của tín hiệu hay $\Phi_B$), **không trực tiếp ước lượng được ma trận hiệp phương sai nhiễu $Q, R$** của Kalman.
   - Khi chạy RLS trực tiếp trên chuỗi đo $y[k] = u[k] + v[k]$ để ước lượng hệ số hồi quy, nhiễu đo $v[k]$ xuất hiện ở cả hai vế gây ra **ước lượng chệch (Errors-in-Variables bias)**; ghép ngược vào Kalman dễ gây dao động hoặc phân kỳ nếu không có kỹ thuật chống chệch (như Instrumental Variables) và kiểm soát hệ số quên $\lambda$.

*(Tóm lại: Phương án **Wiener vs. Kalman** và **Kalman 2 cấu trúc mô hình** vừa khớp hoàn hảo lý thuyết môn học, vừa phản ánh đúng ưu thế mô hình MPU-6050 bạn đã xây, lại sạch về mặt toán học.)*

---

## 4. Hướng dẫn từng bước thiết kế bộ lọc khi "coi mô hình cảm biến là không biết"

**Nguyên tắc cốt lõi (Black-box):** Coi `MPU6050.from_config("config/default.yaml")` như một con chip thật đóng kín. **Tuyệt đối không** đọc $N, B, T_B, K, b_0, S, M$ từ `default.yaml` để nạp vào bộ lọc, và **chỉ dùng** `ideal_*` ở Bước 4 để chấm điểm (RMSE/SNR).

### Bước 1: Thu thập dữ liệu thực nghiệm (tách biệt tập nhận dạng & tập kiểm thử)
Giống như đưa một cảm biến thật lên bàn thí nghiệm, bạn sinh 2 nhóm dữ liệu với `random_seed` khác nhau:

1. **Tập dữ liệu tĩnh dài hạn (dùng để nhận dạng nhiễu cảm biến):**
   - Chạy `sensor.simulate(...)` với đầu vào đứng yên (`ideal_acc = [0, 0, 9.80665]`, `ideal_gyr = [0, 0, 0]`) trong thời gian đủ dài (ví dụ **30–60 phút** ở $f_s = 200\text{ Hz}$).
   - *Lý do:* Khi đặt cảm biến đứng yên trên mặt phẳng, tín hiệu thật là hằng số đã biết, nên toàn bộ biến động đo được là biểu hiện thuần túy của quá trình nhiễu cảm biến.
2. **Tập dữ liệu động kiểm thử (dùng `random_seed` khác):**
   - Dùng các kịch bản như trong [basic_simulation.py](file:///home/duc/Workspace/master/random-signal-processing/examples/basic_simulation.py): đứng yên, hình sin đơn tần, đa tần số, và thêm 1 kịch bản dài hơn (60–300 giây) để thấy rõ ảnh hưởng của trôi bias.

### Bước 2: Nhận dạng thống kê & tham số nhiễu hoàn toàn từ dữ liệu tĩnh
Từ chuỗi đo tĩnh $y_{\text{static}}[k]$, tách thành phần nhiễu quan sát được $z[k] = y_{\text{static}}[k] - u_{\text{ref}}$ (với $u_{\text{ref}} = 0$, riêng $a_z$ thì $u_{\text{ref}} = g_0$):

1. **Ước lượng và bù sai số bias tĩnh (Calibration sơ bộ):**
   - Tính trung bình mẫu trên đoạn tĩnh đầu tiên: $\hat{b}_0 = \frac{1}{N_{\text{cal}}}\sum_{k=1}^{N_{\text{cal}}} z[k]$.
   - Trừ $\hat{b}_0$ ra khỏi dữ liệu đo trước khi đưa vào Wiener (vì Wiener giả định kỳ vọng bằng 0). *(Với Kalman Cấu trúc 2 có trạng thái bias, có thể giữ nguyên hoặc trừ bớt để kiểm chứng khả năng tự bù bias).*
2. **Phân tích hàm tự tương quan (ACF) và mật độ phổ công suất (PSD — Welch):**
   - Tính $\hat{R}_{zz}[\tau]$ và $\hat{S}_{zz}(f)$ trên chuỗi $z_0[k] = z[k] - \hat{b}_0$.
   - **Đọc kết quả từ đồ thị:**
     - Phần nền phẳng ở dải tần cao ($10\text{–}100\text{ Hz}$) $\Rightarrow$ xác nhận có nhiễu trắng và ước lượng được mức PSD nhiễu trắng $\hat{S}_{vv}$.
     - Phần PSD vọt lên mạnh ở tần số thấp ($f \to 0$) và ACF giảm rất chậm $\Rightarrow$ **bằng chứng trực tiếp từ dữ liệu** cho thấy cảm biến có nhiễu màu tương quan thời gian / trôi không dừng (căn cứ để thiết kế Cấu trúc Kalman 2).
3. **Ước lượng tham số không gian trạng thái bằng đường cong Allan Deviation (ADEV):**
   - Áp dụng công thức Allan Variance thực nghiệm (Eq. 23 trong [mathematical_model.md](file:///home/duc/Workspace/master/random-signal-processing/docs/mathematical_model.md#L224-L228)) trực tiếp lên tập dữ liệu tĩnh:
     - **Vùng dốc $-1/2$ ($\tau$ nhỏ):** $\hat{\sigma}_A(\tau) \approx \hat{N}/\sqrt{\tau} \Rightarrow$ đọc $\hat{N}$ tại $\tau = 1\text{ s}$, suy ra phương sai nhiễu đo rời rạc cho Kalman:
       $$\hat{R} = \hat{N}^2 f_s$$
     - **Vùng đáy phẳng (dốc $0$):** $\hat{\sigma}_{A,\min} \approx 0.664\,\hat{B} \Rightarrow$ ước lượng $\hat{B}$ và ước lượng thời gian tương quan $\hat{T}_B$ từ vị trí đáy / hoặc dùng khớp đường cong (curve fitting) phương trình (22) vào đường cong ADEV thực nghiệm để thu đồng thời $(\hat{N}, \hat{B}, \hat{T}_B, \hat{K})$.
     - **Vùng dốc $+1/2$ ($\tau$ lớn):** $\hat{\sigma}_A(\tau) \approx \hat{K}\sqrt{\tau/3} \Rightarrow$ đọc $\hat{K}$ tại $\tau = 3\text{ s}$.
   - Từ $(\hat{B}, \hat{T}_B, \hat{K})$ nhận dạng được, tính trực tiếp các tham số rời rạc $\hat{\Phi}_B = e^{-\Delta t/\hat{T}_B}$, $\hat{q}_B$, $\hat{q}_K$ theo Eq. (14)–(16) trong [mathematical_model.md](file:///home/duc/Workspace/master/random-signal-processing/docs/mathematical_model.md#L144-L162).

### Bước 3: Thiết kế cụ thể từng bộ lọc

#### A. Thiết kế Bộ lọc Wiener (cho Phương án 1)
Từ dữ liệu đo thực tế $y[k] = u[k] + v[k]$ (giả sử tín hiệu $u$ và nhiễu $v$ độc lập thống kê) và dữ liệu tĩnh đã có ở Bước 2:
- **Ước lượng phổ / tương quan tín hiệu sạch mà KHÔNG cần ground truth:**
  $$\hat{S}_{uu}(f) = \max\!\Big(\hat{S}_{yy}(f) - \hat{S}_{vv}(f),\; 0\Big), \qquad \hat{R}_{uu}[\tau] = \hat{R}_{yy}[\tau] - \hat{R}_{vv}[\tau]$$
  *(trong đó $\hat{S}_{yy}, \hat{R}_{yy}$ tính từ chuỗi đo $y[k]$; còn $\hat{S}_{vv}, \hat{R}_{vv}$ lấy từ tập tĩnh Bước 2).*
- **Cài đặt 1 trong 2 dạng (hoặc cả 2):**
  - *Wiener miền tần số (xử lý theo khối / offline):*
    $$H_W(f) = \frac{\hat{S}_{uu}(f)}{\hat{S}_{uu}(f) + \hat{S}_{vv}(f)}, \qquad \hat{U}(f) = H_W(f)\,Y(f)$$
  - *Wiener FIR nhân quả bậc $M$ (giải phương trình Wiener–Hopf):*
    $$\hat{\mathbf{R}}_{yy}\,\mathbf{w}^* = \hat{\mathbf{r}}_{uy} \approx \hat{\mathbf{r}}_{yy} - \hat{\mathbf{r}}_{vv}$$
    (thử vài bậc $M$, ví dụ $M \in \{15, 31, 63\}$, chọn bậc cân bằng giữa độ mượt và độ trễ).

#### B. Thiết kế Bộ lọc Kalman (cho Phương án 1 & Phương án 2)
1. **Cấu trúc 1 — Mô hình cơ bản (chỉ xét nhiễu đo trắng):**
   - Chọn mô hình động học tổng quát cho tín hiệu $u_k$ (vì không biết trước dạng sóng $u_k$), ví dụ mô hình **Constant Velocity (CV)** hoặc **Constant Acceleration (CA)**:
     $$\mathbf{x}_k = \begin{bmatrix} u_k \\ \dot{u}_k \end{bmatrix}, \quad \mathbf{F}_1 = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}, \quad \mathbf{H}_1 = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad R_1 = \hat{R}$$
   - Ma trận nhiễu quá trình $\mathbf{Q}_1 = q_u \begin{bmatrix} \Delta t^3/3 & \Delta t^2/2 \\ \Delta t^2/2 & \Delta t \end{bmatrix}$ với $q_u$ là tham số độ linh hoạt động học của chuyển động.
2. **Cấu trúc 2 — Mô hình mở rộng (có trạng thái nhiễu màu / trôi bias):**
   - *Nếu lọc đơn trục ($u_k$ biến thiên xoay chiều / đa tần):* Mở rộng thêm trạng thái bias $b_k$:
     $$\mathbf{x}_k = \begin{bmatrix} u_k \\ \dot{u}_k \\ b_k \end{bmatrix}, \quad \mathbf{F}_2 = \begin{bmatrix} 1 & \Delta t & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \hat{\Phi}_B \end{bmatrix}, \quad \mathbf{H}_2 = \begin{bmatrix} 1 & 0 & 1 \end{bmatrix}, \quad \mathbf{Q}_2 = \mathrm{blkdiag}(\mathbf{Q}_1,\; \hat{q}_B + \hat{q}_K)$$
     *(Lưu ý: ở chế độ đơn trục, nếu $u_k$ là hằng số thì $u_k$ và $b_k$ không thể tách rời nhau về mặt toán học; chúng chỉ tách được khi $u_k$ dao động còn $b_k$ trôi chậm).*
   - *Nếu làm bài toán hợp nhất 6 trục (ước lượng góc nghiêng $\theta_k$ từ Accel + Gyro — rất khuyến khích):*
     - **Cấu trúc 1 (1 trạng thái):** $\theta_{k+1} = \theta_k + \tilde{\omega}_k \Delta t + w_{\theta,k}$, đo $y_k = \theta_{\text{acc},k} = \theta_k + v_k$.
     - **Cấu trúc 2 (2 trạng thái $\mathbf{x}_k = [\theta_k,\, b_{g,k}]^T$):**
       $$\mathbf{x}_{k+1} = \begin{bmatrix} 1 & -\Delta t \\ 0 & \hat{\Phi}_B \end{bmatrix}\mathbf{x}_k + \begin{bmatrix} \Delta t \\ 0 \end{bmatrix}\tilde{\omega}_k + \mathbf{w}_k, \qquad y_k = \begin{bmatrix} 1 & 0 \end{bmatrix}\mathbf{x}_k + v_k$$
       với $R$ lấy từ phương sai góc đo bởi accel, và $Q$ lấy từ $(\hat{N}_g, \hat{q}_{B,g}, \hat{q}_{K,g})$ nhận dạng ở Bước 2.
3. **Tinh chỉnh tham số $q_u$ mà KHÔNG nhìn ground truth:**
   - Dựa vào chuỗi **phần dư dự đoán (Innovation)** $\nu_k = y_k - \mathbf{H}\hat{\mathbf{x}}_{k|k-1}$ và phương sai lý thuyết $S_k = \mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + R$:
     - Điều chỉnh $q_u$ sao cho **$\nu_k$ gần nhiễu trắng nhất** (ACF của $\nu_k$ tại lag $\tau \ge 1$ nằm trong dải tin cậy $\pm 1.96/\sqrt{N}$).
     - Kiểm tra **NIS (Normalized Innovation Squared)**: $\overline{\text{NIS}} = \frac{1}{N}\sum_{k=1}^N \frac{\nu_k^2}{S_k} \approx 1$ (với đo vô hướng). Nếu $\overline{\text{NIS}} \gg 1$ nghĩa là $q_u$ quá nhỏ (bộ lọc ì, bám không kịp tín hiệu); nếu $\overline{\text{NIS}} \ll 1$ nghĩa là $q_u$ quá lớn (lọc chưa đủ mượt).

### Bước 4: Đánh giá định lượng & Kiểm chứng (lúc này mới dùng `ideal_*`)
Chạy toàn bộ các bộ lọc trên cùng một bộ dữ liệu kiểm thử (cùng `random_seed` để công bằng) và lập bảng so sánh theo 5 tiêu chí:
1. **Độ chính xác:** RMSE trước lọc vs. sau lọc, và mức cải thiện SNR ($\Delta\text{SNR}\text{ [dB]}$).
2. **Độ lệch dài hạn (Bias error):** Sai số trung bình khi kéo dài thời gian mô phỏng (60–300 s) để làm nổi bật tác dụng khử trôi của Kalman Cấu trúc 2 so với Wiener và Kalman Cấu trúc 1.
3. **Độ méo tín hiệu động:** Độ trễ pha (phase lag) và suy hao biên độ trên kịch bản sin / đa tần số.
4. **Tính nhất quán thống kê:** Đồ thị ACF của Innovation và giá trị $\overline{\text{NIS}}$ (chứng minh bộ lọc hoạt động đúng lý thuyết ngay cả khi không có ground truth).
5. **Độ bền vững & Thời gian thực:** Quét tăng mức nhiễu / bias drift (như Kịch bản 4 trong repo) xem bộ lọc nào chịu nhiễu tốt hơn, kèm thời gian xử lý mỗi mẫu (so với chu kỳ lấy mẫu $\Delta t = 5\text{ ms}$ tại $200\text{ Hz}$).

---

Nếu bạn muốn, bước tiếp theo mình có thể giúp bạn chọn cụ thể **1 trong 2 kịch bản ứng dụng** (lọc khử nhiễu trực tiếp tín hiệu đo gia tốc/gyro, hoặc ước lượng góc nghiêng Roll/Pitch từ 6 trục Accel+Gyro) rồi viết luôn script Python chuẩn hóa toàn bộ quy trình từ **Bước 1 $\to$ Bước 4** chạy trực tiếp trên `mpu6050_sim`. Bạn muốn đi theo hướng nào?