# Báo cáo Lab 1: Khảo sát mạng học sâu CNN trên MNIST

## 1. Mô tả đề tài
- Đề tài: Nhận diện chữ số viết tay (0-9) trên bộ dữ liệu MNIST bằng mạng CNN.
- Mục tiêu:
  1. Chọn một chương trình nguồn mở và giải thích cơ chế cập nhật tham số bằng gradient.
  2. Dùng phương pháp parabol để thay cho phương pháp dựa trên gradient.

Trong báo cáo này, chương trình nguồn mở được chọn là **PyTorch** (https://pytorch.org), một framework deep learning phổ biến và hoàn toàn nguồn mở.

## 2. Mô hình CNN cho MNIST
Gọi tập huấn luyện là
$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{N}, \quad x_i\in\mathbb{R}^{1\times 28\times 28},\; y_i\in\{0,1,\dots,9\}.
$$

Kiến trúc CNN đơn giản:
- Convolution 1: $1\to 16$ kênh, kernel $3\times 3$, ReLU, MaxPool $2\times2$.
- Convolution 2: $16\to 32$ kênh, kernel $3\times 3$, ReLU, MaxPool $2\times2$.
- Fully Connected 1: $32\cdot 7\cdot 7 \to 64$, ReLU.
- Fully Connected 2: $64 \to 10$ (logits).

Với một mẫu $x$, đầu ra trước softmax là vector logits
$$
z = f(x;\theta)\in\mathbb{R}^{10},
$$
trong đó $\theta$ là toàn bộ tham số (trọng số và bias) của mạng.

Xác suất dự đoán lớp $k$:
$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{10} e^{z_j}}.
$$

Hàm mất mát cross-entropy cho một mẫu:
$$
\ell(x,y;\theta) = -\log p_y.
$$

Dạng tương đương theo logits (đúng với cách cài đặt thủ công trong notebook):
$$
\ell = \log\left(\sum_{k=1}^{10} e^{z_k}\right) - z_y.
$$

Hàm mất mát trung bình trên mini-batch $B$:
$$
L_B(\theta)=\frac{1}{|B|}\sum_{(x,y)\in B}\ell(x,y;\theta).
$$

## 3. Cơ chế cập nhật tham số bằng gradient trong CNN (PyTorch)

### 3.1. Lan truyền thuận
Với từng lớp, ta tính toán theo thứ tự:
$$
a^{(l)} = W^{(l)} * h^{(l-1)} + b^{(l)} \quad (\text{với conv, } * \text{ là phép tích chập}),
$$
$$
h^{(l)} = \sigma\big(a^{(l)}\big),
$$
trong đó $\sigma$ có thể là ReLU: $\sigma(t)=\max(0,t)$.

Sau cùng thu được logits $z$ và hàm mất mát $L_B(\theta)$. Về bản chất, toàn mạng là hợp của nhiều hàm:
$$
z=f(x;\theta)=f^{(L)}\circ f^{(L-1)}\circ \cdots \circ f^{(1)}(x).
$$
Vì loss phụ thuộc vào đầu ra cuối cùng, nên mọi tham số ở các lớp trước đều ảnh hưởng gián tiếp đến loss thông qua chuỗi phép biến đổi này.

### 3.2. Lan truyền ngược (Backpropagation)
Mục tiêu là tính gradient:
$$
\nabla_\theta L_B(\theta).
$$

Lý do phải lan truyền ngược:
- Ta cần gradient của **toàn bộ** tham số để cập nhật, không chỉ lớp cuối.
- Nếu dùng sai phân hữu hạn cho từng tham số $\theta_j$:
$$
\frac{\partial L}{\partial \theta_j}\approx \frac{L(\theta+\varepsilon e_j)-L(\theta-\varepsilon e_j)}{2\varepsilon},
$$
thì với $P$ tham số cần khoảng $2P$ lần forward cho mỗi bước học, rất tốn kém khi $P$ lớn.
- Backprop tính tất cả gradient trong một lần backward bằng cách tái sử dụng các đại lượng trung gian, nên hiệu quả hơn rất nhiều.

Theo quy tắc dây chuyền:
$$
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l)}}\cdot \frac{\partial a^{(l)}}{\partial W^{(l)}},
\qquad
\frac{\partial L}{\partial b^{(l)}} = \frac{\partial L}{\partial a^{(l)}}.
$$

Với lớp softmax + cross-entropy, đạo hàm theo logits có dạng rất gọn:
$$
\frac{\partial \ell}{\partial z_k}=p_k-\mathbf{1}(k=y).
$$

Vì sao mỗi lớp có gradient riêng:
- Mỗi lớp có bộ tham số riêng $\{W^{(l)}, b^{(l)}\}$.
- Mức độ ảnh hưởng của từng lớp đến loss khác nhau, nên cần đạo hàm riêng:
$$
\left\{\frac{\partial L}{\partial W^{(1)}},\dots,\frac{\partial L}{\partial W^{(L)}}\right\},\quad
\left\{\frac{\partial L}{\partial b^{(1)}},\dots,\frac{\partial L}{\partial b^{(L)}}\right\}.
$$
- Optimizer sẽ dùng đúng gradient của từng lớp để cập nhật đúng tham số lớp đó.

Vì sao lớp convolution cũng tuân theo chain rule:
- Mỗi phần tử đầu ra convolution vẫn là hàm khả vi của kernel và đầu vào:
$$
y_{u,v,c_o}=\sum_{c_i}\sum_{m,n} K_{c_o,c_i,m,n}\,x_{u+m,v+n,c_i}+b_{c_o}.
$$
- Do đó vẫn áp dụng quy tắc dây chuyền như các lớp khác:
$$
\frac{\partial L}{\partial K_{c_o,c_i,m,n}}
=\sum_{u,v}\frac{\partial L}{\partial y_{u,v,c_o}}\cdot x_{u+m,v+n,c_i}.
$$
Sai số được truyền ngược qua các lớp FC, pooling, convolution, ReLU, tạo thành gradient cho tất cả tham số.

Trong PyTorch:
- `loss = cross_entropy_from_logits(outputs, labels)` tính $L_B$ (hàm tự cài đặt trong notebook, không dùng `nn.CrossEntropyLoss`).
- `loss.backward()` tự động tính toán toàn bộ gradient bằng autograd.

### 3.3. Cập nhật tham số

#### a) Gradient Descent/SGD
$$
\theta_{t+1}=\theta_t - \eta\nabla_\theta L_B(\theta_t),
$$
với $\eta$ là learning rate.

Viết theo từng lớp:
$$
W^{(l)}_{t+1}=W^{(l)}_t-\eta\frac{\partial L_B}{\partial W^{(l)}},\qquad
b^{(l)}_{t+1}=b^{(l)}_t-\eta\frac{\partial L_B}{\partial b^{(l)}}.
$$
Điều này cho thấy rõ vai trò “mỗi lớp một gradient riêng” trong quá trình cập nhật.

Trong notebook đi kèm, bước cập nhật này được cài đặt thủ công theo đúng Gradient Descent cơ bản:
$$
\theta \leftarrow \theta - \eta \nabla_\theta L,
$$
không dùng `optim.SGD`.

Tóm lại, cơ chế học tham số bằng gradient trong CNN gồm:
1. Forward để tính loss.
2. Backward để tính gradient.
3. Optimizer để cập nhật tham số.

## 4. Phương pháp parabol thay cho gradient

### 4.1. Ý tưởng
Thay vì tính $\nabla_\theta L$ trực tiếp, ta tối ưu theo một hướng tìm kiếm $d_t$ và xấp xỉ hàm 1 biến
$$
\phi(\alpha)=L_B(\theta_t+\alpha d_t)
$$
bằng một đa thức bậc 2 (parabol).

Chọn ba điểm đối xứng:
$$
\alpha_1=-\delta,\quad \alpha_2=0,\quad \alpha_3=+\delta,
$$
và tính:
$$
L_- = \phi(-\delta),\quad L_0 = \phi(0),\quad L_+ = \phi(+\delta).
$$

### 4.2. Nội suy parabol
Giả sử
$$
q(\alpha)=a\alpha^2+b\alpha+c
$$
nội suy qua ba điểm trên.

Với bộ ba điểm đối xứng, ta có:
$$
a = \frac{L_+ + L_- -2L_0}{2\delta^2},
\qquad
b = \frac{L_+ - L_-}{2\delta},
\qquad
c=L_0.
$$

Ý nghĩa của các hệ số:
- $b$ xấp xỉ độ dốc bậc 1 của $L$ theo hướng $d_t$ tại lân cận $\alpha=0$.
- $a$ xấp xỉ độ cong bậc 2 theo hướng đó.
- Muốn có điểm cực tiểu tin cậy thì cần $a>0$ (parabol mở lên).

Điểm cực tiểu của parabol:
$$
\alpha^* = -\frac{b}{2a}
= \frac{\delta\,(L_- - L_+)}{2\,(L_- -2L_0 + L_+)}.
$$

Nếu mẫu số gần 0 hoặc $a\le 0$ (không bảo đảm có cực tiểu theo mô hình bậc 2), ta dùng cơ chế an toàn:
- Chọn giá trị tốt nhất trong $\{-\delta,0,+\delta\}$,
- Và giới hạn $\alpha^*$ trong đoạn $[-\alpha_{\max},\alpha_{\max}]$ để tránh bước nhảy quá lớn.

### 4.3. Cập nhật tham số không dùng gradient
Sau khi có $\alpha_t$:
$$
\theta_{t+1}=\theta_t + \alpha_t d_t.
$$

Quy trình 1 mini-batch (đúng với notebook):
1. Lưu bản sao tham số hiện tại $\theta_t$.
2. Sinh hướng ngẫu nhiên chuẩn hóa $d_t$.
3. Đánh giá loss tại 3 điểm: $L_-,L_0,L_+$ tương ứng với $\alpha=-\delta,0,+\delta$.
4. Tính $\alpha_t$ bằng công thức parabol; nếu không ổn định thì dùng cơ chế an toàn.
5. Cập nhật tham số: $\theta_{t+1}=\theta_t+\alpha_t d_t$.
6. Tính lại loss/accuracy sau cập nhật để ghi log huấn luyện.

Lưu ý quan trọng:
- Cách này **không cần** `loss.backward()` và không dùng gradient trực tiếp.
- Chỉ cần đánh giá hàm mất mát tại vài điểm (zero-order optimization).
- Chi phí mỗi bước cao hơn vì cần nhiều lần forward (trong code hiện tại là 4 lần mỗi mini-batch: 3 lần để tìm $\alpha_t$ và 1 lần để ghi metric sau cập nhật).

### 4.4. Chọn hướng $d_t$
Trong notebook, hướng $d_t$ được lấy ngẫu nhiên và chuẩn hóa:
$$
d_t \sim \mathcal{N}(0,I), \qquad d_t \leftarrow \frac{d_t}{\|d_t\|_2}.
$$

Các siêu tham số đang dùng trong notebook:
- `delta = 0.02`: khoảng thăm dò hai phía quanh $\alpha=0$.
- `alpha_clip = 0.05`: giới hạn biên độ bước nhảy để tránh cập nhật quá lớn.

Như vậy, mỗi bước học tìm bước nhảy tối ưu cục bộ trên đường thẳng tham số $\theta_t+\alpha d_t$ bằng nội suy parabol, thay cho việc dùng gradient trực tiếp.

## 5. So sánh gradient vs parabol

### 5.1. Gradient (backprop)
- Ưu điểm:
  - Chính xác theo đạo hàm bậc 1.
  - Hội tụ nhanh hơn trên mạng lớn.
  - Có thể triển khai đơn giản bằng SGD cơ bản và dễ diễn giải về mặt toán học.
- Nhược điểm:
  - Cần autograd/backprop.
  - Có thể gặp vấn đề vanishing/exploding gradient nếu mô hình sâu.

### 5.2. Parabol (không gradient)
- Ưu điểm:
  - Không cần tính đạo hàm.
  - Dùng được cho bài toán gradient khó tính/không khả vi.
- Nhược điểm:
  - Tốn nhiều lần forward mỗi cập nhật.
  - Thường hội tụ chậm hơn và nhạy cảm với tham số $\delta$, $\alpha_{\max}$.
  - Khả năng mở rộng cho mạng rất lớn kém hơn backprop.

## 6. Nội dung notebook đi kèm
Notebook `Lab1_MNIST_CNN.ipynb` gồm 2 phần huấn luyện:
1. CNN + cập nhật bằng Gradient Descent cơ bản trên MNIST.
2. CNN + cập nhật bằng phương pháp parabol.

Thiết lập dữ liệu trong notebook:
- Cả hai phương pháp đều dùng toàn bộ tập train MNIST gồm 60,000 mẫu để so sánh công bằng.
- Hàm mất mát cross-entropy được tự cài từ logits.

Notebook được chia theo nguyên tắc: **mỗi cell code đều có 1 cell markdown ngay phía trước** để giải thích rõ mục đích.

## 7. Kết luận
- Cơ chế học tham số chuẩn trong CNN là tối ưu dựa trên gradient thông qua backpropagation.
- Phương pháp parabol cho phép cập nhật tham số theo hướng không gradient bằng cách nội suy hàm mất mát theo 1 biến.
- Trên MNIST, cách parabol vẫn có thể học được, nhưng thường chậm và kém hiệu quả hơn gradient.
- Tuy nhiên, đây là hướng tiếp cận có giá trị học thuật để hiểu rõ sự khác nhau giữa tối ưu bậc 1 (gradient-based) và tối ưu không đạo hàm (derivative-free).
