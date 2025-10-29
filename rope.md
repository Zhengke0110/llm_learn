# RoPE (Rotary Position Embedding) 数学公式

## 1. precompute_cis - 预计算旋转矩阵

### 频率计算公式

$$\theta_i = \theta^{-\frac{2i}{d}}, \quad i = 0, 1, ..., \frac{d}{2}-1$$

其中：

- $d$ 是特征维度（head_dim）
- $\theta$ 是基础频率参数（默认为10000）
- $i$ 是维度对的索引

### 复数形式（欧拉公式）

$$f_{m,i} = e^{im\theta_i} = \cos(m\theta_i) + i\sin(m\theta_i)$$

其中：

- $m$ 是序列中的位置索引，$m \in [0, end)$
- $f_{m,i}$ 是位置 $m$ 在维度 $i$ 上的旋转因子

---

## 2. reshape_for_broadcast - 形状调整

### 形状变换公式

$$(seq\_len, \frac{d}{2}) \rightarrow (1, seq\_len, 1, ..., 1, \frac{d}{2})$$

这个变换使得旋转矩阵可以在 batch 和 head 维度上进行广播操作。

---

## 3. apply_rotary_pos_emb - 应用旋转编码

### 复数转换

将每两个连续的实数维度转换为一个复数：
$$z_i = x_{2i} + ix_{2i+1}$$

### 旋转操作（核心公式）

$$q'_{m} = q_{m} \odot e^{im\theta}$$
$$k'_{m} = k_{m} \odot e^{im\theta}$$

其中 $\odot$ 表示复数的逐元素相乘（旋转操作）。

### 复数乘法展开形式

$$(a + bi) \times (\cos\theta + i\sin\theta) = (a\cos\theta - b\sin\theta) + i(a\sin\theta + b\cos\theta)$$

### 等价的矩阵旋转形式

$$\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$$

这是一个标准的2D旋转矩阵，证明了 RoPE 本质上是对每对维度进行旋转变换。

---

## RoPE 的核心思想

RoPE 通过复数乘法实现位置编码，具有以下特点：

1. **相对位置编码**：两个位置 $m$ 和 $n$ 的 query-key 点积只依赖于相对位置 $(m-n)$
2. **远程衰减**：随着相对距离增大，不同频率的旋转使得相关性自然衰减
3. **外推能力**：可以处理比训练时更长的序列
4. **无需额外参数**：完全通过数学变换实现，不需要学习额外的位置编码参数
