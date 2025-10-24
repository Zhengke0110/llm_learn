# 三种归一化方法原理

## 1. BatchNorm (批归一化)

**核心思想**：在batch和sequence维度上归一化，每个特征维度共享统计量

**公式**：
$$
\text{output} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

其中：

- $\mu = \text{mean}(x, \text{dim}=(0, 1))$：在batch和sequence维度上计算均值，形状为 $(1, 1, d)$
- $\sigma = \text{std}(x, \text{dim}=(0, 1))$：在batch和sequence维度上计算标准差，形状为 $(1, 1, d)$
- $\epsilon = 10^{-5}$：防止除零的小常数

**特点**：每个特征维度有一个全局的均值和标准差

---

## 2. LayerNorm (层归一化)

**核心思想**：在特征维度上归一化，每个位置独立计算统计量

**公式**：
$$
\text{output} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

其中：

- $\mu = \text{mean}(x, \text{dim}=-1)$：在特征维度上计算均值，形状为 $(N, L, 1)$
- $\sigma = \text{std}(x, \text{dim}=-1)$：在特征维度上计算标准差，形状为 $(N, L, 1)$
- $\epsilon = 10^{-5}$：防止除零的小常数

**特点**：每个样本的每个位置有独立的均值和标准差，不依赖batch size

---

## 3. RMSNorm (均方根归一化)

**核心思想**：LayerNorm的简化版本，省略均值中心化，只做缩放归一化

**公式**：
$$
\text{output} = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

其中：

- $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon} = \sqrt{\text{mean}(x^2) + \epsilon}$
- $\gamma$：可学习的缩放参数（权重）
- $\epsilon$：防止除零的小常数

**实现**：
$$
\text{output} = x \cdot \frac{1}{\sqrt{\text{mean}(x^2, \text{dim}=-1) + \epsilon}} \cdot \gamma
$$

**特点**：

- 不减去均值，计算更高效（比LayerNorm快7%-64%）
- 在Transformer模型中广泛使用（如LLaMA、GPT等）

---

## 核心区别对比

| 归一化方法 | 归一化维度 | 统计量形状 | 是否中心化 |
|-----------|-----------|-----------|-----------|
| BatchNorm | dim=(0,1) | (1,1,d) | ✓ |
| LayerNorm | dim=-1 | (N,L,1) | ✓ |
| RMSNorm | dim=-1 | (N,L,1) | ✗ |

其中 $N$ = batch size, $L$ = sequence length, $d$ = feature dimension
