# FFN (Feed-Forward Network) 数学公式分析

## 概述

FFN是Transformer架构中的关键组件，用于对每个位置的特征进行非线性变换。本文档分析了 `ffn.py` 文件中涉及的数学公式及其与代码行的对应关系。

## 数学公式与代码对应关系

### 1. 隐藏层维度计算

**数学公式**:

分步计算过程：

1. $$h_1 = \left\lfloor\frac{2h_{in}}{3}\right\rfloor$$
2. $$h_2 = \begin{cases}
   \lfloor h_1 \times \text{multiplier} \rfloor & \text{if multiplier} \neq \text{None} \\
   h_1 & \text{otherwise}
   \end{cases}$$
3. $$h_{final} = \text{multiple\_of} \times \left\lfloor\frac{h_2 + \text{multiple\_of} - 1}{\text{multiple\_of}}\right\rfloor$$

**对应代码行**: 第11-15行

```python
hidden_dim = int(2 * hidden_dim / 3)
if ffn_dim_multiplier is not None:
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
```

**参数说明**:

- $h_{in}$ = `hidden_dim` (初始输入的隐藏层维度参数)
- $\text{multiplier}$ = `ffn_dim_multiplier` (维度放大系数)
- $\text{multiple\_of}$ = `multiple_of` (对齐基数，确保维度是其整数倍)

**作用**: 确保隐藏层维度是 `multiple_of` 的整数倍，有利于硬件加速。

**注**: 第3步公式 $\left\lfloor\frac{h_2 + \text{multiple\_of} - 1}{\text{multiple\_of}}\right\rfloor$ 等价于向上取整 $\left\lceil\frac{h_2}{\text{multiple\_of}}\right\rceil$。

**计算示例**:

- 输入参数：`hidden_dim=2048`, `multiple_of=64`, `ffn_dim_multiplier=1`
- 步骤1：$h_1 = \lfloor\frac{2 \times 2048}{3}\rfloor = \lfloor 1365.33 \rfloor = 1365$
- 步骤2：$h_2 = \lfloor 1365 \times 1 \rfloor = 1365$
- 步骤3：$h_{final} = 64 \times \lfloor\frac{1365 + 64 - 1}{64}\rfloor = 64 \times \lfloor 22.31 \rfloor = 64 \times 22 = 1408$

### 2. SwiGLU激活函数

**数学公式**:

$$\text{FFN}(x) = W_2(\text{SiLU}(W_1x) \odot W_3x)$$

其中，SiLU (Sigmoid Linear Unit) 的定义为：

$$\text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}} = \frac{x}{1 + e^{-x}}$$

**对应代码行**: 第23行

```python
return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**参数说明**:

- $W_1$ = `self.w1` (第一个线性变换，维度: dim → hidden_dim)
- $W_2$ = `self.w2` (第二个线性变换，维度: hidden_dim → dim)
- $W_3$ = `self.w3` (第三个线性变换，维度: dim → hidden_dim)
- $\odot$ 表示逐元素乘法 (Hadamard product)
- $\text{SiLU}$ = `F.silu` (Sigmoid Linear Unit激活函数)

**作用**: 通过非线性变换增强模型的表达能力。

## 变体设计

### 1. 标准FFN vs SwiGLU FFN

标准FFN的公式：

$$\text{FFN}_{\text{standard}}(x) = W_2(\text{GELU}(W_1x))$$

SwiGLU FFN的公式（本实现采用）：

$$\text{FFN}_{\text{SwiGLU}}(x) = W_2(\text{SiLU}(W_1x) \odot W_3x)$$

### 2. 维度变化过程

输入输出维度的变化：

$$x \in \mathbb{R}^{b \times s \times d} \xrightarrow{W_1, W_3} \mathbb{R}^{b \times s \times h} \xrightarrow{\text{SiLU}, \odot} \mathbb{R}^{b \times s \times h} \xrightarrow{W_2} \mathbb{R}^{b \times s \times d}$$

其中：

- $b$ = batch size
- $s$ = sequence length
- $d$ = 输入/输出维度 (dim)
- $h$ = 隐藏层维度 (hidden_dim)

## 实现细节

### 1. 偏置项设置

所有线性变换均不使用偏置项（bias=False）：

```python
self.w1 = nn.Linear(dim, hidden_dim, bias=False)
self.w2 = nn.Linear(hidden_dim, dim, bias=False)
self.w3 = nn.Linear(dim, hidden_dim, bias=False)
```

### 2. 维度调整策略

1. 首先将输入维度缩减：$h_1 = \lfloor\frac{2 \times \text{hidden\_dim}}{3}\rfloor$
2. 应用可选的维度乘数：$h_2 = \lfloor h_1 \times \text{ffn\_dim\_multiplier}\rfloor$（如果提供）
3. 向上对齐到multiple_of的整数倍：使用 $(h_2 + \text{multiple\_of} - 1) // \text{multiple\_of}$ 实现向上取整

## 总结

本实现的FFN具有以下特点：

1. **使用SwiGLU激活**: 相比标准GELU，提供更好的性能
2. **无偏置设计**: 所有线性层都不使用偏置项，减少参数量
3. **维度对齐**: 确保隐藏层维度是指定值的整数倍，有利于硬件加速
4. **可配置放大倍数**: 通过ffn_dim_multiplier灵活调整网络容量
