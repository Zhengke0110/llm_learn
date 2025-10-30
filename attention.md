# Attention 机制数学公式与代码对应关系

## 1. 线性变换公式

### 数学公式

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

### 对应代码

- **第 82 行**: `xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)`
- **第 49-55 行**: 线性层定义

  ```python
  self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
  self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
  self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
  ```

---

## 2. 多头重塑公式

### 数学公式

$$Q = \text{reshape}(Q, [B, L, H, d_h])$$

其中 $d_h = d / H$（每个头的维度）

### 对应代码

- **第 85-87 行**:

  ```python
  xq = xq.view(bz, seq_len, self.n_local_heads, self.head_dim)
  xk = xk.view(bz, seq_len, self.n_kv_heads, self.head_dim)
  xv = xv.view(bz, seq_len, self.n_kv_heads, self.head_dim)
  ```

---

## 3. RoPE 旋转位置编码

### 数学公式

**频率计算：**
$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, \frac{d}{2}-1$$

**位置编码矩阵：**
$$f_{m,i} = e^{im\theta_i} = \cos(m\theta_i) + i\sin(m\theta_i)$$

**旋转变换：**
$$\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}$$

### 对应代码

- **第 90-92 行**: 应用 RoPE

  ```python
  xq, xk = apply_rotary_pos_emb(xq, xk, freqs_cis=freqs_cis)
  ```

- **第 147-149 行**: RoPE 预计算

  ```python
  freqs_cis = precompute_cis(dim=args.dim // args.n_heads, end=args.max_seq_len * 2)
  ```

---

## 4. KV 缓存机制

### 数学公式

**缓存更新：**
$$\text{cache}_K[:, \text{start\_pos}:\text{start\_pos}+L] = K_{\text{new}}$$
$$\text{cache}_V[:, \text{start\_pos}:\text{start\_pos}+L] = V_{\text{new}}$$

**完整序列获取：**
$$K_{\text{full}} = \text{cache}_K[:, :\text{start\_pos}+L]$$
$$V_{\text{full}} = \text{cache}_V[:, :\text{start\_pos}+L]$$

### 对应代码

- **第 95-96 行**: 缓存更新

  ```python
  self.cache_k[:bz, start_pos : start_pos + seq_len] = xk
  self.cache_v[:bz, start_pos : start_pos + seq_len] = xv
  ```

- **第 99-102 行**: 获取完整序列

  ```python
  keys = self.cache_k[:bz, : start_pos + seq_len]
  values = self.cache_v[:bz, : start_pos + seq_len]
  ```

---

## 5. 分组查询注意力 (GQA)

### 数学公式

**KV 头重复：**
$$K_{\text{repeated}} = \text{repeat}(K, n_{\text{rep}}), \quad n_{\text{rep}} = \frac{H_Q}{H_{KV}}$$

**重复操作：**
$$\text{repeat}(X, n) : [B, L, H_{KV}, d_h] \rightarrow [B, L, H_{KV} \times n, d_h]$$

### 对应代码

- **第 105-106 行**: KV 头重复

  ```python
  keys = repeat_kv(keys, self.n_rep)
  values = repeat_kv(values, self.n_rep)
  ```

- **第 20-32 行**: `repeat_kv`函数实现

  ```python
  x[:, :, :, None, :].expand(bz, seq_len, n_kv_heads, n_rep, head_dim)
  .reshape(bz, seq_len, n_kv_heads * n_rep, head_dim)
  ```

---

## 6. 维度转置

### 数学公式

$$Q, K, V : [B, L, H, d_h] \rightarrow [B, H, L, d_h]$$

### 对应代码

- **第 109-111 行**:

  ```python
  xq = xq.transpose(1, 2)      # [1, 8, 50, 64]
  keys = keys.transpose(1, 2)  # [1, 8, seq_len, 64]
  values = values.transpose(1, 2)  # [1, 8, seq_len, 64]
  ```

---

## 7. 注意力分数计算

### 数学公式

**缩放点积：**
$$S = \frac{QK^T}{\sqrt{d_h}}, \quad S \in \mathbb{R}^{B \times H \times L_Q \times L_K}$$

### 对应代码

- **第 114-116 行**:

  ```python
  scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
  ```

---

## 8. 因果掩码应用

### 数学公式

$$S_{\text{masked}} = S + M$$

其中掩码矩阵：
$$M_{i,j} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

### 对应代码

- **第 119-120 行**: 掩码应用

  ```python
  if mask is not None:
      scores = scores + mask
  ```

- **第 132-138 行**: 掩码创建

  ```python
  mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
  mask = mask.masked_fill(mask == 1, float("-inf")).masked_fill(mask == 0, float(0.0))
  ```

---

## 9. Softmax 归一化

### 数学公式

$$A = \text{softmax}(S_{\text{masked}}) = \frac{\exp(S_{\text{masked}})}{\sum_{k=1}^{L_K} \exp(S_{\text{masked}}[k])}$$

### 对应代码

- **第 123-125 行**:

  ```python
  scores = F.softmax(scores.float(), dim=-1).type_as(xq)
  ```

---

## 10. 加权求和

### 数学公式

$$O = AV, \quad O \in \mathbb{R}^{B \times H \times L_Q \times d_h}$$

### 对应代码

- **第 128 行**:

  ```python
  output = torch.matmul(scores, values)  # [1, 8, 50, 64]
  ```

---

## 11. 输出重塑和投影

### 数学公式

**重塑和输出投影：**
$$O_{\text{concat}} = \text{reshape}(O^T, [B, L_Q, H \times d_h])$$
$$\text{Output} = O_{\text{concat}}W^O$$

### 对应代码

- **第 131-133 行**: 重塑

  ```python
  output = output.transpose(1, 2).contiguous().view(bz, seq_len, -1)  # [1, 50, 512]
  ```

- **第 136 行**: 输出投影

  ```python
  return self.wo(output)  # [1, 50, 512]
  ```

---

## 12. 完整的多头注意力公式

### 数学公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

其中每个头：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

基础注意力：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 对应代码

**整个 forward 方法 (第 74-136 行)** 实现了这个完整公式！

---

## 维度变换总结

### 完整流程的维度变化

1. **输入**: $X \in \mathbb{R}^{B \times L \times d}$ → `[1, 50, 512]`
2. **线性变换**: $Q, K, V \in \mathbb{R}^{B \times L \times d}$ → `[1, 50, 512]`
3. **多头重塑**: $Q, K, V \in \mathbb{R}^{B \times L \times H \times d_h}$ → `[1, 50, 8, 64]`
4. **转置**: $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$ → `[1, 8, 50, 64]`
5. **注意力分数**: $S \in \mathbb{R}^{B \times H \times L \times L}$ → `[1, 8, 50, 50]`
6. **输出**: $O \in \mathbb{R}^{B \times H \times L \times d_h}$ → `[1, 8, 50, 64]`
7. **重塑**: $O \in \mathbb{R}^{B \times L \times d}$ → `[1, 50, 512]`

### 参数说明

- $B$: batch size (1)
- $L$: sequence length (50)
- $H$: number of heads (8)
- $d$: embedding dimension (512)
- $d_h$: head dimension (64 = 512/8)
