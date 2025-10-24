import torch
import torch.nn as nn

# 定义张量的维度：batch批次大小、sequence_length序列长度、dim特征维度
batch, sequence_length, dim = 3, 4, 5

# 生成随机输入张量，形状为 (batch, sequence_length, dim) = (3, 4, 5)
input = torch.randn(batch, sequence_length, dim)

# 创建BatchNorm1d层，特征维度为dim，affine=False表示不使用可学习的缩放和偏移参数
batch_norm = nn.BatchNorm1d(dim, affine=False)

# ==================== 方法1：使用PyTorch API进行批归一化 ====================
# BatchNorm1d要求输入格式为 (N, C, L)，其中N是batch，C是通道数（特征维度），L是序列长度
# 我们的输入是 (N, L, C) 格式，需要先转置

# 步骤：(N, L, C) -> transpose -> (N, C, L) -> BatchNorm1d -> (N, C, L) -> transpose -> (N, L, C)
batch_norm_input_api = batch_norm(input.transpose(-1, -2)).transpose(-1, -2)

# ==================== 方法2：手动实现批归一化计算 ====================
# 批归一化的核心公式：output = (input - mean) / sqrt(variance + epsilon)

# 计算均值：在batch维度(0)和sequence_length维度(1)上求平均，保持维度以便广播
# 结果形状：(1, 1, dim)，表示每个特征维度的全局均值
batch_norm_mean = input.mean(dim=(0, 1), keepdim=True)

# 计算标准差：在batch维度(0)和sequence_length维度(1)上计算标准差
# unbiased=False 使用总体标准差（除以N），而非样本标准差（除以N-1）
# 结果形状：(1, 1, dim)，表示每个特征维度的全局标准差
batch_norm_std = input.std(dim=(0, 1), keepdim=True, unbiased=False)

# 执行归一化：(输入 - 均值) / (标准差 + 小常数)
# 1e-5是一个很小的常数，防止除零错误
# 这个操作会将数据标准化为均值为0，标准差为1的分布
batch_norm_input_1 = (input - batch_norm_mean) / (batch_norm_std + 1e-5)

# ==================== BatchNorm 结果输出 ====================
print("BatchNorm1d API output:\n", batch_norm_input_api)
print("BatchNorm1d manual output:\n", batch_norm_input_1)


print("=" * 100)

# ==================== 方法3：使用PyTorch API进行层归一化 ====================
# LayerNorm (层归一化) 与 BatchNorm 的核心区别：
# - BatchNorm: 在batch和sequence维度上归一化，对每个特征维度计算统计量
# - LayerNorm: 在特征维度上归一化，对每个样本的每个位置独立计算统计量
#
# LayerNorm优势：
# 1. 不依赖batch size，适合小batch或在线学习
# 2. 每个样本独立归一化，不受其他样本影响
# 3. 在RNN和Transformer中表现更好
#
# elementwise_affine=False 表示不使用可学习的缩放(γ)和偏移(β)参数
layer_norm = nn.LayerNorm(
    dim, elementwise_affine=False
)  # 创建LayerNorm层，特征维度为dim

# 直接对输入应用LayerNorm，无需转置
# LayerNorm默认在最后一个维度(dim=-1)上进行归一化
batch_norm_api = layer_norm(input)

# ==================== 方法4：手动实现层归一化计算 ====================
# LayerNorm的核心公式：output = (input - mean) / sqrt(variance + epsilon)
# 关键：在特征维度(dim=-1)上计算每个位置的统计量

# 计算均值：在特征维度(-1)上求平均，保持维度以便后续广播
# 对于形状为(3, 4, 5)的输入：
# - 在每个(batch, sequence_length)位置上，对5个特征求平均
# - 结果形状：(3, 4, 1)，表示3×4=12个位置，每个位置有1个均值
layer_norm_mean = input.mean(dim=-1, keepdim=True)

# 计算标准差：在特征维度(-1)上计算标准差
# unbiased=False: 使用总体标准差公式 σ = sqrt(Σ(x-μ)²/N)
# unbiased=True:  使用样本标准差公式 s = sqrt(Σ(x-μ)²/(N-1))
# 结果形状：(3, 4, 1)，表示12个位置，每个位置有1个标准差
layer_norm_std = input.std(dim=-1, keepdim=True, unbiased=False)

# 执行归一化：(输入 - 均值) / (标准差 + 小常数)
# 广播机制：
# - input: (3, 4, 5)
# - layer_norm_mean: (3, 4, 1) -> 广播到 (3, 4, 5)
# - layer_norm_std: (3, 4, 1) -> 广播到 (3, 4, 5)
# 1e-5 防止标准差为0时的除零错误
# 每个位置的5个特征会被独立归一化为均值0、标准差1
layer_norm_input_1 = (input - layer_norm_mean) / (layer_norm_std + 1e-5)

print("LayerNorm API output:\n", batch_norm_api)
print("LayerNorm manual output:\n", layer_norm_input_1)

