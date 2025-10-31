import math
import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from rope import precompute_cis, reshape_for_broadcast, apply_rotary_pos_emb


@dataclasses.dataclass
class ModelArgs:
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: Optional[int] = None
    max_batch_size: int = 32
    max_seq_len: int = 2048


def repeat_kv(x, n_rep):
    bz, seq_len, n_kv_heads, head_dim = (
        x.shape
    )  # 输入: [batch, seq_len, n_kv_heads, head_dim]
    if n_rep == 1:
        return x  # 不需要重复时直接返回
    return (
        x[:, :, :, None, :]  # 添加新维度: [batch, seq_len, n_kv_heads, 1, head_dim]
        .expand(
            bz, seq_len, n_kv_heads, n_rep, head_dim
        )  # 扩展: [batch, seq_len, n_kv_heads, n_rep, head_dim]
        .reshape(
            bz, seq_len, n_kv_heads * n_rep, head_dim
        )  # 重塑: [batch, seq_len, n_kv_heads*n_rep, head_dim]
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 如果n_kv_heads为None，则使用n_heads作为默认值
        n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size  # 本地query头数: 8
        self.n_kv_heads = n_kv_heads // model_parallel_size  # 本地key-value头数: 8

        # 假设KV的头和Q的头数量是不一致的，所以需要将KV的头的数量复制到和Q相同的数量
        self.n_rep = self.n_local_heads // self.n_kv_heads  # KV头重复次数: 8//8=1

        self.head_dim = args.dim // args.n_heads  # 每个头的维度: 512//8=64

        # 线性变换层，输入维度->输出维度
        self.wq = nn.Linear(
            args.dim, args.n_heads * self.head_dim, bias=False
        )  # [512] -> [512]
        self.wk = nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False
        )  # [512] -> [512]
        self.wv = nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=False
        )  # [512] -> [512]
        self.wo = nn.Linear(
            args.n_heads * self.head_dim, args.dim, bias=False
        )  # [512] -> [512]

        # KV缓存，存储之前计算的key和value
        self.cache_k = torch.zeros(
            args.max_batch_size,
            args.max_seq_len,
            self.n_kv_heads,
            self.head_dim,  # [32, 2048, 8, 64]
        )
        self.cache_v = torch.zeros(
            args.max_batch_size,
            args.max_seq_len,
            self.n_kv_heads,
            self.head_dim,  # [32, 2048, 8, 64]
        )

    def forward(self, x, start_pos, freqs_cis, mask):
        # 1. x -> wq, wk, wv -> q, k, v
        # 2. q, k, v (shape [batch, seq_len, dim])-> view -> dim -> head*head_dim(拆多头) -> [batch, seq_len, head*head_dim]
        # 3. q, k -> rope ->softmax(q*k^T/dim)*v -> output -> wo -> output -> [batch, seq_len, dim]

        bz, seq_len, _ = x.shape  # 输入: [batch, seq_len, dim] = [1, 50, 512]

        # 线性变换得到query, key, value
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  # 各自: [1, 50, 512]

        # 重塑为多头格式
        xq = xq.view(bz, seq_len, self.n_local_heads, self.head_dim)  # [1, 50, 8, 64]
        xk = xk.view(bz, seq_len, self.n_kv_heads, self.head_dim)  # [1, 50, 8, 64]
        xv = xv.view(bz, seq_len, self.n_kv_heads, self.head_dim)  # [1, 50, 8, 64]

        # 应用旋转位置编码
        xq, xk = apply_rotary_pos_emb(
            xq, xk, freqs_cis=freqs_cis
        )  # 形状不变: [1, 50, 8, 64]

        # 更新KV缓存
        self.cache_k[:bz, start_pos : start_pos + seq_len] = xk  # 写入缓存
        self.cache_v[:bz, start_pos : start_pos + seq_len] = xv  # 写入缓存

        # 从缓存中获取完整的keys和values（包括历史）
        keys = self.cache_k[:bz, : start_pos + seq_len]  # [1, start_pos+seq_len, 8, 64]
        values = self.cache_v[
            :bz, : start_pos + seq_len
        ]  # [1, start_pos+seq_len, 8, 64]

        # 重复KV头以匹配query头数（GQA机制）
        keys = repeat_kv(keys, self.n_rep)  # [1, start_pos+seq_len, 8, 64]
        values = repeat_kv(values, self.n_rep)  # [1, start_pos+seq_len, 8, 64]

        # 转置以便矩阵乘法: [batch, n_heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)  # [1, 8, 50, 64]
        keys = keys.transpose(1, 2)  # [1, 8, start_pos+seq_len, 64]
        values = values.transpose(1, 2)  # [1, 8, start_pos+seq_len, 64]

        # 计算注意力分数: Q @ K^T / sqrt(head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )  # [1, 8, 50, start_pos+seq_len]

        # 应用因果掩码（如果提供）
        if mask is not None:
            scores = scores + mask  # 广播加法

        # 计算注意力权重
        scores = F.softmax(scores.float(), dim=-1).type_as(
            xq
        )  # [1, 8, 50, start_pos+seq_len]

        # 应用注意力权重到values
        output = torch.matmul(scores, values)  # [1, 8, 50, 64]

        # 转置并重塑回原始格式
        output = (
            output.transpose(1, 2).contiguous().view(bz, seq_len, -1)
        )  # [1, 50, 512]

        # 输出线性变换
        return self.wo(output)  # [1, 50, 512]


# 第一次需要mask
def create_mask(seq_len, n_heads):
    mask = torch.triu(
        torch.ones((seq_len, seq_len)), diagonal=1
    )  # 上三角矩阵: [seq_len, seq_len]
    mask = mask.masked_fill(mask == 1, float("-inf")).masked_fill(
        mask == 0, float(0.0)
    )  # 上三角填充-inf
    mask = mask.repeat(n_heads, 1, 1)  # 复制给每个头: [n_heads, seq_len, seq_len]

    return mask


# args = ModelArgs(dim=512, n_heads=8, max_batch_size=32, max_seq_len=200)

# attention = Attention(args)

# x = torch.randn(1, 50, 512)  # 输入: [batch=1, seq_len=50, dim=512]

# # 创建Mask
# mask = create_mask(50, args.n_heads)  # [8, 50, 50]
# mask = mask.unsqueeze(0).expand(1, -1, -1, -1)  # 扩展batch维度: [1, 8, 50, 50]

# # 预计算旋转位置编码
# freqs_cis = precompute_cis(
#     dim=args.dim // args.n_heads, end=args.max_seq_len * 2
# )  # [400, 32]

# freqs_cis_1 = freqs_cis[:50, :]  # 取前50个位置: [50, 32]

# # 第一次前向传播
# output = attention(
#     x, start_pos=0, freqs_cis=freqs_cis_1, mask=mask
# )  # 输出: [1, 50, 512]

# print(output.shape)  # [1, 50, 512]
