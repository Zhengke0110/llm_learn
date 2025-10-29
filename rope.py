import torch


def precompute_cis(dim, end, theta=10000):
    """
    预计算RoPE的旋转矩阵（复数形式）

    参数:
        dim: 特征维度，必须是偶数
        end: 序列最大长度
        theta: 基础频率，默认10000
    返回:
        shape为(end, dim//2)的复数张量
    """
    # 计算频率: 1 / (theta^(2i/dim))，i为维度索引
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 生成位置索引: [0, 1, 2, ..., end-1]
    m = torch.arange(end)

    # 计算位置和频率的外积，得到每个位置在每个维度上的旋转角度
    freqs = torch.outer(m, freqs)

    # 将角度转为复数: e^(i*angle) = cos(angle) + i*sin(angle)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    """
    调整freqs_cis的形状以匹配x，用于广播操作

    参数:
        freqs_cis: shape为(seq_len, dim//2)
        x: 输入张量
    返回:
        重塑后的freqs_cis，可与x进行广播
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # 只保留序列维度(索引1)和最后维度，其他维度设为1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_pos_emb(xq, xk, freqs_cis):
    """
    对query和key应用旋转位置编码

    参数:
        xq: query张量
        xk: key张量
        freqs_cis: 预计算的旋转矩阵
    返回:
        应用RoPE后的(xq, xk)
    """
    # 将实数张量转为复数: 每两个维度组成一个复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 调整freqs_cis形状以匹配xq_和xk_
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 复数乘法实现旋转，然后转回实数
    xq_out = torch.view_as_real((xq_ * freqs_cis)).flatten(3)
    xk_out = torch.view_as_real((xk_ * freqs_cis)).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


dim = 2
end = 3

fres_cos = precompute_cis(dim, end)
xq = torch.rand(1, end, dim)
xk = torch.rand(1, end, dim)
apply_rotary_pos_emb(xq, xk, freqs_cis=fres_cos)
