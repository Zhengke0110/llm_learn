from dataclasses import dataclass
from typing import Optional
from attention import Attention
from ffn import FeedForward
from norm import RMSNorm
from rope import precompute_cis
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096  # 每个词的向量维度（词的"DNA长度"）
    n_layers: int = 2  # Transformer层数（处理的层数）
    n_heads: int = 8  # 注意力头数量
    n_kv_heads: Optional[int] = None
    vocab_size: int = 1000  # 词汇表大小
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048  # 最大序列长度


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim

        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args=args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.dim * 4,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)

        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(dim=params.dim, eps=params.norm_eps)

        self.outout = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.freqs_cis = precompute_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens, start_pos):
        bsz, seq_len = tokens.size()  # bsz=2（2个句子），seq_len=200（每句200个词）

        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seq_len, start_pos)), mask])

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)

        output = self.outout(h).float()
        return output


args = ModelArgs()

llama = Transformer(args)

x = torch.randint(0, args.vocab_size, (2, 200))

print(llama(x, 0).shape)
