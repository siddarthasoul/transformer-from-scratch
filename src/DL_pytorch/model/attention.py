import math
import torch
import torch.nn as nn

from src.DL_pytorch.model.linear import Linear
from src.DL_pytorch.model.softmax import Softmax
from src.DL_pytorch.model.rope import RoPE


class MHA(nn.Module):

    def __init__(self, d_model: int, heads: int):
        super().__init__()

        assert d_model % heads == 0

        self.d_model = d_model
        self.n_head = heads
        self.d_k = d_model // heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        # Your current RoPE works on (B,T,D)
        self.q_rope = RoPE(d_model)
        self.k_rope = RoPE(d_model)

        self.softmax = Softmax(dim=-1)

    def forward(self, x):

        B, T, D = x.shape

        # ---------- Q K V ----------
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ---------- RoPE ----------
        q = self.q_rope(q)
        k = self.k_rope(k)

        # ---------- Split Heads ----------
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # ---------- Attention ----------
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)

        # ---------- Causal Mask ----------
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        scores = scores.masked_fill(
            mask.unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )

        # ---------- Softmax ----------
        weights = self.softmax(scores)

        # ---------- Weighted Sum ----------
        output = torch.matmul(weights, v)

        # ---------- Merge Heads ----------
        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(B, T, D)
        )

        # ---------- Output Projection ----------
        output = self.out_proj(output)


        return output




