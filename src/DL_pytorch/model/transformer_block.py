import torch
import torch.nn as nn

from src.DL_pytorch.model.attention import MHA
from src.DL_pytorch.model.layernorm import LayerNorm
from src.DL_pytorch.model.mlp import MLP


class TransformerBlock(nn.Module):

    def __init__(self, d_model, heads):

        super().__init__()

        self.ln1 = LayerNorm(d_model)

        self.attn = MHA(d_model, heads)

        self.ln2 = LayerNorm(d_model)

        self.mlp = MLP(d_model, d_model * 4)


    def forward(self, x):

        attn_out = self.attn(self.ln1(x))

        x = x + attn_out

        mlp_out = self.mlp(self.ln2(x))

        x = x + mlp_out

        return x