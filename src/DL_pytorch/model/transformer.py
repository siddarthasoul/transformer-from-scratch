import torch
import torch.nn as nn

from src.DL_pytorch.model.embedding import Embedding
from src.DL_pytorch.model.transformer_block import TransformerBlock
from src.DL_pytorch.model.layernorm import LayerNorm
from src.DL_pytorch.model.lmHead import LMHead


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, heads, num_layers):

        super().__init__()

        self.embedding = Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    heads,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(d_model)
        self.lm_head = LMHead(d_model, vocab_size)


    def forward(self, x):

        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits