import torch 
import torch.nn as nn

from src.DL_pytorch.model.linear import Linear

class LMHead(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.proj = Linear(d_model, vocab_size)

    def forward(self, x):

        return self.proj(x)