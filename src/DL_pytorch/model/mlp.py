import torch
import torch.nn as nn

from src.DL_pytorch.model.linear import Linear

class MLP(nn.Module):

    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = Linear(d_model, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
