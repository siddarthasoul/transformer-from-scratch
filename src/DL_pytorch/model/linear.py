import torch
import torch.nn as nn

class Linear(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) * 0.02
        )

        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

    def forward(self, x):

        return x @ self.weight + self.bias


