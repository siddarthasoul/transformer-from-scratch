import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)

        x_hat = (x - mean) / std

        return self.gamma * x_hat + self.beta