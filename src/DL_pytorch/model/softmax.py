import torch
import torch.nn as nn

class Softmax(nn.Module):

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x - torch.max(x, dim=self.dim, keepdim=True).values

        e = torch.exp(x)

        softmax = e / torch.sum(e, dim=self.dim, keepdim=True)

        return softmax



