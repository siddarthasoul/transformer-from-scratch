import torch
import torch.nn as nn


class RoPE(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()

        self.d_model = d_model

        pair_index = torch.arange(d_model // 2, dtype=torch.float32)

        inv_freq = 1.0 / (
            10000 ** (2 * pair_index / d_model)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):

        B, T, D = x.shape

        x = x.reshape(B, T, D // 2, 2)

        position = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)[:, None]

        inv_freq = self.inv_freq[None, :]

        angle = position * inv_freq

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        x1 = x[..., 0]
        x2 = x[..., 1]

        rot_x = x1 * cos - x2 * sin
        rot_y = x1 * sin + x2 * cos

        out = torch.stack([rot_x, rot_y], dim=-1)

        return out.reshape(B, T, D)