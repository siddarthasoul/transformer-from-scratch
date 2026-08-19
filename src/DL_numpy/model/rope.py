import numpy as np


class RoPE:

    def __init__(self, base=10000.0):
        self.base = base

    def _get_cos_sin(self, seq_len, dim):
        # Position indices: (seq_len, 1)
        pos = np.arange(seq_len)[:, None]

        # Dimension indices for pairs: (dim // 2,)
        i = np.arange(0, dim, 2)
        inv_freq = 1.0 / (self.base ** (i / dim))

        # Frequencies theta: (seq_len, dim // 2)
        angles = pos * inv_freq

        # Compute cos and sin, repeat each value twice for (v1, v2) pairs
        cos = np.repeat(np.cos(angles), 2, axis=-1)  # (seq_len, dim)
        sin = np.repeat(np.sin(angles), 2, axis=-1)  # (seq_len, dim)

        return cos, sin

    def _rotate_half(self, x):
        """
        Swaps and negates paired elements: [-v2, v1, -v4, v3, ...]
        """
        x_rot = np.zeros_like(x)
        x_rot[..., 0::2] = -x[..., 1::2]
        x_rot[..., 1::2] = x[..., 0::2]
        return x_rot

    def forward(self, x):
        self.batch_size, self.seq_len, self.dim = x.shape

        # Compute cos and sin shapes: (1, seq_len, dim) for broadcasting
        cos, sin = self._get_cos_sin(self.seq_len, self.dim)
        self.cos = cos[None, :, :]
        self.sin = sin[None, :, :]

        # R_theta * x = x * cos(theta) + rotate_half(x) * sin(theta)
        out = x * self.cos + self._rotate_half(x) * self.sin
        return out

    def backward(self, grad):
        # The transpose rotation matrix uses -sin(theta)
        # R_theta^T * grad = grad * cos(theta) - rotate_half(grad) * sin(theta)
        dx = grad * self.cos - self._rotate_half(grad) * self.sin
        return dx