import numpy as np
from src.DL_numpy.model.parameter import Parameter


class LmHead:

    def __init__(self, d_model, vocab_size):
        # Xavier/He initialization
        self.weight = Parameter(
            np.random.randn(d_model, vocab_size) / np.sqrt(d_model)
        )

    def forward(self, x):
        self.x = x
        logits = x @ self.weight.data
        return logits

    def backward(self, d_logits):
        orig_shape = self.x.shape
        d_model = orig_shape[-1]
        vocab_size = d_logits.shape[-1]

        # Reshape to 2D for clean matrix multiplication
        x_2d = self.x.reshape(-1, d_model)
        grad_2d = d_logits.reshape(-1, vocab_size)

        # Calculate weight gradient
        dweight = x_2d.T @ grad_2d

        # Accumulate gradients safely (+=)
        if self.weight.grad is None:
            self.weight.grad = dweight
        else:
            self.weight.grad += dweight

        # Gradient with respect to inputs (dx)
        dx_2d = grad_2d @ self.weight.data.T

        return dx_2d.reshape(orig_shape)

    def parameters(self):
        return [self.weight]