import numpy as np
from src.DL_numpy.model.parameter import Parameter


class LayerNorm:

    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = Parameter(np.ones(d_model))
        self.beta = Parameter(np.zeros(d_model))

    def forward(self, x):
        self.x = x

        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)

        self.x_hat = (x - self.mean) / self.std

        return self.gamma.data * self.x_hat + self.beta.data

    def backward(self, grad):
        N = self.x.shape[-1]
        sum_axes = tuple(range(grad.ndim - 1))

        # Calculate parameter gradients
        dgamma = np.sum(grad * self.x_hat, axis=sum_axes)
        dbeta = np.sum(grad, axis=sum_axes)

        # Accumulate gradients into Parameter objects (+= safely)
        if self.gamma.grad is None:
            self.gamma.grad = dgamma
        else:
            self.gamma.grad += dgamma

        if self.beta.grad is None:
            self.beta.grad = dbeta
        else:
            self.beta.grad += dbeta

        # Gradient through scale & standard deviation
        dx_hat = grad * self.gamma.data

        # Analytical LayerNorm gradient formula (Correct)
        dx = (
            (1.0 / (N * self.std))
            * (
                N * dx_hat
                - np.sum(dx_hat, axis=-1, keepdims=True)
                - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)
            )
        )

        return dx

    def parameters(self):
        return [self.gamma, self.beta]