import numpy as np
from src.DL_numpy.model.parameter import Parameter


class Linear:
    """
    Fully Connected (Linear) Layer

    Forward:
        y = x @ W + b

    Backward:
        Computes & accumulates gradients for:
            - W
            - b
            - x
    """

    def __init__(self, in_features, out_features):
        # Xavier / He initialization
        self.weight = Parameter(
            np.random.randn(in_features, out_features) / np.sqrt(in_features)
        )
        self.bias = Parameter(np.zeros(out_features))

    def forward(self, x):
        self.x = x
        return x @ self.weight.data + self.bias.data

    def backward(self, grad):
        # Preserve original shape for return
        orig_shape = self.x.shape
        in_features = orig_shape[-1]
        out_features = grad.shape[-1]

        # Reshape inputs & gradients to 2D matrix: (N, in_features) and (N, out_features)
        x_2d = self.x.reshape(-1, in_features)
        g_2d = grad.reshape(-1, out_features)

        # Compute parameter gradients
        dweight = x_2d.T @ g_2d
        dbias = np.sum(g_2d, axis=0)

        # Safely accumulate gradients (+=)
        if self.weight.grad is None:
            self.weight.grad = dweight
        else:
            self.weight.grad += dweight

        if self.bias.grad is None:
            self.bias.grad = dbias
        else:
            self.bias.grad += dbias

        # Compute gradient wrt input: dx = g @ W^T
        grad_input_2d = g_2d @ self.weight.data.T

        # Reshape back to match input's original dimensions
        return grad_input_2d.reshape(orig_shape)

    def parameters(self):
        return [self.weight, self.bias]