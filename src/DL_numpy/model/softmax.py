import numpy as np


class Softmax:

    def forward(self, x):

        self.x = x

        x = x - np.max(x, axis=-1, keepdims=True)

        exp = np.exp(x)

        self.probs = exp / np.sum(exp, axis=-1, keepdims=True)

        return self.probs


    def backward(self, grad):
        """
        grad:
            dL/dSoftmax

        Returns:
            dL/dLogits
        """

        dot = np.sum(
            grad * self.probs,
            axis=-1,
            keepdims=True,
        )

        dx = self.probs * (grad - dot)

        return dx