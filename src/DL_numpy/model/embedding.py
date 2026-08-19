import numpy as np
from src.DL_numpy.model.parameter import Parameter


class Embedding:

    def __init__(self, vocab_size, d_model):
        self.weight = Parameter(
            np.random.randn(vocab_size, d_model) * 0.02
        )

    def forward(self, token):
        self.token = token
        return self.weight.data[token]

    def backward(self, grad):

        if self.weight.grad is None:
            self.weight.grad = np.zeros_like(self.weight.data)

        np.add.at(self.weight.grad, self.token, grad)

        return None

    def parameters(self):
        return [self.weight]