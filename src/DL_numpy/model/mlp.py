import numpy as np
from src.DL_numpy.model.layernorm import LayerNorm
from src.DL_numpy.model.linear import Linear


class Mlp:

    def __init__(self, d_model, hidden_dim):
        self.hidden_dim = hidden_dim
        self.d_model = d_model

        self.layer_norm = LayerNorm(d_model)

        self.fc1 = Linear(d_model, hidden_dim)
        self.fc2 = Linear(hidden_dim, d_model)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_backward(self, grad):
        # Pass gradient through ReLU derivative
        grad_relu = grad.copy()
        grad_relu[self.h1 <= 0] = 0
        return grad_relu

    def forward(self, attention_output):
        self.x = attention_output

        self.h1 = self.fc1.forward(self.x)
        self.h2 = self.relu(self.h1)
        self.y = self.fc2.forward(self.h2)

        # Residual connection before LayerNorm
        self.residual = self.y + self.x
        self.out = self.layer_norm.forward(self.residual)

        return self.out

    def backward(self, grad):
        # 1. Backprop through LayerNorm -> returns d(residual)
        grad_norm = self.layer_norm.backward(grad)

        # 2. Residual connection splits gradient into fc2 path and skip path
        grad_skip = grad_norm.copy()

        # 3. Backprop through MLP layers
        grad_fc2 = self.fc2.backward(grad_norm)
        grad_relu = self.relu_backward(grad_fc2)
        grad_fc1 = self.fc1.backward(grad_relu)

        # 4. Combine gradients from MLP path and residual skip path
        total_grad_input = grad_fc1 + grad_skip

        return total_grad_input

    def parameters(self):
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.layer_norm.parameters())
        return params