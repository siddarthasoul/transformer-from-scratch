import numpy as np


class Adam:

    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        # Store parameters as a fixed list
        self.parameters = list(parameters)

        # Hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Time step counter
        self.t = 0

        # Initialize moments in parallel lists
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def zero_grad(self):
        """Reset all parameter gradients to zero."""
        for p in self.parameters:
            if hasattr(p, "grad") and p.grad is not None:
                p.grad.fill(0)

    def step(self):
        """Update weights using Adam moment estimates."""
        self.t += 1

        # Bias correction factors for step t
        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad

            # Update first and second moment estimates using list index i
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias-corrected moments
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            # Apply parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)