import numpy as np
from src.DL_numpy.model.layernorm import LayerNorm
from src.DL_numpy.model.linear import Linear
from src.DL_numpy.model.rope import RoPE
from src.DL_numpy.model.softmax import Softmax


class MultiHeadAttention:

    def __init__(self, seq_length, d_model, heads):
        self.seq_len = seq_length
        self.d_model = d_model
        self.head = heads
        self.head_dim = self.d_model // self.head

        self.q_rope = [RoPE() for _ in range(self.head)]
        self.k_rope = [RoPE() for _ in range(self.head)]
        self.softmax = [Softmax() for _ in range(self.head)]

        self.layer_norm = LayerNorm(self.d_model)

        self.q_linear = [
            Linear(self.head_dim, self.head_dim) for _ in range(self.head)
        ]
        self.k_linear = [
            Linear(self.head_dim, self.head_dim) for _ in range(self.head)
        ]
        self.v_linear = [
            Linear(self.head_dim, self.head_dim) for _ in range(self.head)
        ]

        self.out_linear = Linear(d_model, d_model)

    def forward(self, x):
        self.x = x
        batch_size, seq_len, _ = x.shape

        self.Q = []
        self.K = []
        self.V = []
        self.head_inputs = []
        self.scores = []
        self.weights = []
        self.outputs = []
        self.masks = []

        X_heads = x.reshape(batch_size, seq_len, self.head, self.head_dim)
        self.X_heads = X_heads

        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        for h in range(self.head):
            Xh = X_heads[:, :, h, :]
            self.head_inputs.append(Xh)

            Q = self.q_linear[h].forward(Xh)
            K = self.k_linear[h].forward(Xh)
            V = self.v_linear[h].forward(Xh)

            Q = self.q_rope[h].forward(Q)
            K = self.k_rope[h].forward(K)

            # Scaled Dot-Product Attention
            scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.head_dim)

            # Apply Causal Mask
            scores = np.where(mask[None, :, :], -1e9, scores)

            weights = self.softmax[h].forward(scores)
            out = weights @ V

            self.scores.append(scores)
            self.Q.append(Q)
            self.K.append(K)
            self.V.append(V)
            self.weights.append(weights)
            self.outputs.append(out)
            self.masks.append(mask)

        self.concat = np.concatenate(self.outputs, axis=-1)
        final = self.out_linear.forward(self.concat)

        # Pre-LN / Residual connection
        self.residual = final + self.x
        attention_output = self.layer_norm.forward(self.residual)

        return attention_output

    def backward(self, grad):
        # 1. Backprop through LayerNorm
        grad_norm = self.layer_norm.backward(grad)

        # 2. Gradient splits into out_linear path and residual path
        grad_residual = grad_norm.copy()
        grad_out = self.out_linear.backward(grad_norm)

        batch_size, seq_len, _ = grad_out.shape

        grad_heads = grad_out.reshape(
            batch_size, seq_len, self.head, self.head_dim
        )
        grad_input = np.zeros_like(self.X_heads)

        for h in range(self.head):
            grad_head = grad_heads[:, :, h, :]

            # dV = W^T @ dOut
            dV = self.weights[h].transpose(0, 2, 1) @ grad_head

            # dWeights = dOut @ V^T
            dWeights = grad_head @ self.V[h].transpose(0, 2, 1)

            # Backprop Softmax
            dScores = self.softmax[h].backward(dWeights)

            # Zero out gradients for masked future positions!
            dScores = np.where(self.masks[h][None, :, :], 0.0, dScores)

            # Scale factor 1/sqrt(d_k) applies to dScores for Q and K derivatives
            dScores_scaled = dScores / np.sqrt(self.head_dim)

            dQ = dScores_scaled @ self.K[h]
            dK = dScores_scaled.transpose(0, 2, 1) @ self.Q[h]

            # Backprop RoPE
            dQ = self.q_rope[h].backward(dQ)
            dK = self.k_rope[h].backward(dK)

            # Backprop Linear projections
            dXq = self.q_linear[h].backward(dQ)
            dXk = self.k_linear[h].backward(dK)
            dXv = self.v_linear[h].backward(dV)

            grad_input[:, :, h, :] = dXq + dXk + dXv

        grad_input = grad_input.reshape(batch_size, seq_len, self.d_model)

        # Add residual gradient back
        grad_input += grad_residual

        return grad_input

    def parameters(self):
        params = []
        for h in range(self.head):
            params.extend(self.q_linear[h].parameters())
            params.extend(self.k_linear[h].parameters())
            params.extend(self.v_linear[h].parameters())

        params.extend(self.out_linear.parameters())
        params.extend(self.layer_norm.parameters())

        return params