from src.DL_numpy.model.attention import MultiHeadAttention
from src.DL_numpy.model.mlp import Mlp


class TransformerBlock:

    def __init__(self, d_model, heads, hidden_dim, seq_length):
        self.mha = MultiHeadAttention(seq_length, d_model, heads)
        self.mlp = Mlp(d_model, hidden_dim)

    def forward(self, x):
        attn = self.mha.forward(x)
        out = self.mlp.forward(attn)
        return out

    def backward(self, grad):
        grad = self.mlp.backward(grad)
        grad = self.mha.backward(grad)
        return grad

    def parameters(self):
        params = []
        params.extend(self.mha.parameters())
        params.extend(self.mlp.parameters())
        return params