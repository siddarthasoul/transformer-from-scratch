from src.DL_numpy.model.transformer_block import TransformerBlock
from src.DL_numpy.model.embedding import Embedding
from src.DL_numpy.model.layernorm import LayerNorm
from src.DL_numpy.model.lmHead import LmHead


class Transformer:

    def __init__(self, vocab_size, seq_length, d_model, heads, hidden_dim, num_layers):
        self.embedding = Embedding(vocab_size, d_model)
        
        self.blocks = [
            TransformerBlock(
                d_model,
                heads,
                hidden_dim,
                seq_length,
            )
            for _ in range(num_layers)
        ]
        
        # Final LayerNorm before Language Model Head
        self.norm = LayerNorm(d_model)
        self.lm_head = LmHead(d_model, vocab_size)

    def forward(self, token):
        # 1. Embed tokens
        x = self.embedding.forward(token)

        # 2. Pass through Transformer Blocks
        for block in self.blocks:
            x = block.forward(x)

        # 3. Final normalization
        x = self.norm.forward(x)

        # 4. Project to vocabulary logits
        logits = self.lm_head.forward(x)

        return logits

    def backward(self, d_logits):
        # 1. Backprop through LmHead
        grad = self.lm_head.backward(d_logits)

        # 2. Backprop through Final LayerNorm
        grad = self.norm.backward(grad)

        # 3. Backprop through Transformer Blocks in reverse order
        for block in reversed(self.blocks):
            grad = block.backward(grad)

        # 4. Backprop through Embedding layer
        grad = self.embedding.backward(grad)

        return grad

    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())

        for block in self.blocks:
            params.extend(block.parameters())

        params.extend(self.norm.parameters())
        params.extend(self.lm_head.parameters())

        return params