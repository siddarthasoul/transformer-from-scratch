import torch
import torch.nn as nn

class Embedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.waight = nn.Parameter(
            torch.rand(vocab_size, d_model) * 0.2
        )

    def forward(self, token_ids):

        embeddings = self.waight[token_ids]

        return embeddings
