import torch

from src.DL_pytorch.model.embedding import Embedding


embedding = Embedding(
    vocab_size=20,
    d_model=4
)

tokens = torch.tensor([
    [2, 5, 2]
])

print("=" * 60)
print("Embedding Weight")
print("=" * 60)
print(embedding.waight)

emb = embedding(tokens)

print("\nEmbedding Output")
print(emb)

# Fake loss
loss = emb.sum()

print("\nLoss")
print(loss)

loss.backward()

print("\nGradient Shape")
print(embedding.waight.grad.shape)

print("\nGradient Matrix")
print(embedding.waight.grad)