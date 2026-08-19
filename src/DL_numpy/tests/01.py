import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Input text
# -----------------------------
text = "The cat sat on the mat"
tokens = text.split()

# -----------------------------
# 2. Build tiny vocabulary
# -----------------------------
vocab = {w: i for i, w in enumerate(sorted(set(tokens)))}
token_ids = [vocab[t] for t in tokens]

print("Vocabulary:", vocab)
print("Token IDs :", token_ids)

# -----------------------------
# 3. Embedding lookup
# -----------------------------
d_model = 4
embedding = np.random.randn(len(vocab), d_model)

X = embedding[token_ids]

print("\nEmbedding Shape:", X.shape)

# -----------------------------
# 4. Q K V
# -----------------------------
Wq = np.random.randn(d_model, d_model) / np.sqrt(d_model)
Wk = np.random.randn(d_model, d_model) / np.sqrt(d_model)
Wv = np.random.randn(d_model, d_model) / np.sqrt(d_model)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

# -----------------------------
# 5. Attention
# -----------------------------
scores = (Q @ K.T) / np.sqrt(d_model)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

weights = softmax(scores)
output = weights @ V

# -----------------------------
# 6. Print matrices
# -----------------------------
np.set_printoptions(precision=3, suppress=True)

print("\nEmbeddings\n", X)
print("\nQ\n", Q)
print("\nK\n", K)
print("\nV\n", V)
print("\nScores\n", scores)
print("\nWeights\n", weights)
print("\nAttention Output\n", output)

# -----------------------------
# 7. Visualization
# -----------------------------
fig, axs = plt.subplots(2, 3, figsize=(14,8))

plots = [
    ("Embeddings", X),
    ("Q", Q),
    ("K", K),
    ("Scores", scores),
    ("Weights", weights),
    ("Attention Output", output),
]

for ax, (title, data) in zip(axs.flat, plots):
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)

    if data.shape[0] == len(tokens):
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)

    if data.shape[1] == len(tokens):
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}",
                    ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.show()
