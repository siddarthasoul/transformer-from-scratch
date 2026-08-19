import numpy as np


seq_len = 4
d_model = 8
heads = 2
hidden_dim = 32

gamma = np.ones(d_model)
beta = np.zeros(d_model)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def residual(x,a):
    y = x+a
    return y;



def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x = (x - mean) / np.sqrt(var + eps)

    return gamma * x + beta

def relu(x):
    return np.maximum(0, x)

X = np.random.randn(seq_len, d_model)

print("Original")
print(X)

head_dim = d_model // heads

X_heads = X.reshape(seq_len, heads, head_dim)

print("\nAfter Split")
print(X_heads)

print("\nHead 1")
print(X_heads[:,0,:])

print("\nHead 2")
print(X_heads[:,1,:])

Wq1 = np.random.randn(head_dim, head_dim) / np.sqrt(head_dim)
Wk1 = np.random.randn(head_dim, head_dim) / np.sqrt(head_dim)
Wv1 = np.random.randn(head_dim, head_dim) / np.sqrt(head_dim)

Wq2 = np.random.randn(head_dim, head_dim) / np.sqrt(head_dim)
Wk2 = np.random.randn(head_dim, head_dim) / np.sqrt(head_dim)
Wv2 = np.random.randn(head_dim, head_dim) / np.sqrt(head_dim)

X1 = X_heads[:,0,:]

Q1 = X1 @ Wq1
K1 = X1 @ Wk1
V1 = X1 @ Wv1

scores1 = (Q1 @ K1.T) / np.sqrt(head_dim)

W1 = softmax(scores1)
h1_output = W1 @ V1

X2 = X_heads[:,1,:]

Q2 = X2 @ Wq2
K2 = X2 @ Wk2
V2 = X2 @ Wv2

scores2 = (Q2 @ K2.T) / np.sqrt(head_dim)

W2 = softmax(scores2)
h2_output = W2 @ V2

final = np.concatenate([h1_output, h2_output], axis=-1)

Wo = np.random.randn(d_model, d_model) / np.sqrt(d_model)

final = final @ Wo


res = residual(final, X)
attention_output = layer_norm(res, gamma, beta)





W1 = np.random.randn(d_model, hidden_dim) / np.sqrt(d_model)
b1 = np.zeros(hidden_dim)

W2 = np.random.randn(hidden_dim, d_model) / np.sqrt(hidden_dim)
b2 = np.zeros(d_model)



H = attention_output @ W1 + b1
print("After Linear1:", H.shape)

H = relu(H)
print("After ReLU:", H.shape)

Y = H @ W2 + b2
print("After Linear2:", Y.shape)

Y = Y + attention_output
print("After Residual:", Y.shape)

output = layer_norm(Y, gamma, beta)
print("After LayerNorm:", Y.shape)