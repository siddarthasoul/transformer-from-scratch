import torch

print("=" * 60)
print("1. Create Tensor")
print("=" * 60)

x = torch.arange(24).reshape(2, 3, 4)

print(x)
print("Shape:", x.shape)

# --------------------------------------------------

print("\n" + "=" * 60)
print("2. reshape()")
print("=" * 60)

y = x.reshape(6, 4)

print(y)
print(y.shape)

# --------------------------------------------------

print("\n" + "=" * 60)
print("3. view()")
print("=" * 60)

z = x.view(6, 4)

print(z)
print(z.shape)

# --------------------------------------------------

print("\n" + "=" * 60)
print("4. transpose()")
print("=" * 60)

t = x.transpose(1, 2)

print(t.shape)
print(t)

# --------------------------------------------------

print("\n" + "=" * 60)
print("5. permute()")
print("=" * 60)

p = x.permute(2, 0, 1)

print(p.shape)
print(p)

# --------------------------------------------------

print("\n" + "=" * 60)
print("6. matmul()")
print("=" * 60)

A = torch.randn(2, 3)

B = torch.randn(3, 4)

C = torch.matmul(A, B)

print(A.shape)
print(B.shape)
print(C.shape)

print(C)

# --------------------------------------------------

print("\n" + "=" * 60)
print("7. chunk()")
print("=" * 60)

chunks = torch.chunk(x, chunks=2, dim=-1)

print(chunks[0].shape)
print(chunks[1].shape)

print(chunks[0])
print(chunks[1])

# --------------------------------------------------

print("\n" + "=" * 60)
print("8. cat()")
print("=" * 60)

cat = torch.cat(chunks, dim=-1)

print(cat.shape)
print(cat)

# --------------------------------------------------

print("\n" + "=" * 60)
print("9. stack()")
print("=" * 60)

a = torch.tensor([1,2,3])

b = torch.tensor([4,5,6])

stack = torch.stack([a,b], dim=0)

print(stack)
print(stack.shape)

# --------------------------------------------------

print("\n" + "=" * 60)
print("10. sin() / cos()")
print("=" * 60)

theta = torch.arange(5).float()

print(theta)

print(torch.sin(theta))

print(torch.cos(theta))

# --------------------------------------------------

print("\n" + "=" * 60)
print("11. arange()")
print("=" * 60)

print(torch.arange(10))

print(torch.arange(0,10,2))

# --------------------------------------------------

print("\n" + "=" * 60)
print("12. einsum()")
print("=" * 60)

Q = torch.randn(2,4,8)

K = torch.randn(2,4,8)

scores = torch.einsum("bqd,bkd->bqk",Q,K)

print(scores.shape)

print(scores)