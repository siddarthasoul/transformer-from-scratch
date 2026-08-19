import torch


print("=" * 70)
print("PYTORCH TENSOR BASICS")
print("=" * 70)


# ================================================================
# 1. Create Tensor
# ================================================================

print("\n" + "=" * 70)
print("1. Create Tensor")
print("=" * 70)

a = torch.tensor([1, 2, 3])

print("Tensor:")
print(a)


# ================================================================
# 2. Data Type
# ================================================================

print("\n" + "=" * 70)
print("2. Data Type")
print("=" * 70)

x = torch.tensor([1, 2, 3], dtype=torch.float32)

print("Tensor:")
print(x)

print("dtype:", x.dtype)


# ================================================================
# 3. Shape Information
# ================================================================

print("\n" + "=" * 70)
print("3. Shape Information")
print("=" * 70)

print("Shape:", x.shape)
print("Dimensions:", x.ndim)
print("Number of Elements:", x.numel())


# ================================================================
# 4. Creating Different Tensors
# ================================================================

print("\n" + "=" * 70)
print("4. Creating Different Tensors")
print("=" * 70)

print("\nZeros:")
print(torch.zeros(2, 3))

print("\nOnes:")
print(torch.ones(2, 3))

print("\nRandom Normal:")
print(torch.randn(2, 3))

print("\nRandom Uniform:")
print(torch.rand(2, 3))

print("\nArange:")
print(torch.arange(10))


# ================================================================
# 5. Reshape
# ================================================================

print("\n" + "=" * 70)
print("5. Reshape")
print("=" * 70)

y = torch.arange(12)

print("\nOriginal:")
print(y)

print("\nReshape (3 x 4):")
print(y.reshape(3, 4))

print("\nView (2 x 6):")
print(y.view(2, 6))


# ================================================================
# 6. Transpose
# ================================================================

print("\n" + "=" * 70)
print("6. Transpose")
print("=" * 70)

matrix = torch.arange(12).reshape(3, 4)

print("\nOriginal:")
print(matrix)

print("\nTranspose:")
print(matrix.transpose(0, 1))


# ================================================================
# 7. Permute
# ================================================================

print("\n" + "=" * 70)
print("7. Permute")
print("=" * 70)

tensor_3d = torch.randn(2, 3, 4)

print("Original Shape:", tensor_3d.shape)

permuted = tensor_3d.permute(2, 1, 0)

print("Permuted Shape:", permuted.shape)


# ================================================================
# 8. Unsqueeze
# ================================================================

print("\n" + "=" * 70)
print("8. Unsqueeze")
print("=" * 70)

print("Original Shape:", x.shape)

x1 = x.unsqueeze(0)

print("Unsqueeze(0):", x1.shape)

x2 = x.unsqueeze(1)

print("Unsqueeze(1):", x2.shape)


# ================================================================
# 9. Squeeze
# ================================================================

print("\n" + "=" * 70)
print("9. Squeeze")
print("=" * 70)

tensor = torch.randn(1, 3, 1, 4)

print("Original Shape:", tensor.shape)

squeezed = tensor.squeeze()

print("After Squeeze:", squeezed.shape)


# ================================================================
# 10. Indexing and Slicing
# ================================================================

print("\n" + "=" * 70)
print("10. Indexing and Slicing")
print("=" * 70)

z = torch.arange(20).reshape(4, 5)

print("\nMatrix:")
print(z)

print("\nFirst Row:")
print(z[0])

print("\nFirst Column:")
print(z[:, 0])

print("\nColumns 1 Onward:")
print(z[:, 1:])

print("\nLast Row:")
print(z[-1])


# ================================================================
# 11. Mathematical Operations
# ================================================================

print("\n" + "=" * 70)
print("11. Mathematical Operations")
print("=" * 70)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("A:", a)
print("B:", b)

print("\nAddition:")
print(a + b)

print("\nSubtraction:")
print(a - b)

print("\nElement-wise Multiplication:")
print(a * b)

print("\nDivision:")
print(a / b)


print("\nMatrix Multiplication:")

A = torch.randn(2, 3)
B = torch.randn(3, 4)

print(A @ B)


# ================================================================
# 12. Reduction Operations
# ================================================================

print("\n" + "=" * 70)
print("12. Reduction Operations")
print("=" * 70)

r = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)

print("Tensor:")
print(r)

print("\nSum:")
print(torch.sum(r))

print("\nMean:")
print(torch.mean(r))

print("\nStandard Deviation:")
print(torch.std(r))

print("\nMaximum:")
print(torch.max(r))

print("\nMinimum:")
print(torch.min(r))

print("\nArgmax:")
print(torch.argmax(r))

print("\nArgmin:")
print(torch.argmin(r))

print("\nProduct:")
print(torch.prod(r))


# ================================================================
# 13. Reduction Along Dimensions
# ================================================================

print("\n" + "=" * 70)
print("13. Reduction Along Dimensions")
print("=" * 70)

print("Tensor:")
print(r)

print("\nSum along dim=0:")
print(torch.sum(r, dim=0))

print("\nSum along dim=1:")
print(torch.sum(r, dim=1))

print("\nKeep dimension:")
print(torch.sum(r, dim=1, keepdim=True))


# ================================================================
# 14. Broadcasting
# ================================================================

print("\n" + "=" * 70)
print("14. Broadcasting")
print("=" * 70)

matrix = torch.arange(12).reshape(3, 4)

vector = torch.tensor([1, 2, 3, 4])

print("Matrix:")
print(matrix)

print("\nVector:")
print(vector)

print("\nMatrix + Vector:")
print(matrix + vector)


# ================================================================
# 15. Boolean Masking
# ================================================================

print("\n" + "=" * 70)
print("15. Boolean Masking")
print("=" * 70)

values = torch.arange(1, 11)

print("Values:")
print(values)

mask = values > 5

print("\nMask:")
print(mask)

print("\nValues greater than 5:")
print(values[mask])


# ================================================================
# 16. Where
# ================================================================

print("\n" + "=" * 70)
print("16. torch.where")
print("=" * 70)

values = torch.arange(1, 11)

result = torch.where(values > 5, 100, 0)

print("Original:")
print(values)

print("\nWhere values > 5:")
print(result)


# ================================================================
# 17. Clamp
# ================================================================

print("\n" + "=" * 70)
print("17. Clamp")
print("=" * 70)

values = torch.tensor([1.0, 3.0, 5.0, 8.0, 10.0])

print("Original:")
print(values)

print("\nClamped between 3 and 7:")
print(torch.clamp(values, 3, 7))


# ================================================================
# 18. GPU / Device
# ================================================================

print("\n" + "=" * 70)
print("18. GPU / Device")
print("=" * 70)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

device_tensor = torch.randn(2, 2).to(device)

print("\nTensor:")
print(device_tensor)

print("\nTensor Device:")
print(device_tensor.device)


# ================================================================
# Finished
# ================================================================

print("\n" + "=" * 70)
print("Finished PyTorch Tensor Basics")
print("=" * 70)