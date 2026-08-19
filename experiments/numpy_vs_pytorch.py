import numpy as np
import torch

print("=" * 70)
print("NUMPY vs PYTORCH")
print("=" * 70)

# -------------------------------------------------
# Create Data
# -------------------------------------------------

np_x = np.random.randint(1, 10, (3, 4))
torch_x = torch.tensor(np_x)

print("\nOriginal Matrix")
print("----------------------")

print("\nNumPy")
print(np_x)

print("\nPyTorch")
print(torch_x)

# -------------------------------------------------
# Shape
# -------------------------------------------------

print("\nShape")
print("----------------------")

print(np_x.shape)
print(torch_x.shape)

# -------------------------------------------------
# Sum
# -------------------------------------------------

print("\nSum")
print("----------------------")

print(np.sum(np_x))
print(torch.sum(torch_x))

# -------------------------------------------------
# Mean
# -------------------------------------------------

print("\nMean")
print("----------------------")

print(np.mean(np_x))
print(torch.mean(torch_x.float()))

# -------------------------------------------------
# Max
# -------------------------------------------------

print("\nMax")
print("----------------------")

print(np.max(np_x))
print(torch.max(torch_x))

# -------------------------------------------------
# Min
# -------------------------------------------------

print("\nMin")
print("----------------------")

print(np.min(np_x))
print(torch.min(torch_x))

# -------------------------------------------------
# Std
# -------------------------------------------------

print("\nStd")
print("----------------------")

print(np.std(np_x))
print(torch.std(torch_x.float()))

# -------------------------------------------------
# Axis
# -------------------------------------------------

print("\nAxis / Dim")
print("----------------------")

print(np.sum(np_x, axis=0))
print(torch.sum(torch_x, dim=0))

print()

print(np.sum(np_x, axis=1))
print(torch.sum(torch_x, dim=1))

# -------------------------------------------------
# keepdims / keepdim
# -------------------------------------------------

print("\nKeep Dimension")
print("----------------------")

print(np.sum(np_x, axis=1, keepdims=True))
print(torch.sum(torch_x, dim=1, keepdim=True))

# -------------------------------------------------
# Broadcasting
# -------------------------------------------------

print("\nBroadcast")
print("----------------------")

np_b = np.array([1,2,3,4])
torch_b = torch.tensor([1,2,3,4])

print(np_x + np_b)
print(torch_x + torch_b)

# -------------------------------------------------
# Mask
# -------------------------------------------------

print("\nMask")
print("----------------------")

print(np_x > 5)
print(torch_x > 5)

# -------------------------------------------------
# Boolean Index
# -------------------------------------------------

print("\nBoolean Index")
print("----------------------")

print(np_x[np_x > 5])
print(torch_x[torch_x > 5])

# -------------------------------------------------
# Where
# -------------------------------------------------

print("\nWhere")
print("----------------------")

print(np.where(np_x > 5, 100, 0))
print(torch.where(torch_x > 5, 100, 0))

# -------------------------------------------------
# Clamp / Clip
# -------------------------------------------------

print("\nClamp")
print("----------------------")

print(np.clip(np_x, 3, 7))
print(torch.clamp(torch_x, 3, 7))

print("\nDone")