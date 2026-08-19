import numpy as np

path = "DL_numpy/checkpoint/latest.npz"

# Load the archive
data = np.load(path, allow_pickle=True)

print("=" * 60)
print("              CHECKPOINT INSPECTOR")
print("=" * 60)

# 1. Print scalar training values
print(f"Epoch Saved : {data['epoch']}")
print(f"Loss Recorded: {data['loss']:.6f}")
print(f"Adam Step (t): {data['t']}")
print("-" * 60)

# 2. Inspect Weights and Optimizer Moments
print(f"{'Key Name':<15} | {'Shape':<15} | {'Mean':<10} | {'Min / Max'}")
print("-" * 60)

for key in data.files:
    if key in ["epoch", "loss", "t"]:
        continue
    
    arr = data[key]
    print(f"{key:<15} | {str(arr.shape):<15} | {arr.mean():<10.4f} | [{arr.min():.4f}, {arr.max():.4f}]")

print("=" * 60)