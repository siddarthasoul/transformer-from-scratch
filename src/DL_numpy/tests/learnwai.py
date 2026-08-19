import matplotlib.pyplot as plt
import numpy as np

# Load checkpoint
checkpoint = np.load("src/checkpoint/latest.npz")

print("=== WEIGHT HEALTH ANALYSIS ===")

for key in checkpoint.keys():
    # Filter for weights (ignoring scalars or pointers)
    w = checkpoint[key]

    print(f"\nLayer: {w}")


