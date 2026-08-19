import os
import numpy as np


class Checkpoint:

    @staticmethod
    def save(model, optimizer, epoch, loss, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "t": getattr(optimizer, "t", 0) if optimizer else 0,
        }

        # Save model weights and optimizer states
        for i, p in enumerate(model.parameters()):
            checkpoint[f"param_{i}"] = p.data

            if optimizer is not None:
                if hasattr(optimizer, "m") and i < len(optimizer.m):
                    checkpoint[f"m_{i}"] = optimizer.m[i]

                if hasattr(optimizer, "v") and i < len(optimizer.v):
                    checkpoint[f"v_{i}"] = optimizer.v[i]

        np.savez(path, **checkpoint)
        print(f"\n✅ Checkpoint saved -> {path}")

    @staticmethod
    def load(model, optimizer, path):
        if not os.path.exists(path):
            print("\n⚠️ No checkpoint found. Starting from scratch.\n")
            return 0, None

        checkpoint = np.load(path, allow_pickle=True)

        # Restore model parameters
        for i, p in enumerate(model.parameters()):
            if f"param_{i}" in checkpoint:
                p.data[...] = checkpoint[f"param_{i}"]

        # Restore optimizer states only if an optimizer is provided
        if optimizer is not None:
            optimizer.m = []
            optimizer.v = []

            for i, p in enumerate(model.parameters()):
                if f"m_{i}" in checkpoint:
                    optimizer.m.append(checkpoint[f"m_{i}"])

                if f"v_{i}" in checkpoint:
                    optimizer.v.append(checkpoint[f"v_{i}"])

            if "t" in checkpoint:
                optimizer.t = int(checkpoint["t"])

        epoch = int(checkpoint["epoch"]) if "epoch" in checkpoint else 0
        loss = float(checkpoint["loss"]) if "loss" in checkpoint else 0.0

        print(f"\n✅ Loaded checkpoint: {path}")
        print(f"Epoch : {epoch}")
        print(f"Loss  : {loss:.6f}")

        return epoch, loss