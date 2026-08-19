import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    checkpoint_path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved -> {checkpoint_path}")


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device,
):
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting from scratch.")
        return 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    print(
        f"Checkpoint loaded from {checkpoint_path}"
    )
    print(
        f"Resuming from epoch {start_epoch}"
    )

    return start_epoch