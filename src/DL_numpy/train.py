import os
import numpy as np

from src.utils.checkpoint.checkpoint_np import Checkpoint
from src.utils.config import Config
from src.dataset.dataloader import DataLoader
from src.dataset.dataset import Dataset
from src.DL_numpy.model.transformer import Transformer
from src.DL_numpy.training.optimizer import Adam
from src.DL_numpy.training.loss import Loss

# =====================================
# Debug Helpers
# =====================================


def debug_batch(tokens, targets):
    print("\n================ BATCH ================")
    print("Tokens Shape :", tokens.shape)
    print("Targets Shape:", targets.shape)
    print("Token Min :", np.min(tokens))
    print("Token Max :", np.max(tokens))
    print("=======================================\n")


def debug_forward(logits):
    print("----------- FORWARD -----------")
    print("Logits Shape :", logits.shape)
    print("Min :", logits.min())
    print("Max :", logits.max())
    print("Mean:", logits.mean())
    if np.isnan(logits).any():
        print("❌ NaN detected in logits")
    if np.isinf(logits).any():
        print("❌ Inf detected in logits")
    print("-------------------------------\n")


def debug_loss(loss):
    print("Loss :", loss)
    if np.isnan(loss):
        raise RuntimeError("Loss became NaN")
    if np.isinf(loss):
        raise RuntimeError("Loss became Inf")


def debug_gradients(model):
    print("\n=========== GRADIENTS ===========")
    for p in model.parameters():
        norm = np.linalg.norm(p.grad) if p.grad is not None else 0.0
        print(f"{p.shape}  Grad Norm : {norm:.6f}")
        if p.grad is not None and np.isnan(p.grad).any():
            raise RuntimeError("NaN Gradient")
        if p.grad is not None and np.isinf(p.grad).any():
            raise RuntimeError("Inf Gradient")
    print("=================================\n")


def debug_weights(model):
    print("\n=========== WEIGHTS ===========")
    for p in model.parameters():
        print(
            f"{p.shape} "
            f"Mean={p.data.mean():.6f} "
            f"Std={p.data.std():.6f}"
        )
    print("===============================\n")


def main():
    cfg = Config()

    # -------------------------
    # Random Seed
    # -------------------------
    np.random.seed(cfg.random_seed)

    # -------------------------
    # Dataset & DataLoader
    # -------------------------
    dataset = Dataset(
        cfg.token_file,
        cfg.seq_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # -------------------------
    # Model Setup
    # -------------------------
    model = Transformer(
        vocab_size=cfg.vocab_size,
        seq_length=cfg.seq_length,
        d_model=cfg.d_model,
        heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    )

    # -------------------------
    # Loss & Optimizer
    # -------------------------
    criterion = Loss()
    optimizer = Adam(
        model.parameters(),
        lr=cfg.learning_rate,
    )

    # -------------------------
    # Checkpoint Loading
    # -------------------------
    start_epoch = 0
    latest_ckpt = os.path.join(cfg.checkpoint_dir, "latest.npz")
    if os.path.exists(latest_ckpt):
        start_epoch, last_loss = Checkpoint.load(model, optimizer, latest_ckpt)
        print(f"Resumed from epoch {start_epoch} with loss {last_loss:.6f}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("Training Started")
    print("=" * 60)

    for epoch in range(start_epoch, cfg.epochs):
        epoch_loss = 0.0

        for step, (tokens, targets) in enumerate(dataloader):
            # 1. Zero out previous gradients BEFORE backward
            optimizer.zero_grad()

            # 2. Forward pass
            logits = model.forward(tokens)

            # 3. Compute Loss & Initial Gradient
            loss = criterion.forward(logits, targets)
            grad = criterion.backward()

            # 4. Backward pass through model
            model.backward(grad)

            # 5. Step optimizer weights
            optimizer.step()

            epoch_loss += loss

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} | "
                    f"Step {step}/{len(dataloader)} | "
                    f"Loss: {loss:.6f}"
                )

        avg_loss = epoch_loss / len(dataloader)

        print("=" * 60)
        print(f"Epoch {epoch + 1} Finished")
        print(f"Average Loss : {avg_loss:.6f}")
        print("=" * 60)

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(
            cfg.checkpoint_dir, f"epoch_{epoch + 1}.npz"
        )

        Checkpoint.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=avg_loss,
            path=checkpoint_path,
        )

        Checkpoint.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=avg_loss,
            path=latest_ckpt,
        )


if __name__ == "__main__":
    main()