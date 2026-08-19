import torch
import torch.nn as nn
from torch.optim import AdamW

from src.DL_pytorch.model.transformer import Transformer
from src.dataset.dataset import Dataset
from src.dataset.dataloader import DataLoader
from src.utils.config import Config
from src.utils.checkpoint.checkpoint_torch import (
    save_checkpoint,
    load_checkpoint,
)


def main():

    cfg = Config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    dataset = Dataset(
        cfg.token_file,
        cfg.seq_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = Transformer(
        cfg.vocab_size,
        cfg.d_model,
        cfg.num_heads,
        cfg.num_layers,
    ).to(device)

    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
    )



    start_epoch = load_checkpoint(
        model,
        optimizer,
        cfg.checkpoint_pytorch,
        device,
    )

    torch.save(
        model.state_dict(),
        "src/DL_pytorch/checkpoint/initial_model.pt"
    )

    for epoch in range(start_epoch, cfg.epochs):

        epoch_loss = 0.0

        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()

            logits = model(input_ids)

            logits = logits.view(-1, cfg.vocab_size)
            target_ids = target_ids.view(-1)

            loss = criterion(logits, target_ids)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:

                print(
                    f"Epoch {epoch+1}/{cfg.epochs} | "
                    f"Step {batch_idx+1}/{len(dataloader)} | "
                    f"Loss {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(dataloader)

        print(
            f"Epoch {epoch+1} Finished | "
            f"Average Loss: {avg_loss:.4f}"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=avg_loss,
            checkpoint_path=cfg.checkpoint_pytorch,
        )


if __name__ == "__main__":
    main()