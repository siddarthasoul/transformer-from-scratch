from dataclasses import dataclass


@dataclass
class Config:

    # -----------------------------
    # Dataset
    # -----------------------------
    train_file = "src/data/raw/train.txt"
    vocab_file = "src/data/vocab/vocab.json"
    merges_file = "src/data/vocab/merges.json"
    token_file = "src/data/processed/token_ids.npy"

    # -----------------------------
    # Model
    # -----------------------------
    vocab_size = 1003
    seq_length = 10

    d_model = 32
    hidden_dim = 64

    num_heads = 4
    num_layers = 2

    # -----------------------------
    # Training
    # -----------------------------
    batch_size = 4
    epochs = 11

    learning_rate = 1e-3

    # -----------------------------
    # Checkpoints
    # -----------------------------
    save_every = 1

    checkpoint_dir = "src/DL_numpy/checkpoint"
    checkpoint_pytorch = "src/DL_pytorch/checkpoint/model.pt"

    # -----------------------------
    # Debug
    # -----------------------------
    print_every = 10

    debug = True

    random_seed = 42