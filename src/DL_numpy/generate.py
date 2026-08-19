import numpy as np

from src.utils.checkpoint.checkpoint_np import Checkpoint
from src.utils.config import Config
from src.DL_numpy.model.transformer import Transformer
from src.tokenizer.decode import Decoder
from src.tokenizer.encode import Encoder

cfg = Config()

encoder = Encoder(cfg.vocab_file, cfg.merges_file)
decoder = Decoder(cfg.vocab_file)


def softmax(x):
    # Numerically stable softmax along last axis
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# 1. Initialize model and optimizer
model = Transformer(
    vocab_size=cfg.vocab_size,
    seq_length=cfg.seq_length,
    d_model=cfg.d_model,
    heads=cfg.num_heads,
    hidden_dim=cfg.hidden_dim,
    num_layers=cfg.num_layers,
)

# Pass model to load weights (optimizer is optional during inference)
Checkpoint.load(model, None, "src/DL_numpy/checkpoint/latest.npz")

prompt = "Kernels"
tokens = encoder.encode(prompt)

print("=" * 70)
print("Prompt:", prompt)
print("=" * 70)

for step in range(50):
    # Truncate context to max sequence length if needed
    context = tokens[-cfg.seq_length:]

    # Prepare input batch (1, len(context))
    x = np.array(context).reshape(1, -1)

    # Forward pass
    logits = model.forward(x)

    # Get logits for the LAST token position in current sequence
    next_logits = logits[0, -1]

    # Calculate probabilities
    probs = softmax(next_logits)

    # Top 10 prediction indices
    top_ids = np.argsort(probs)[-10:][::-1]

    print(f"\n===== STEP {step+1} =====")
    print("Current Text:")
    print(decoder.decode(tokens))
    print()

    print("Top Predictions:")
    for idx in top_ids:
        try:
            token_str = decoder.decode([int(idx)])
        except Exception:
            token_str = "<raw byte>"

        print(f"{idx:5d} | {repr(token_str):<20} | {probs[idx]:.6f}")

    # Greedy choice (highest probability)
    next_token = int(top_ids[0])

    # End-of-sequence token check (e.g., </s> or <eos>)
    if next_token == 370:
        print("\n✅ End token predicted.")
        break

    tokens.append(next_token)

print("\n" + "=" * 70)
print("Final Output:")
print("=" * 70)
print(decoder.decode(tokens))
