import torch

from src.utils.config import Config
from src.DL_pytorch.model.transformer import Transformer
from src.utils.checkpoint.checkpoint_torch import load_checkpoint

from src.tokenizer.encode import Encoder
from src.tokenizer.decode import Decoder


cfg = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(cfg.vocab_file, cfg.merges_file)
decoder = Decoder(cfg.vocab_file)

model = Transformer(
    cfg.vocab_size,
    cfg.d_model,
    cfg.num_heads,
    cfg.num_layers,
).to(device)

optimizer = torch.optim.AdamW(model.parameters())

load_checkpoint(
    model,
    optimizer,
    cfg.checkpoint_pytorch,
    device,
)

model.eval()


prompt = "Kernels"

tokens = encoder.encode(prompt)

tokens = torch.tensor(
    [tokens],
    dtype=torch.long,
    device=device,
)

with torch.no_grad():

    for _ in range(50):

        logits = model(tokens)

        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        if next_token.item() == 370:
            break

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

generated = tokens[0].tolist()

print(decoder.decode(generated))