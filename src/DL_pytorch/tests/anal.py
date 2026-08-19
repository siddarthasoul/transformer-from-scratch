import torch

checkpoint = torch.load(
    "src/DL_pytorch/checkpoint/model.pt",
    map_location="cpu",
)

# if you saved a full checkpoint
state_dict = checkpoint["model_state_dict"]

for name, param in state_dict.items():
    print(
        name,
        torch.isnan(param).any().item(),
        torch.isinf(param).any().item(),
    )

for name, param in state_dict.items():
    print(
        f"{name:35}",
        param.shape,
        param.dtype,
    )


for name, param in state_dict.items():
    print(
        f"{name:35}",
        f"mean={param.mean().item():8.5f}",
        f"std={param.std().item():8.5f}",
        f"min={param.min().item():8.5f}",
        f"max={param.max().item():8.5f}",
    )    