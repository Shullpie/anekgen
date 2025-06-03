import torch

def collate_fn(tokenizer, *args, **kwargs) -> torch.Tensor:
    def inner(batch: torch.Tensor) -> torch.Tensor:
        return tokenizer(batch, *args, **kwargs)
    return inner

