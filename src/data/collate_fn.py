import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(pad_idx: int) -> torch.Tensor:
    def inner(batch: torch.Tensor) -> torch.Tensor:
        return pad_sequence(batch, padding_value=pad_idx, batch_first=True)
    return inner

