import torch
from torch.nn.functional import softmax

class NucleusSampler:
    def __init__(self, p: float = 0.95, temperature: float = .7):
        self.p = p
        self.temperature = temperature
        print(self.temperature)

    def __call__(self, logits: torch.Tensor):
        logits = logits/self.temperature

        sorted_probs, sorted_indices = softmax(logits, dim=-1).sort(descending=True)

        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = cumsum_probs <= self.p

        if not nucleus_mask.any():
            nucleus_mask[0] = True
        
        nucleus_probs = sorted_probs[nucleus_mask]
        nucleus_indices = sorted_indices[nucleus_mask]
        nucleus_probs /= nucleus_probs.sum()

        choosen_idx = torch.multinomial(nucleus_probs, num_samples=1)
        return nucleus_indices[choosen_idx]
