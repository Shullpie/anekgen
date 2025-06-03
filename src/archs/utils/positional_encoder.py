import math
import torch
import torch.nn as nn

class SinosuidalPositionalEncoder(nn.Module):
    def __init__(self, dropout_p: float, max_length: int = 256, embedding_dim: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.pe = torch.zeros(max_length, embedding_dim)

        positions = torch.arange(0, max_length).unsqueeze(1)
        div_term = (torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float)) 
                    * (-math.log(10_000)/embedding_dim)).unsqueeze(0)
        
        arguments = positions*div_term # seq_dim x emb_dim
        self.pe[:,0::2] = torch.sin(arguments)
        self.pe[:,1::2] = torch.cos(arguments)
        self.pe = nn.Parameter(self.pe.unsqueeze(0), requires_grad=False) # 1 x seq_dim x emb_dim

    def forward(self, x: torch.Tensor):
        # x: [batch_size, seq_len, embedding_dim]
        return x + self.dropout(self.pe[:, :x.shape[1]])
    
    
class LearnedPositionalEncoder(nn.Module):
    def __init__(self, dropout_p: float, max_length: int = 256, embedding_dim: int = 256):
        super().__init__()
        self.pe = nn.Embedding(max_length, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.dropout(self.pe(positions))
