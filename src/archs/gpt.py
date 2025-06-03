import math
import logging

import torch
from typing import Literal
import torch.nn as nn
from torch.nn.functional import softmax

from src.archs.utils import positional_encoder
from config import config

logger = logging.getLogger(__name__)

class Atention(nn.Module):
    def __init__(self, embedding_dim: int, attention_size: int, attention_dropout: float):
        super().__init__()
        self.wq = nn.Linear(embedding_dim, attention_size, bias=False)
        self.wk = nn.Linear(embedding_dim, attention_size, bias=False)
        self.wv = nn.Linear(embedding_dim, attention_size, bias=False)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: torch.Tensor = None
                ) -> torch.Tensor:
        # x: [batch_size, seq_len, embedding_dim]
        q = self.wq(q) # [batch_size, seq_len, attention_size]
        k = self.wk(k) # [batch_size, seq_len, attention_size]
        v = self.wv(v) # [batch_size, seq_len, attention_size]

        prod = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            prod = prod.masked_fill_(mask, -math.inf)
        prod = prod / math.sqrt(k.shape[-1]) # [batch_size, seq_len, seq_len]
        attention = torch.bmm(self.dropout(softmax(prod, dim=-1)),  v) # [batch_size, seq_len, attention_size]
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 attention_size: int, 
                 attention_dropout: float, 
                 mh_dropout: float,
                 n_heads: int,
                 ):
        super().__init__()
        self.heads = nn.ModuleList([Atention(embedding_dim, attention_size//n_heads, attention_dropout)
                                    for _ in range(n_heads)])
        self.linear = nn.Linear(attention_size, embedding_dim, bias=False)
        self.dropout = nn.Dropout(mh_dropout)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: torch.Tensor = None
                ) -> torch.Tensor:
        output = []
        for head in self.heads:
            output.append(head(q, k, v, mask))
        output = torch.cat(output, dim=-1)
        return self.dropout(self.linear(output))

class FeedForward(nn.Module):
    def __init__(self, 
                 embedding_dim: int,
                 dropout_p: float
                 ):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, embedding_dim*4)
        self.dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(embedding_dim*4, embedding_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.linear1(x)))
        return self.linear2(x)


class GPTDecoderLayer(nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 attention_size: int,
                 attention_dropout: float,
                 mh_dropout: float,
                 ff_dropout: float,
                 n_heads: int, 
                ):
        super().__init__()
        self.self_attn = MultiHeadAttention(embedding_dim=embedding_dim, 
                                            attention_size=attention_size, 
                                            attention_dropout=attention_dropout, 
                                            mh_dropout=mh_dropout, 
                                            n_heads=n_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim=embedding_dim, dropout_p=ff_dropout)
        
    
    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        x = x + self.self_attn(norm_x, norm_x, norm_x, mask=mask)
        x = x + self.ff(self.norm2(x)) # [b s e]
        return x
        

class GPT(nn.Module):
    def __init__(self,
                 n_layers: int,
                 positional_encoder_type: Literal['sinosuidal'] | Literal['learned'],
                 positional_encoder_dropout_p: float,
                 max_seq_len: int,
                 vocab_size: int,
                 pad_idx: int,
                 embedding_dim: int, 
                 attention_size: int,
                 attention_dropout: float,
                 mh_dropout: float,
                 ff_dropout: float,
                 n_heads: int, 
                 ):
        super().__init__()

        logger.info('GPT initialization started')
        self.n_layers = n_layers
        self.positional_encoder_type = positional_encoder_type
        self.positional_encoder_dropout_p = positional_encoder_dropout_p
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.attention_size = attention_size
        self.attention_dropout = attention_dropout
        self.mh_dropput = mh_dropout
        self.ff_dropout = ff_dropout
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        logger.info('Multihead attention with %s heads', self.n_heads)

        self.positional_encoder = None
        if positional_encoder_type == 'learned':
            self.positional_encoder = positional_encoder.LearnedPositionalEncoder(
                dropout_p=positional_encoder_dropout_p,
                max_length=self.max_seq_len,
                embedding_dim=self.embedding_dim
            )
        elif positional_encoder_type == 'sinosuidal':
            self.positional_encoder = positional_encoder.SinosuidalPositionalEncoder(
                dropout_p=positional_encoder_dropout_p,
                max_length=self.max_seq_len,
                embedding_dim=self.embedding_dim
            )
        else:
            raise TypeError('missing positional encoder')
        
        logger.info('Positional Encoder: %s', self.positional_encoder_type)
        

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, self.pad_idx)
        self.layers = nn.ModuleList([GPTDecoderLayer(embedding_dim=self.embedding_dim,
                                                     attention_size=self.attention_size,
                                                     attention_dropout=self.attention_dropout,
                                                     mh_dropout=self.mh_dropput,
                                                     ff_dropout = self.ff_dropout,
                                                     n_heads=self.n_heads)
                                                for _ in range(self.n_layers)])

        self.norm = nn.LayerNorm(self.embedding_dim)
        self.liner = nn.Linear(self.embedding_dim, self.vocab_size)
        logger.info('GPT initialization completed')
    
    @staticmethod
    def get_mask(padding_mask):
        seq_len = padding_mask.size(1)
        padding_mask = (padding_mask == 0)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
        look_ahead = torch.triu(torch.ones(seq_len, seq_len, device=padding_mask.device), diagonal=1).bool()
        return padding_mask | look_ahead
        
    def forward(self, **kwargs):
        x = kwargs['input_ids']
        mask = kwargs['attention_mask']
        mask = self.get_mask(mask)
        x = self.embeddings(x)
        x = self.positional_encoder(x)

        for layer in self.layers:
            x = layer(x, mask) # [b s e]

        return self.liner(self.norm(x)) # [b s v]
    
    
        
        





