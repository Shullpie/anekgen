import logging

import torch
import torch.nn as nn

from config import config
from src.archs.utils.blocks import get_block

logger = logging.getLogger(__name__)

class RNN(nn.Module):
    def __init__(self, 
                 block,
                 embedding_size, 
                 hidden_size,
                 num_layers,
                 block_dropout: float,
                 pad_idx: int, 
                 vocab_size: int = 0
                 ):
        super().__init__()
        self.block = block
        self.block_dropout = block_dropout
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.block = get_block(embedding_size=self.embedding_size,
                               block_name=self.block,
                               num_layers=self.num_layers,
                               block_dropout=self.block_dropout,
                               hidden_size=self.hidden_size,
                               bidirectional=False)
        logger.info('RNN %s initialization started.', self.block._get_name())
        
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.embeddings = nn.Embedding(
            self.vocab_size, 
            embedding_dim=self.embedding_size,
            padding_idx=self.pad_idx,
        )

        self.output  = nn.Linear(self.hidden_size, self.vocab_size)
        logger.info('RNN %s initialization completed.', self.block._get_name())

    def forward(self, **kwargs) -> torch.Tensor:
        x = kwargs['input_ids']
        x = self.embeddings(x)
        x, _ = self.block(x)
        return self.output(x)

