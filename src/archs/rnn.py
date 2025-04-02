import logging

import torch
import torch.nn as nn

from config import config
from src.archs.optim.blocks import get_block

logger = logging.getLogger(__name__)

class RNN(nn.Module):
    def __init__(self, padding_idx: int, vocab_size: int = 0):
        super().__init__()
        self.embedding_size = config['archs']['embedding_size']
        self.hidden_size = config['archs']['RNN']['hidden_size']
        self.block = get_block(self.embedding_size)
        logger.info('RNN %s initialization started.', self.block._get_name())
        
        self.batch_size = config['dataloader']['batch_size']
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.batch_size = config['dataloader']['batch_size']
        self.embedding_size = config['archs']['embedding_size']
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.embeddings = nn.Embedding(
            self.vocab_size, 
            embedding_dim=config['archs']['embedding_size'],
            padding_idx=self.padding_idx,
        )

        self.output = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout1d(float(config['archs']['RNN']['output_dropout'])),
            nn.Linear(in_features=256, out_features=self.vocab_size)
        )
        logger.info('RNN %s initialization completed.', self.block._get_name())

    # TODO output alias
    def forward(self, x: torch.Tensor):
        x = self.embeddings(x)
        x, _ = self.block(x)
        return self.output(x)

