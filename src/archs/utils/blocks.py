from typing import Literal

import torch.nn as nn

def get_block(embedding_size: int,
              block_name: Literal['lstm'] | Literal['gru'] | Literal['rnn'], 
              num_layers: int,
              block_dropout: float,
              hidden_size: int,
              bidirectional: bool,
              ) -> nn.RNN | nn.LSTM | nn.GRU:
    block_name = block_name.lower()
    if block_name == 'lstm':
        from torch.nn import LSTM as block
    elif block_name == 'gru':
        from torch.nn import GRU as block
    elif block_name == 'rnn':
        from torch.nn import GRU as block
    else:
        raise NotImplementedError(f'Block {block_name} is not recognized.')
    return block(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=block_dropout,
            bias=True,
            batch_first=True
        )
