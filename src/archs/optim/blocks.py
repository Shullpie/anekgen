import torch.nn as nn

from config import config

def get_block(embedding_size: int) -> nn.RNN | nn.LSTM | nn.GRU:
    block_name = config['archs']['RNN']['block']
    num_layers = config['archs']['RNN']['num_layers']
    block_dropout = config['archs']['RNN']['block_dropout']
    hidden_size = config['archs']['RNN']['hidden_size']

    if block_name == 'LSTM':
        from torch.nn import LSTM as block
    elif block_name == 'GRU':
        from torch.nn import GRU as block
    elif block_name == 'RNN':
        from torch.nn import GRU as block
    else:
        raise NotImplementedError(f'Block {block_name} is not recognized.')
    return block(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=block_dropout,
            bias=True,
            batch_first=True
        )
