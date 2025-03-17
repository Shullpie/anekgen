def get_nn_model(model_name: str, pad_idx: int, vocab_size: int):
    if model_name.lower() == 'rnn':
        from .rnn import RNN as model
    else:
        raise NotImplementedError(f'Architecture {model_name} is not recognized.')
    
    m = model(padding_idx=pad_idx, vocab_size=vocab_size)

    return m
