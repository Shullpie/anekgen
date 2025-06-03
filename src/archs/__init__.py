from transformers import AutoModelForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig

def get_nn_model(
                 model_name: str,
                 config: dict, 
                 pad_idx: int, 
                 vocab_size: int, 
                 load_checkpoint_path: str = None,
                 is_trainable: bool = True):
    model_name = model_name.lower()
    arch_params = config['archs'][model_name]
    m = None
    if model_name == 'rnn':
        from .rnn import RNN
        m = RNN(block=arch_params['block'],
                block_dropout=arch_params['block_dropout'],
                embedding_size=arch_params['embedding_size'],
                hidden_size=arch_params['hidden_size'],
                num_layers=arch_params['num_layers'],
                pad_idx=pad_idx,
                vocab_size=vocab_size)
        
    elif model_name == 'gpt':
        from .gpt import GPT
        m = GPT(n_layers=arch_params['n_layers'],
                positional_encoder_type=arch_params['positional_encoder']['type'],
                positional_encoder_dropout_p=arch_params['positional_encoder']['dropout'],
                max_seq_len=arch_params['positional_encoder']['max_length'],
                attention_dropout=float(arch_params['multihead']['attention']['dropout']),
                attention_size=arch_params['multihead']['attention']['size'],
                mh_dropout=float(arch_params['multihead']['dropout']),
                n_heads=arch_params['multihead']['n_heads'],
                ff_dropout=float(arch_params['feadforward']['dropout']), 
                embedding_dim=arch_params['embedding_size'],
                pad_idx=pad_idx,
                vocab_size=vocab_size)
    elif model_name == 'gpt_lora':
        try:
            m = AutoModelForCausalLM.from_pretrained(arch_params['model_path'])
        except OSError:
            m = AutoModelForCausalLM.from_pretrained(arch_params['model_path_hf'])

        if load_checkpoint_path:
            m = PeftModel.from_pretrained(m, load_checkpoint_path, is_trainable=is_trainable)
        else:
            lora_config = LoraConfig(**arch_params['lora_config'])
            m = get_peft_model(m, lora_config)
        m.lm_head.weight.requires_grad = True

    else:
        raise NotImplementedError(f'Architecture {model_name} is not recognized.')
    
    
    return m
