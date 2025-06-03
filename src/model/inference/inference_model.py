import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import config
from src.data.tokenizer import Tokenizer
from src.data.nucleus_sampler import NucleusSampler
from src.archs import get_nn_model

class InferenceModel:
    def __init__(self, selected_model: str, config: dict):
        self.selected_model = selected_model
        self.lora_path = config['archs'][selected_model]['lora_path'] \
                         if self.selected_model == 'gpt_lora' \
                         else None
        self.max_len = config['max_len']
        self.tokenizer = Tokenizer(config['archs'][self.selected_model]['tokenizer_path'])
        self.sampler = NucleusSampler(**config['archs'][self.selected_model]['sampler']['nucleus'])
        self.model = self.load_model(path=config['archs'][self.selected_model]['model_path'],
                                     config=config)


    def load_model(self, path: str, config: dict):
        model = get_nn_model(model_name=self.selected_model,
                             config=config, 
                             pad_idx=self.tokenizer.pad_idx, 
                             vocab_size=self.tokenizer.vocab_size,
                             load_checkpoint_path=self.lora_path, 
                             is_trainable=False)
        
        if self.selected_model == 'gpt_lora':
            path = path + '/state.pt'
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def next_token(self, x: dict):
        logits = self.model(**x)
        if isinstance(logits, CausalLMOutputWithCrossAttentions):
            logits = logits.logits
        logits = logits.squeeze(0)[-1]
        out = self.sampler(logits)
        return out

    @torch.inference_mode()
    def generate(self, prefix=''):
        out = None
        token_count = 0

        x = self.tokenizer(prefix, 
                           padding=True, 
                           return_tensors='pt', 
                           truncation=True, 
                           return_token_type_ids=False)#[:-1]
        x = self.prepare_tokens(x)
        print(x)

        while out != self.tokenizer.eos_idx and token_count < self.max_len:
            out  = self.next_token(x).unsqueeze(0)
            x['input_ids'] = torch.cat((x['input_ids'], out), dim=-1)
            x['attention_mask'] = torch.cat((x['attention_mask'], torch.ones(1, 1)), dim=-1)
            token_count += 1

        x = self.tokenizer.decode(x['input_ids'][0], 
                                  skip_special_tokens=True, 
                                  clean_up_tokenization_spaces=True
                                  ).replace(' ##', '')
        return x

    @staticmethod
    def prepare_tokens(tokens: dict) -> dict:
        for key, value in tokens.items():
            tokens[key] = value[:, :-1]
        return tokens

