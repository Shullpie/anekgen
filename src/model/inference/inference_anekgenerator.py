from typing import Literal
import torch
from src.model.inference.inference_model import InferenceModel

class AnekGenerator:
    def __init__(self, config: dict):
        rnn = InferenceModel('rnn', config=config)
        gpt = InferenceModel('gpt', config=config)
        gpt_lora = InferenceModel('gpt_lora', config=config)
        self.models = {'rnn': rnn, 'gpt': gpt, 'gpt_lora': gpt_lora}

    @torch.inference_mode()
    def generate(self, 
                 model: Literal['rnn'] | Literal['gpt'] | Literal['gpt_lora'], 
                 prefix: str
                 ) -> str:
        return self.models[model.lower()].generate(prefix)
    
