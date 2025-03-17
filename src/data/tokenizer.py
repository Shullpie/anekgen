import torch
import logging
from transformers import AutoTokenizer
from config import config

logger = logging.getLogger(__name__)

class Tokenizer():
    def __init__(self):
        logger.info('Tokenizer initialization started.')
        self.tokenizer = self._get_tokenizer()
        logger.info('Tokenizer initialization completed.')

    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(
            self.tokenizer.bos_token + text + self.tokenizer.eos_token
        )

    def encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenize(text)
        return torch.LongTensor(self.tokenizer.convert_tokens_to_ids(tokens))

    def get_pad_idx(self):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
    

    @staticmethod
    def _get_tokenizer() -> AutoTokenizer:
       return AutoTokenizer.from_pretrained(config['data']['vocab_path'])
    

