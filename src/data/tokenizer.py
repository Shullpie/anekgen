import torch
import logging
from transformers import PreTrainedTokenizerFast
from config import config

logger = logging.getLogger(__name__)

class Tokenizer():
    def __init__(self):
        logger.info('Tokenizer initialization started.')
        self.tokenizer = self._get_tokenizer()
        logger.info('Tokenizer initialization completed.')
        self.pad_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.eos_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.vocab_size = self.tokenizer.vocab_size


    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(
            self.tokenizer.bos_token + text + self.tokenizer.eos_token
        )

    def encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenize(text)
        return torch.LongTensor(self.tokenizer.convert_tokens_to_ids(tokens))
    
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @staticmethod
    def _get_tokenizer() -> PreTrainedTokenizerFast:
       return  PreTrainedTokenizerFast(
            tokenizer_file=config['data']['tokenizer_path'],
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            bos_token='[BOS]',
            eos_token='[EOS]'
        )
    
    

