import torch
import logging
from transformers import PreTrainedTokenizerFast, AutoTokenizer

logger = logging.getLogger(__name__)

class Tokenizer():
    def __init__(self, path: str):
        logger.info('Tokenizer initialization started.')
        self.tokenizer = self._get_tokenizer(path)
        self.pad_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.eos_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.vocab_size = self.tokenizer.vocab_size
        logger.info('Tokenizer initialization completed.')


    def __len__(self):
        return len(self.tokenizer)

    def _add_special_tokens(self, texts: list) -> list[str]:
        if not isinstance(texts, str):
            return [
                self.tokenizer.bos_token + text + self.tokenizer.eos_token 
                for text in texts
            ]
        return self.tokenizer.bos_token + texts + self.tokenizer.eos_token

    def __call__(self, texts, *args, **kwargs) -> torch.Tensor:
        tokens = self._add_special_tokens(texts)
        return self.tokenizer(tokens, *args, **kwargs)
    
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @staticmethod
    def _get_tokenizer(path) -> PreTrainedTokenizerFast | AutoTokenizer:
       if path.split('/')[-1] == 'gpt_lora':
           return AutoTokenizer.from_pretrained(path)
       
       return  PreTrainedTokenizerFast(
            tokenizer_file=path,
            unk_token='[UNK]',
            pad_token='[PAD]',
            cls_token='[CLS]',
            sep_token='[SEP]',
            mask_token='[MASK]',
            bos_token='[BOS]',
            eos_token='[EOS]',
            max_len=256,
            return_token_type_ids=False
        )
    
    

