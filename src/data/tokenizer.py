import logging
from transformers import AutoTokenizer
from config import config

logger = logging.getLogger(__name__)

class Tokenizer():
    def __init__(self):
        logger.info('Tokenizer initialization started.')
        self.tokenizer = self._get_tokenizer()
        logger.info('Tokenizer initialization completed.')

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)


    def tokenize_as_ids(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    @staticmethod
    def _get_tokenizer() -> AutoTokenizer:
       return AutoTokenizer.from_pretrained(config['data']['vocab_path'])
