from transformers import AutoTokenizer
from config import config

class Tokenizer():
    def __init__(self):
        self.tokenizer = self._get_tokenizer()

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    @staticmethod
    def _get_tokenizer() -> AutoTokenizer:
       return AutoTokenizer.from_pretrained(config['data']['vocab_path'])

    def tokenize_as_ids(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(tokens)
