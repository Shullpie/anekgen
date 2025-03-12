import json
import logging
from torch.utils.data import Dataset
from src.data.tokenizer import Tokenizer
from config import config

logger = logging.getLogger(__name__)

class AnekDataset(Dataset):
    def __init__(self) -> None:
        logger.info('AnekDataset initialization started.')
        self.tokenizer = Tokenizer()
        
        if config['mode'] == 'train':
            with open(config['data']['aneks_path'], 'r') as f:
                self.aneks = json.load(f)
            logger.info('aneks.json loaded to self.aneks.')
        logger.info('AnekDataset initialization completed.')

    def __getitem__(self, idx: int) -> list[int]:
        anek = self.aneks[str(idx)]
        return self.tokenizer.encode(anek)

    def __len__(self) -> int:
        return len(self.aneks)
