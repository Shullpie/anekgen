import json
import logging
from typing import Literal

import torch
from torch.utils.data import Dataset

from src.data.tokenizer import Tokenizer
from config import config

logger = logging.getLogger(__name__)

class AnekDataset(Dataset):
    def __init__(self, dataset_type: Literal['train'] | Literal['test']) -> None:
        logger.info('AnekDataset(%s) initialization started.', dataset_type)
        self.tokenizer = Tokenizer()
        
        if config['mode'] == 'train':
            with open(config['data']['aneks_path'][dataset_type], 'r') as f:
                self.aneks = json.load(f)
            logger.info('%s loaded to self.aneks.', dataset_type)
        logger.info('AnekDataset(%s) initialization completed.', dataset_type)

    def __getitem__(self, idx: int) -> torch.Tensor:
        anek = self.aneks[str(idx)]
        return self.tokenizer.encode(anek)

    def __len__(self) -> int:
        return len(self.aneks)
