import json
import logging
from typing import Literal

import torch
from torch.utils.data import Dataset

from src.data.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

class AnekDataset(Dataset):
    def __init__(self, 
                 dataset_type: Literal['train'] | Literal['test'],
                 datasets_path: str,
                 tokenizer_path: str
                 ) -> None:
        logger.info('AnekDataset(%s) initialization started.', dataset_type)
        self.tokenizer = Tokenizer(tokenizer_path)
        
        with open(f'{datasets_path}{dataset_type}.json', 'r') as f:
            self.aneks = json.load(f)
        logger.info('%s loaded to self.aneks.', dataset_type)
        logger.info('AnekDataset(%s) initialization completed.', dataset_type)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        anek = self.aneks[str(idx)]
        return anek

    def __len__(self) -> int:
        return len(self.aneks)
