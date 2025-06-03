from typing import Literal
import logging

from torch.utils.data import DataLoader

from src.data.datasets import create_dataset
from src.data.collate_fn import collate_fn

logger = logging.getLogger(__name__)

def get_dataloader(dataset_type: Literal['train'] | Literal['test'], 
                   dataset_path: str,
                   tokenizer_path: str,
                   batch_size: int
                   ) -> DataLoader:
    logger.info('%sDataloader initialization started', dataset_type)
    dataset = create_dataset(dataset_type=dataset_type, 
                             dataset_path=dataset_path, 
                             tokenizer_path=tokenizer_path)
    dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn(dataset.tokenizer, 
                              return_tensors="pt", 
                              padding=True, 
                              truncation=True, 
                              return_token_type_ids=False),#collate_fn(dataset.tokenizer.pad_idx), 
        pin_memory=True,
        shuffle=True,
    )
    logger.info('%sDataloader initialization completed.', dataset_type)
    return dl
    

