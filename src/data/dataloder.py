from typing import Literal
import logging

from torch.utils.data import DataLoader

from src.data.datasets import create_dataset
from src.data.collate_fn import collate_fn
from config import config

logger = logging.getLogger(__name__)

def get_dataloder(dataset_type: Literal['train'] | Literal['test']) -> DataLoader:
    logger.info('%sDataloader initialization started', dataset_type)
    dataset = create_dataset(dataset_type=dataset_type)
    batch_size = config['dataloader']['batch_size']
    dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn(dataset.tokenizer.pad_idx), 
        pin_memory=True,
        shuffle=True
    )
    logger.info('%sDataloader initialization completed.', dataset_type)
    return dl
    

