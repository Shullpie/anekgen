from typing import Literal
from .anekdataset import AnekDataset

def create_dataset(dataset_type: Literal['train'] | Literal['test']) -> AnekDataset:
    return AnekDataset(dataset_type=dataset_type)
