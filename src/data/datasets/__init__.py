from typing import Literal
from .anekdataset import AnekDataset

def create_dataset(dataset_type: Literal['train'] | Literal['test'],
                   dataset_path: str, 
                   tokenizer_path: str
                   ) -> AnekDataset:
    return AnekDataset(dataset_type=dataset_type, 
                       datasets_path=dataset_path,
                       tokenizer_path=tokenizer_path)
