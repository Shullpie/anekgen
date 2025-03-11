import json

import pandas as pd
from torch.nn.utils import Dataset

import config


class AneksDataset(Dataset):
    def __init__(self, path: str, load_to_ram: bool = True) -> None:
        self.path = path
        self.load_to_ram = load_to_ram

        if self.load_to_ram:
            aneks = pd.read_csv(config['data']['aneks_path'], sep=';')

        with open(config['data']['vocab_word2idx']) as w2i, open(config['data']['vocab_idx2embedding']) as i2e:
            self.word2idx = json.load(w2i)
            self.idx2embedding = json.load(i2e)
        


    def __getitem__(self, idx: int):
        NotImplementedError('Do not use BaseDataSet. Please, use concrete pipeline instand.')

    def __len__(self) -> int:
        pass
