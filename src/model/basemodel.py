import os
import logging

import torch
from torch.nn import CrossEntropyLoss
from torch.amp import GradScaler

from src.data.dataloder import get_dataloder
from src.archs import get_nn_model
from src.archs.optim.optimizers import get_optimizer
from src.archs.optim.schedulers import get_scheduler
from config import config

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self):
        self.train_loader = get_dataloder('train')
        self.test_loader = get_dataloder('test')
        self.n_train_batches = len(self.train_loader)
        self.n_test_batches = len(self.test_loader)

        self.model = get_nn_model(model_name=config['model'],
                                  pad_idx=self.train_loader.dataset.tokenizer.get_pad_idx(),
                                  vocab_size=len(self.train_loader.dataset.tokenizer))
                                  
        self.optimizer = get_optimizer(model_params=self.model.parameters(), 
                                       optimizer_options=config['optimizer'])
        self.scheduler = get_scheduler(optimizer=self.optimizer, 
                                       option_scheduler=config['scheduler'])
        
        self.device = config['device']
        logger.info('Device: %s.', self.device)

        self.n_epoches = config['epoches']
        logger.info('Number of epoches: %s.', self.n_epoches)

        self.make_checkpoint_every_n_epoch = config['make_checkpoint_every_n_epoch']
        if self.make_checkpoint_every_n_epoch:
            logger.info('Make checpoints every %d epoches.', self.make_checkpoint_every_n_epoch)
        
        self.empty_cuda_cache_every_n_epoch = config['empty_cuda_cache_every_n_epoch']
        if self.empty_cuda_cache_every_n_epoch:
            logger.info('Empty cuda cache every %d.', self.empty_cuda_cache_every_n_epoch)

        self.loss = CrossEntropyLoss()
        logger.info('Loss function: %s.', self.loss._get_name())

        self.amp = config['amp']['enabled']

        if self.amp:
            self.scaler = GradScaler(init_scale=config['amp']['scaler_init_scale'])
            logger.info('AMP enabled.')
        
        self.save_checkpoint_path = config['save_checkpoint_path']
        
        load_checkpoint_path = config['load_checkpoint_path']
        if load_checkpoint_path:
            self.load_model_from_checkoint(load_checkpoint_path)
            logger.info('Model loaded from %s.', load_checkpoint_path)

        embeddings_path = config['embeddings_path']
        if embeddings_path:
            self.load_embeddings(embeddings_path)
            logger.info('Embeddings loaded from %s.', embeddings_path)

        self.train_losses = []
        self.test_losses = []


    def fit(self):
        NotImplementedError('Do not use BaseModel. Please, use concrete pipeline instand.')

    def _make_checkpoint(self, cur_epoch: int) -> None:
        path = config['save_checkpoint_path'] + f'{self.model._get_name()}_{type(self.optimizer).__name__}_' + \
            f'{type(self.scheduler).__name__}/'
        path = path.lower()

        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        try:
            torch.save(
                {
                    'epoch': cur_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'test_losses': self.test_losses
                },
                path+f'{cur_epoch}_epoch.pt'
            )
        except Exception as ex:
            self.logger.error(ex, exc_info=True)
    
    def _save_embeddings(self, cur_epoch: int) -> None:
        path = config['save_embeddings_path']
        file_name = f'{self.model._get_name()}_{type(self.optimizer).__name__}_' + \
            f'{type(self.scheduler).__name__}_{cur_epoch}.pt'
        file_name = file_name.lower()

        try:
            torch.save(self.model.embeddings.weight.data, path+file_name)
        except Exception as ex:
            self.logger.error(ex, exc_info=True)


    def load_model_from_checkoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.curr_epoch = checkpoint['epoch'] + 1

    def load_embeddings(self, embeddings_path: str):
        loaded_embeddings = torch.load(embeddings_path)
        self.model.embeddings.from_pretrained(loaded_embeddings)
