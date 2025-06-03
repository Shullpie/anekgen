import os
import logging

import torch
from torch.nn import CrossEntropyLoss
from torch.amp import GradScaler

from src.data.dataloader import get_dataloader
from src.archs import get_nn_model
from src.optim.optimizers import get_optimizer
from src.optim.schedulers import get_scheduler

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, config: dict):
        self.curr_epoch = 1        
        self.train_losses = []
        self.test_losses = []
        
        self.device = config['device']
        logger.info('Device: %s.', self.device)
        print(logger)
        
        selected_model = config['selected_model']
        self.train_loader = get_dataloader(dataset_type='train', 
                                          dataset_path=config['data']['aneks_path'],
                                          batch_size=config['data']['batch_size'],
                                          tokenizer_path=config['archs'][selected_model]['tokenizer_path'])
        self.test_loader = get_dataloader(dataset_type='test', 
                                          dataset_path=config['data']['aneks_path'],
                                          batch_size=config['data']['batch_size'],
                                          tokenizer_path=config['archs'][selected_model]['tokenizer_path'])

        self.batch_size = self.train_loader.batch_size
        self.n_train_batches = len(self.train_loader)
        self.n_test_batches = len(self.test_loader)

        pad_idx = self.train_loader.dataset.tokenizer.pad_idx
        vocab_size = self.train_loader.dataset.tokenizer.vocab_size
        load_checkpoint_path = config['load_checkpoint_path']
        

        self.model = get_nn_model(model_name=config['selected_model'],
                                  config=config,
                                  pad_idx=pad_idx,
                                  vocab_size=vocab_size,
                                  load_checkpoint_path=load_checkpoint_path
                                  ).to(self.device)
        logger.info('Model params: \n %s', self.model)

        self.n_epoches = config['epoches']
        logger.info('Number of epoches: %s.', self.n_epoches)

        self.make_checkpoint_every_n_epoch = config['make_checkpoint_every_n_epoch']
        if self.make_checkpoint_every_n_epoch:
            logger.info('Make checpoints every %s epoches.', self.make_checkpoint_every_n_epoch)

        self.loss = CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
        logger.info('Loss function: %s.', self.loss._get_name())

        self.amp = config['amp']['enabled']
        if self.amp:
            self.scaler = GradScaler(init_scale=config['amp']['scaler_init_scale'])
            logger.info('AMP enabled.')
        else: 
            logger.info('AMP disabled.')
        
        self.save_checkpoint_path = config['save_checkpoint_path']
        
        self.optimizer = get_optimizer(model_params=self.model.parameters(), 
                                       optimizer_options=config['optimizer'])
        self.scheduler = get_scheduler(optimizer=self.optimizer, 
                                       option_scheduler=config['scheduler'])
        
        if load_checkpoint_path:
            self.load_model_from_checkoint(load_checkpoint_path)
            logger.info('Model loaded from %s.', load_checkpoint_path)
        
        logger.info('Optimizer: %s.', self.optimizer)
        logger.info('Scheduler: %s.', self.scheduler)
        logger.info('Scheduler params: %s.', self.scheduler.state_dict())

    def fit(self):
        NotImplementedError('Do not use BaseModel. Please, use concrete pipeline instand.')

    def make_checkpoint(self, cur_epoch: int) -> None:
        path = self.save_checkpoint_path + f'{self.model._get_name()}_{type(self.optimizer).__name__}_' + \
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

    def load_model_from_checkoint(self, checkpoint_path: str) -> None:
        self.model.to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.curr_epoch = checkpoint['epoch'] + 1

