import logging

from torch.nn import BCEWithLogitsLoss

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
        logger.info('Make checpoints every n epoches')

        self.loss = BCEWithLogitsLoss()
        logger.info('Loss function: %s.', self.loss._get_name())


    def fit(self):
        NotImplementedError('Do not use BaseModel. Please, use concrete pipeline instand.')

