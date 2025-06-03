import logging

import torch
from src.model.basemodel import BaseModel
from tqdm import tqdm
from torch.amp import autocast

logger = logging.getLogger(__name__)

class TrainModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config=config)
    
    def fit(self):
        self.model.to(self.device)
        for epoch in range(self.curr_epoch, self.n_epoches+1):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            test_loss = self.test_epoch(epoch)
            self.test_losses.append(test_loss)
            self.scheduler.step(train_loss)

            if self.make_checkpoint_every_n_epoch and epoch % self.make_checkpoint_every_n_epoch == 0:
                self.make_checkpoint(epoch)

            print(f'train:, {train_loss}, test:, {test_loss}, lr: {self.scheduler._last_lr[0]}')

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0

        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.n_epoches} - Train', ncols=200):
            loss = self.train_batch(batch)
            epoch_loss += loss
        self.optimizer.zero_grad()
        epoch_loss = epoch_loss / self.n_train_batches
        return epoch_loss


    def train_batch(self, batch: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        with autocast(device_type=self.device, dtype=torch.float16, enabled=self.amp):
            output = self.model(**self.prepare_batch(batch)).transpose(1, 2)
            loss = self.loss(output, self.prepare_batch(batch, train=False)['input_ids'])

        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

        return loss.item()

    @torch.inference_mode()
    def test_epoch(self, epoch: int) -> float:
        epoch_loss = 0
        for batch in tqdm(self.test_loader, desc=f'Epoch {epoch}/{self.n_epoches} - Test ', ncols=200):
            loss = self.test_batch(batch)
            epoch_loss += loss
        
        epoch_loss = epoch_loss / self.n_test_batches
        return epoch_loss
    
    def test_batch(self, batch: torch.Tensor):
        output = self.model(**self.prepare_batch(batch)).transpose(1, 2)
        loss = self.loss(output, self.prepare_batch(batch, train=False)['input_ids'])
        return loss.item()
    
    def prepare_batch(self, batch, train=True):
        t_batch = batch.copy() 
        for key, value in t_batch.items():
            if train:
                t_batch[key] = value[:, :-1]
            else:
                t_batch[key] = value[:, 1:]
        return t_batch.to(self.device)
# 85
