
import torch
from src.model.basemodel import BaseModel
from tqdm import tqdm
from torch.amp import autocast

class RNNModel(BaseModel):
    def __init__(self):
        super().__init__()
    
    def fit(self):
        self.model.to(self.device)
        for epoch in range(1, self.n_epoches+1):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            test_loss = self.test_epoch(epoch)
            self.test_losses.append(test_loss)
            self.scheduler.step(train_loss)

            if self.make_checkpoint_every_n_epoch and epoch % self.make_checkpoint_every_n_epoch == 0:
                self._make_checkpoint(epoch)
                self._save_embeddings(epoch)
            
            if self.empty_cuda_cache_every_n_epoch and epoch%self.empty_cuda_cache_every_n_epoch == 0:
                torch.cuda.empty_cache()

            print(train_loss, test_loss)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0

        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.n_epoches} - Train', ncols=200):
            loss = self.train_batch(batch)
            epoch_loss += loss
        
        epoch_loss = epoch_loss / self.n_train_batches
        return epoch_loss


    def train_batch(self, batch: torch.Tensor) -> float:
        
        batch = batch.to(self.device)
        with autocast(device_type=self.device, dtype=torch.float16, enabled=self.amp):
            output = self.model(batch[:, :-1]).transpose(1, 2)
            loss = self.loss(output, batch[:, 1:])

        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

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
        batch = batch.to(self.device)
        output = self.model(batch[:, :-1]).transpose(1, 2)
        loss = self.loss(output, batch[:, 1:])
        return loss.item()

