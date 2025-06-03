import os
import logging

import torch
from torch.amp import autocast
from peft import PeftModel, get_peft_model, LoraConfig

from src.model.train_model import TrainModel

logger = logging.getLogger(__name__)

class LoraModel(TrainModel):
    def __init__(self, config):
        super().__init__(config)
        model_config = config['archs']['gpt_lora']

        self.gradient_accumulation_steps = model_config.get('gradient_accumulation_steps', 1)
        self.total_steps = 0

        logger.info('gradient accumulation steps: %d', self.gradient_accumulation_steps)
        self.model.print_trainable_parameters()


    def train_batch(self, batch: torch.Tensor) -> float:
        with autocast(device_type=self.device, dtype=torch.float16, enabled=self.amp):
            output = self.model(**self.prepare_batch(batch)).logits.transpose(1, 2)

        loss = self.loss(output, self.prepare_batch(batch, train=False)['input_ids']) / self.gradient_accumulation_steps
        loss.backward()

        if (self.total_steps+1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.total_steps += 1
        return loss.item()
    
    def test_batch(self, batch: torch.Tensor) -> int:
        output = self.model(**self.prepare_batch(batch)).logits.transpose(1, 2)
        loss = self.loss(output, self.prepare_batch(batch, train=False)['input_ids'])
        return loss.item()
    
    def make_checkpoint(self, curr_epoch: int) -> None:
        path = self.save_checkpoint_path + f'{self.model._get_name()}_{type(self.optimizer).__name__}_' + \
            f'{type(self.scheduler).__name__}/'
        path = f'{path.lower()}{curr_epoch}'

        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        try:
            torch.save(
                {
                    'epoch': curr_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'test_losses': self.test_losses,
                    'total_steps': self.total_steps
                },
                path+f'/state.pt'
            )
            self.model.save_pretrained(path)
        except Exception as ex:
            self.logger.error(ex, exc_info=True)
    
    def load_model_from_checkoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(f'{checkpoint_path}/state.pt', map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.curr_epoch = checkpoint['epoch'] + 1
        self.total_steps = checkpoint['total_steps']
        
