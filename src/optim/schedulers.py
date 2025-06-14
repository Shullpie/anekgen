from typing import Callable

from torch.optim import lr_scheduler
from torch.optim import Optimizer
from transformers import get_cosine_schedule_with_warmup

def get_scheduler(optimizer: Optimizer, option_scheduler: dict) -> Callable:
    scheme = option_scheduler.get('scheme', None)
    if scheme is None:
        raise NotImplementedError('Scheduler is None. Please, add to config file')

    scheme = scheme.lower()
    if scheme == 'linearlr':
        end_factor = float(option_scheduler.get('end_factor', 1.0))
        total_iters = option_scheduler.get('total_iters', 5)

        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=total_iters,
            last_epoch=-1)
    
    elif scheme == "reducelronplateau":
        factor = option_scheduler.get('factor', 0.1)
        patience = option_scheduler.get('patience', 10)
        threshold = float(option_scheduler.get("threshold", 1e-4))
        threshold_mode = option_scheduler.get('threshold_mode', "rel")
        cooldown = option_scheduler.get('cooldown', 0)
        eps = float(option_scheduler.get('eps', 1e-8))

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            eps=eps
        )

    elif scheme == 'multistep':
        milestones = option_scheduler.get('milestones')
        gamma = option_scheduler['gamma']

        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
            last_epoch=-1)
    
    elif scheme == "exponentiallr":
        gamma = option_scheduler['gamma']

        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
            last_epoch=-1
        )
    elif scheme == 'cosine_warmup':
        total_training_steps = option_scheduler.get('total_training_steps')
        num_warmup_steps = int(total_training_steps * 0.05)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=total_training_steps,
                                                    num_warmup_steps=num_warmup_steps)
    else:
        raise NotImplementedError(
            f'Neural Network [{scheme}] is not recognized. networks.py doesn\'t know {[scheme]}')
    return scheduler
