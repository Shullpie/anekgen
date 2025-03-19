import logging

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

def get_optimizer(model_params: torch.Tensor, optimizer_options: dict) -> Optimizer:
    name = optimizer_options.get('name', None).lower()
    if name is None:
        raise NotImplementedError(
            'Optimizer is None. Please, add to config file')
    name = name.lower()

    optimizer = None
    if name == "sgd":
        lr = float(optimizer_options.get("lr"))
        momentum = float(optimizer_options.get("momentum", 0.0))
        weight_decay = float(optimizer_options.get('weight_decay', 0.0))
        nesterov = optimizer_options.get('nesterov', False)
        dampening = float(optimizer_options.get('dampening', 0.0))

        optimizer = torch.optim.SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            dampening=dampening
        )
        
    elif name in ('adam', 'adamw'):
        lr = float(optimizer_options.get('lr'))
        beta1 = float(optimizer_options.get('beta1', 0.9))
        beta2 = float(optimizer_options.get('beta2', 0.999))
        weight_decay = float(optimizer_options.get('weight_decay', 0.0))
        amsgrad = optimizer_options.get('amsgrad', False)

        class_optimizer = torch.optim.AdamW if name == 'adamw' else torch.optim.Adam

        optimizer = class_optimizer(
            model_params, 
            lr=lr, 
            betas=(beta1, beta2), 
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )

    elif name in 'adamax':
        lr = float(optimizer_options.get('lr'))
        beta1 = float(optimizer_options.get('beta1', 0.9))
        beta2 = float(optimizer_options.get('beta2', 0.999))
        weight_decay = float(optimizer_options.get('weight_decay', 0.0))

        optimizer = torch.optim.Adamax(
            model_params, 
            lr=lr, 
            betas=(beta1, beta2), 
            weight_decay=weight_decay
        )
        
    else:
        raise NotImplementedError(
            f'Optimizer [{name}] is not recognized. optimizers.py doesn\'t know {[name]}.')
    logger.info('Optimizer: %s.', optimizer)
    return optimizer
