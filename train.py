from config import config
from src.utils.logger import configure_logging
# from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    configure_logging(config['logging_level'])
    if config['selected_model'] == 'gpt_lora':
        from src.model.gpt_lora_model import LoraModel as model
    else:
        from src.model.train_model import TrainModel as model
    m = model(config=config)

    # m.scheduler._last_lr[0] = 1e-5
    # for g in m.optimizer.param_groups:
    #     g['lr'] = 1e-5
    # m.scheduler._last_lr[0] = 1e-5

    m.fit()


if __name__ == "__main__":
    main()
