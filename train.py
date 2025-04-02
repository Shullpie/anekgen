from config import config
from src.utils.logger import configure_logging
from src.model.rnn_model import RNNModel

def main():
    configure_logging(config['logging_level'])
    m = RNNModel()
    # m.scheduler._last_lr[0] = 1e-4
    # for g in m.optimizer.param_groups:
    #     g['lr'] = 1e-3
    # m.scheduler._last_lr[0] = 1e-3
    m.fit()
if __name__ == "__main__":
    main()
#108
