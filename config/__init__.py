import yaml
from pathlib import Path

TRAIN_CONFIG_PATH = Path(__file__).parent / 'config.yaml'
INFERENCE_CONFIG_PATH = Path(__file__).parent / 'inference_config.yaml'


with open(TRAIN_CONFIG_PATH, 'r') as t, open(INFERENCE_CONFIG_PATH, 'r') as i:
    config = yaml.safe_load(t)
    inference_config = yaml.safe_load(i)

__all__ = ['config', 'inference_config']

