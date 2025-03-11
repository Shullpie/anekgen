import yaml
import argparse
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / 'config.yaml'


with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

__all__ = ['config']

