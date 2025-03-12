import logging
from config import config

def configure_logging():
    logging.basicConfig(
        level=config['logging_level'],
        datefmt='%Y-%m-%d %H:%M:%S',
        format='[%(asctime)s.%(msecs)03d] %(module)-15s %(lineno)3d %(levelname)7s : %(message)s'
    )
