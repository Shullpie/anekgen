from src.model.inference.inference_anekgenerator import AnekGenerator
from config import inference_config

__all__=('anek_generator', )
anek_generator = AnekGenerator(config=inference_config)

