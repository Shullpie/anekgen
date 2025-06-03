import asyncio
from src.bot.bot import start_bot
from src.utils.logger import configure_logging
from config import inference_config

def main():
    configure_logging(inference_config['logging_level'])
    asyncio.run(start_bot())

if __name__=='__main__':
    main()
