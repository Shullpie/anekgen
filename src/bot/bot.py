import os

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from src.bot import handlers
from src.bot.keyboards.set_menu import set_main_menu
load_dotenv()

async def start_bot():
    bot_token = os.getenv('BOT_TOKEN')
    bot = Bot(bot_token)
    dp = Dispatcher()
    dp.include_router(handlers.router)

    await bot.delete_webhook(drop_pending_updates=True)
    await set_main_menu(bot)
    await dp.start_polling(bot)

# asyncio.run(main())
