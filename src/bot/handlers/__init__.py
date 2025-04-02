__all__ = ('router', )
from aiogram import Router
from src.bot.handlers import user_handlers
from src.bot.handlers import callback_handlers

router = Router(name=__name__)
router.include_routers(user_handlers.router, callback_handlers.router)
