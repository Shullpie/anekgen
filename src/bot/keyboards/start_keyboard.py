from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardButton

def start_keyboard():
    start = InlineKeyboardButton(
        text='Начнем!', 
        callback_data='main_menu'
    )
    return InlineKeyboardBuilder([[start]]).as_markup()
