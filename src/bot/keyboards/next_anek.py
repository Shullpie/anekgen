from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardButton

def next_anek_keyboard():
    main_menu = InlineKeyboardButton(
        text='Меню', 
        callback_data='menu'
    )
    next_anek = InlineKeyboardButton(
        text='Еще!', 
        callback_data='next'
    )
    return InlineKeyboardBuilder([[main_menu, next_anek]]).as_markup()
