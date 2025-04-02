from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardButton

def start_keyboard():
    rnn_model = InlineKeyboardButton(
        text='Начнем!', 
        callback_data='main_menu'
    )
    return InlineKeyboardBuilder([[rnn_model]]).as_markup()
