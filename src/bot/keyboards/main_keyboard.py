from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardButton

def main_menu_keyboard():
    generate_button = InlineKeyboardButton(
        text='🤣 Начать генерацию', 
        callback_data='generate'
    )
    change_model = InlineKeyboardButton(
        text='🤖 Модель',
        callback_data='change_model'
    )
    change_prefix = InlineKeyboardButton(
        text='✍️ Префикс', 
        callback_data='change_prefix'
    )
    return InlineKeyboardBuilder([[generate_button], [change_model, change_prefix]]).as_markup()
