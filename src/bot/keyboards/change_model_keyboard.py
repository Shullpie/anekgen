from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardButton

def change_model_keyboard():
    rnn_model = InlineKeyboardButton(
        text='RNN', 
        callback_data='rnn_selected'
    )
    transformer_model = InlineKeyboardButton(
        text='🤖 Трансформер [заменить на архитектуру]',
        callback_data='transfromer_selected'
    )
    return InlineKeyboardBuilder([[rnn_model, transformer_model]]).as_markup()
