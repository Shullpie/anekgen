from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardButton

def change_model_keyboard():
    rnn_model = InlineKeyboardButton(
        text='RNN', 
        callback_data='rnn_selected'
    )
    gpt_model = InlineKeyboardButton(
        text='GPT',
        callback_data='gpt_selected'
    )
    lora_model = InlineKeyboardButton(
        text='GPT_LORA',
        callback_data='lora_selected'
    )
    return InlineKeyboardBuilder([[rnn_model, gpt_model], [lora_model]]).as_markup()
