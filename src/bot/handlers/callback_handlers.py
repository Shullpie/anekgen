from aiogram import Router, F
from aiogram.types import CallbackQuery
from aiogram.fsm.context import FSMContext

from src.bot.utils.states import Model
from src.bot.keyboards.change_model_keyboard import change_model_keyboard
from src.bot.handlers.user_handlers import main_menu
from src.bot.lexicon.lexicon_ru import (CHANGE_MODEL_MESSAGE, 
                                        CHANGE_PREFIX_MESSAGE)

router = Router(name=__name__)

@router.callback_query(Model.menu, F.data=='change_model')
async def change_model(callback_query: CallbackQuery, state: FSMContext):
    await state.set_state(Model.model)
    await callback_query.message.edit_text(
        text=CHANGE_MODEL_MESSAGE, 
        reply_markup=change_model_keyboard()
    )

@router.callback_query(Model.menu, F.data=='change_prefix')
async def change_prefix(callback_query: CallbackQuery, state: FSMContext):
    await state.set_state(Model.prefix)
    await callback_query.message.edit_text(
        text=CHANGE_PREFIX_MESSAGE)

@router.callback_query(Model.model, F.data=='rnn_selected')
async def rnn_selected(callback_query: CallbackQuery, state: FSMContext):
    await state.update_data({'model': 'RNN'})
    await main_menu(callback_query.message, state)
    await callback_query.message.delete()

@router.callback_query(Model.model, F.data=='transfromer_selected')
async def transformer_selected(callback_query: CallbackQuery, state: FSMContext):
    await state.update_data({'model': 'BERT'})
    await main_menu(callback_query.message, state)
    await callback_query.message.delete()



