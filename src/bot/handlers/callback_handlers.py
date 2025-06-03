from functools import partial
import logging

import asyncio
from aiogram import Router, F
from aiogram.types import CallbackQuery
from aiogram.fsm.context import FSMContext

from src.bot.utils.states import Model
from src.bot.keyboards.change_model_keyboard import change_model_keyboard
from src.bot.keyboards.next_anek import next_anek_keyboard
from src.bot.handlers.user_handlers import main_menu
from src.bot.lexicon.lexicon_ru import (CHANGE_MODEL_MESSAGE, 
                                        CHANGE_PREFIX_MESSAGE)

from src.model.inference import anek_generator

router = Router(name=__name__)
logger = logging.getLogger(__name__)

@router.callback_query(Model.generation, F.data=='menu')
async def transformer_selected(callback_query: CallbackQuery, state: FSMContext):
    await callback_query.message.edit_reply_markup()
    await state.set_state(Model.menu)
    await main_menu(callback_query.message, state)

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
    logger.info('%s (id: %s) chenged the model to RNN',
            callback_query.from_user.full_name,
            callback_query.from_user.id)
    await main_menu(callback_query.message, state)
    await callback_query.message.delete()

@router.callback_query(Model.model, F.data=='gpt_selected')
async def gpt_selected(callback_query: CallbackQuery, state: FSMContext):
    await state.update_data({'model': 'GPT'})
    logger.info('%s (id: %s) chenged the model to GPT',
            callback_query.from_user.full_name,
            callback_query.from_user.id)
    await main_menu(callback_query.message, state)
    await callback_query.message.delete()

@router.callback_query(Model.model, F.data=='lora_selected')
async def gpt_lora_selected(callback_query: CallbackQuery, state: FSMContext):
    await state.update_data({'model': 'GPT_Lora'})
    logger.info('%s (id: %s) chenged the model to GPT_Lora',
            callback_query.from_user.full_name,
            callback_query.from_user.id)
    await main_menu(callback_query.message, state)
    await callback_query.message.delete()

@router.callback_query(Model.menu, F.data=='generate')
async def generate(callback_query: CallbackQuery, state: FSMContext):
    await state.set_state(Model.generation)
    await callback_query.message.delete()
    loop = asyncio.get_event_loop()
    func = partial(anek_generator.generate, **await state.get_data())
    anek = await loop.run_in_executor(None, func)
    logger.info('%s (id: %s) generated a anek',
            callback_query.from_user.full_name,
            callback_query.from_user.id)
    await callback_query.message.answer(anek, reply_markup=next_anek_keyboard())

@router.callback_query(Model.generation, F.data=='next')
async def next_anek(callback_query: CallbackQuery, state: FSMContext):
    await callback_query.message.edit_reply_markup()
    loop = asyncio.get_event_loop()
    func = partial(anek_generator.generate, **await state.get_data())
    anek = await loop.run_in_executor(None, func)
    logger.info('%s (id: %s) generated next anek',
            callback_query.from_user.full_name,
            callback_query.from_user.id)
    await callback_query.message.answer(anek, reply_markup=next_anek_keyboard())

@router.callback_query()
async def not_handled(callback_query: CallbackQuery, state: FSMContext):
    await callback_query.message.delete()
    await state.set_state(Model.menu)
    await state.set_data({'model': 'GPT', 'prefix': ''})
    await main_menu(callback_query.message, state)


