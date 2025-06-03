import logging

from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext

from src.bot.utils.states import Model
from src.bot.keyboards.main_keyboard import main_menu_keyboard
from src.bot.lexicon.lexicon_ru import START_MESSAGE, HELP_MESSAGE, MAIN_MENU_MESSAGE

router = Router(name=__name__)
logger = logging.getLogger(__name__)

@router.message(Command(commands=['start']))
async def command_start(message: Message, state: FSMContext):
    logger.info('%s (id: %s) started the bot',
                message.from_user.full_name,
                message.from_user.id)
    await state.set_state(Model.menu)
    await state.set_data({'model': 'GPT', 'prefix': ''})
    await message.answer(
        START_MESSAGE, 
        parse_mode=ParseMode.HTML
    )
    await main_menu(message, state)
    
@router.message(Command(commands=['help']))
async def command_help(message: Message):
    logger.info('%s (id: %s) received help',
                message.from_user.full_name,
                message.from_user.id)
    await message.answer(HELP_MESSAGE, parse_mode=ParseMode.HTML)

@router.message(Model.prefix, F.text)
async def filled_prefix(message: Message, state: FSMContext):
    logger.info('%s (id: %s) added a prfix',
            message.from_user.full_name,
            message.from_user.id)
    await state.update_data({'prefix': message.text})
    await main_menu(message, state)
    message.delete()

@router.message(Command(commands=['menu']))
async def main_menu(message: Message, state: FSMContext):
    await state.set_state(Model.menu)

    saved_data = await state.get_data()
    prefix = saved_data.get('prefix')
    model = saved_data.get('model')
    
    await message.answer(
        text=MAIN_MENU_MESSAGE.format(prefix=prefix, model=model), 
        reply_markup=main_menu_keyboard(),
        parse_mode=ParseMode.HTML
    )
    


