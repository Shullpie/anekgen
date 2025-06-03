from aiogram.fsm.state import State, StatesGroup

class Model(StatesGroup):
    menu = State()
    model = State(state='gpt')
    prefix = State(state='')
    generation = State()
