LEXICON_COMMANDS_RU: dict[str, str] = {
    '/start': '😊 Начнем все с чистого листа',
    '/help': ' 🆘 Ничего не понимаю, нужна помощь!',
    '/contact': ' 📲 Связаться в автором'
}

START_MESSAGE = """
🤖 <u><b>"АнекдотоБот 3000" – ваш личный комик в телеграме!</b></u>

<u>Описание:</u>
Привет, Дружище! Я — АнекдотоБот 3000, искусственный интеллект с чувством юмора (ну, почти). Моя миссия — спасать мир от скуки, выдавая шедевры народного творчества и нейросетевого бреда.

<u>Что я умею:</u>
🎭 <b>Генерировать анекдоты</b> – от классики до абсурда (иногда даже смешно!). Вы можете задать начало анекдоту, а я его продолжу!
🛡 <b>Антигрусть-режим</b> – если шутка неудачная, просто нажмите "ещё", и я попробую снова (но без гарантий).

<u>Важно:</u>
⚠️ <b>Бот не виноват</b>, если анекдот несмешной – это нейросети так шутят.
⚠️ <b>Авторские права</b> – все шутки "написал" я (но если узнал чужую, честно скажу).


❗ Если у вас есть вопросы по работе бота – введите /help или выберите соответствующий пункт меню 
"""

HELP_MESSAGE = """
<u><b>"Чем могу помочь?</b></u> 😊

Этот бот умеет генерировать анекдоты. Хотите задать начало шутки? Просто нажмите на кнопку "Префикс", например:
"Вышел заяц на"
"Летели как-то русский, немец и француз"
"Идут 3 богатыря"
(Пишите без кавычек)

Если что-то сломалось или шутки кажутся несмешными — ну, это не баг, а фича! 😅

Доступные команды:
/start — перезапустить бота
/help — это сообщение
/contact — связаться с автором бота
Давайте веселиться! 🎉

❗ Если что-то сломалось – перезапустите бота командой /start
"""

MAIN_MENU_MESSAGE = """
<u><b>Параметры генерации анекдота</b></u>

Префикс: {prefix}
Модель: {model}
"""

CHANGE_MODEL_MESSAGE = 'Выберите архитектуру нейронной сети:'

CHANGE_PREFIX_MESSAGE = 'Введите префикс для анекдота \n("_" - очистить префикс)'

NOT_HANDLED_MESSAGE = 'Я не понимаю👀\n/start'
