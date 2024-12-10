""" main file to run a bot """

import os
import csv
from datetime import datetime
import telebot
from telebot import types
from src.logger import LOGGER  # pylint: disable=import-error
from .reqs import Item, get_product_for_user  # pylint: disable=relative-beyond-top-level

token = os.environ["TELEGRAM_BOT_TOKEN"]
bot = telebot.TeleBot(token, threaded=False)


def send_user_recommendation(chat_id: str, recommendation: Item):
    """
    Func to send_user_recommendation

    :param chat_id: chat id to send recommendation
    :param recommendation: recommendation to send
    """
    LOGGER.info(f"send product {recommendation} for chat {chat_id}")
    markup_inline = types.InlineKeyboardMarkup()
    item_yes = types.InlineKeyboardButton(text='Принять',
                                          callback_data=f"yes {recommendation.item_id}")
    item_no = types.InlineKeyboardButton(text='Отказаться',
                                         callback_data=f"no {recommendation.item_id}")
    markup_inline.add(item_yes, item_no)
    photo_caption = f"Категория: {recommendation.category}\nОписание: {recommendation.description}"
    bot.send_photo(chat_id, recommendation.image_link, caption=photo_caption,
                   reply_markup=markup_inline)


@bot.message_handler(commands=['start'])
def start_handler(message):
    """
    Send message for start command

    :param message: message of start command
    """
    LOGGER.info(f"start message for user {message.from_user.id}")
    bot.send_message(message.chat.id,
                     "Привет, это бот который поможет выбирать товары на Ozon."
                     "Тебе просто надо лайкать любимые товары")
    product = get_product_for_user(message.from_user.id)
    send_user_recommendation(message.chat.id, product)


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    """
    get data from callback

    :param call: callback query of recommendation
    """
    user_id = call.from_user.id
    interaction = 0
    if call.data.startswith("yes"):
        item_id = int(call.data[len("yes "):])
        interaction = 1
        LOGGER.info(f"add positive reward for user {user_id} item {item_id}")
    elif call.data.startswith("no"):
        item_id = int(call.data[len("no "):])
        LOGGER.info(f"add negative reward for user {user_id} item {item_id}")
    bot.answer_callback_query(call.id, "Учли ваш выбор")
    fields = [datetime.now(), user_id, item_id, interaction]
    with open("./data/interactions.csv", 'a', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    product = get_product_for_user(user_id)
    send_user_recommendation(call.message.chat.id, product)


print("start a bot")
bot.infinity_polling()
