""" main file to run a bot """

import os
import csv
from datetime import datetime
import pandas as pd
import telebot
from telebot import types
from src.logger import LOGGER  # pylint: disable=import-error
from .reqs import (Item, get_product_for_user,  # pylint: disable=relative-beyond-top-level
                   update_interactions, # pylint: disable=relative-beyond-top-level
                   escape_description)  # pylint: disable=relative-beyond-top-level

INTERACTION_COUNTER = 0
token = os.environ["TELEGRAM_BOT_TOKEN"]
bot = telebot.TeleBot(token, threaded=False)


def send_user_recommendation(chat_id: str, recommendation: Item):
    """
    Func to send_user_recommendation

    :param chat_id: chat id to send recommendation
    :param recommendation: recommendation to send
    """
    global INTERACTION_COUNTER  # pylint: disable=global-statement
    INTERACTION_COUNTER = INTERACTION_COUNTER + 1

    LOGGER.info(f"send product {recommendation} for chat {chat_id}")
    markup_inline = types.InlineKeyboardMarkup()
    item_yes = types.InlineKeyboardButton(text='Не нравится',
                                          callback_data=f"no {recommendation.item_id}")
    item_no = types.InlineKeyboardButton(text='Нравится',
                                         callback_data=f"yes {recommendation.item_id}")
    markup_inline.add(item_yes, item_no)
    photo_caption = (f"Категория: {recommendation.info.category}\n"
                     f"Описание: {escape_description(recommendation.info.description)}\n"
                     f"Цена: {escape_description(str(recommendation.info.price))}\n"
                     f"[ссылка]({recommendation.link})")
    LOGGER.info(f"photo_caption: {photo_caption}")
    bot.send_photo(chat_id, recommendation.image_link, caption=photo_caption,
                   reply_markup=markup_inline, parse_mode="MarkdownV2")
    if INTERACTION_COUNTER % 20 == 0:
        update_interactions(os.path.abspath("./data/interactions.csv"))


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


@bot.message_handler(commands=['help'])
def help_handler(message):
    """
    Print help data

    :param message: message of help command
    """
    data = "/start - начать выбирать товары\n/reset - удалить вашу историю выборов"
    bot.send_message(message.chat.id, data)


@bot.message_handler(commands=['reset'])
def reset_handler(message):
    """
    Delete user interactions

    :param message: message of reset command
    """
    user_id = message.from_user.id

    interactions = pd.read_csv("./data/interactions.csv")
    without_a_user = interactions[interactions.user_id != user_id]
    without_a_user.to_csv('./data/interactions.csv', index=False, header=True)

    bot.send_message(message.chat.id, "Удалили вашу историю")


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    """
    get data from callback

    :param call: callback query of recommendation
    """
    user_id = call.from_user.id
    interaction = 0
    text = ""
    if call.data.startswith("yes"):
        item_id = int(call.data[len("yes "):])
        interaction = 1
        text = "\nВам понравился этот товар😌"
        LOGGER.info(f"add positive reward for user {user_id} item {item_id}")
    elif call.data.startswith("no"):
        item_id = int(call.data[len("no "):])
        text = "\nВам не понравился этот товар😅"
        LOGGER.info(f"add negative reward for user {user_id} item {item_id}")
    bot.answer_callback_query(call.id, "Учли ваш выбор")
    fields = [datetime.now(), user_id, item_id, interaction]
    with open("./data/interactions.csv", 'a', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    product = get_product_for_user(user_id)
    send_user_recommendation(call.message.chat.id, product)
    new_caption = call.message.caption + text
    bot.edit_message_caption(caption=new_caption, chat_id=call.message.chat.id,
                             message_id=call.message.message_id,
                             reply_markup='', caption_entities=call.message.caption_entities)


print("start a bot")
bot.infinity_polling()
