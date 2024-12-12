""" main file to run a bot """

import os
import csv
from collections import defaultdict
from datetime import datetime
import secrets
import pandas as pd
import telebot
from telebot import types
from src.logger import LOGGER  # pylint: disable=import-error
from .reqs import (Item, get_product_for_user,  # pylint: disable=relative-beyond-top-level
                   update_interactions,  # pylint: disable=relative-beyond-top-level
                   escape_description)  # pylint: disable=relative-beyond-top-level

USERS_TO_REWARD = defaultdict(int)
USER_REWARD_FILE = "./data/rewards.txt"

with open(USER_REWARD_FILE, "a", encoding="utf-8") as f1:
    pass

with open(USER_REWARD_FILE, 'r+', encoding="utf-8") as f1:
    for line in f1.readlines():
        line = line.strip()
        args = line.split()
        if len(args) != 2:
            continue
        uid = args[0]
        if args[1] == "0":
            USERS_TO_REWARD[uid] += 1
        else:
            USERS_TO_REWARD[uid] -= 1
            if USERS_TO_REWARD[uid] <= 0:
                USERS_TO_REWARD.pop(uid)

INTERACTION_DATETIME = datetime.now()
token = os.environ["TELEGRAM_BOT_TOKEN"]
bot = telebot.TeleBot(token, threaded=False)


def send_user_recommendation(user_id: str, chat_id: str, recommendation: Item):
    """
    Func to send_user_recommendation

    :param chat_id: chat id to send recommendation
    :param recommendation: recommendation to send
    """
    global INTERACTION_DATETIME  # pylint: disable=global-statement

    LOGGER.info(f"send product {recommendation} for chat {chat_id}")
    markup_inline = types.InlineKeyboardMarkup()
    item_yes = types.InlineKeyboardButton(text='Не нравится',
                                          callback_data=f"no {recommendation.item_id}")
    item_no = types.InlineKeyboardButton(text='Нравится',
                                         callback_data=f"yes {recommendation.item_id}")
    markup_inline.add(item_yes, item_no)
    photo_caption = (f"Категория: {escape_description(recommendation.info.category)}\n"
                     f"Описание: {escape_description(recommendation.info.description)}\n"
                     f"Цена: {escape_description(str(recommendation.info.price))}\n"
                     f"[ссылка]({recommendation.link})")
    LOGGER.info(f"photo_caption: {photo_caption}")
    bot.send_photo(chat_id, recommendation.image_link, caption=photo_caption,
                   reply_markup=markup_inline, parse_mode="MarkdownV2")
    time = datetime.now()
    if (time - INTERACTION_DATETIME).total_seconds() > 30:
        INTERACTION_DATETIME = time
        users = update_interactions(os.path.abspath("./data/interactions.csv"))
        with open(USER_REWARD_FILE, 'a', encoding="utf-8") as f2:
            for user in users:
                USERS_TO_REWARD[user] += 1
                f2.writelines([f"{user_id} 0\n"])
    if user_id in USERS_TO_REWARD.keys():
        hsh = secrets.token_hex(nbytes=16)[:15]
        bot.send_message(chat_id,
                         f"Поздравляю с усердной работой! Ваш токен: {hsh}.erg. Спасибо!")
        USERS_TO_REWARD[user_id] -= 1
        if USERS_TO_REWARD[user_id] == 0:
            USERS_TO_REWARD.pop(user_id)
        with open(USER_REWARD_FILE, 'a', encoding="utf-8") as f3:
            f3.writelines([f"{user_id} {hsh}\n"])

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
    send_user_recommendation(message.from_user.id, message.chat.id, product)


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
    # bot.answer_callback_query(call.id, "Учли ваш выбор")
    fields = [datetime.now(), user_id, item_id, interaction]
    with open("./data/interactions.csv", 'a', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    product = get_product_for_user(user_id)
    send_user_recommendation(user_id, call.message.chat.id, product)
    try:
        new_caption = call.message.caption + text
        bot.edit_message_caption(caption=new_caption, chat_id=call.message.chat.id,
                             message_id=call.message.message_id,
                             reply_markup='', caption_entities=call.message.caption_entities)
    except Exception as error:
        LOGGER.error(str(error))


print("start a bot")
bot.infinity_polling()
