""" main file to run a bot """

import os
import telebot
from telebot import types
from src.logger import LOGGER  #pylint: disable=E0401

token = os.environ["TELEGRAM_BOT_TOKEN"]
bot = telebot.TeleBot(token, threaded=False)


class Item:
    """ Item to keep info about an item """
    def __init__(self, item_id, image_link, category, description):
        self.item_id = item_id
        self.image_link = image_link
        self.category = category
        self.description = description

    def save_image(self):
        """ save image locally by image link """

    def __repr__(self):
        return f"Item(name={self.item_id}, item_id={self.image_link}, "\
                f"link={self.category}, cat1={self.description})"


def get_product_for_user(user_id: str) -> Item:
    """ a simple version of a bandit to get item """
    LOGGER.info(f"get a product for user {user_id}")
    return Item(1, "https://clck.ru/3F8PEP", "Категория1", "description")


def send_user_recommendation(chat_id: str, recommendation: Item):
    """ func to send_user_recommendation """
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
    """ send message for start command """
    LOGGER.info(f"start message for user {message.from_user.id}")
    bot.send_message(message.chat.id,
                     "Привет, это бот который поможет выбирать товары на Ozon."
                     "Тебе просто надо лайкать любимые товары")
    product = get_product_for_user(message.from_user.id)
    send_user_recommendation(message.chat.id, product)


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    """ get data from callback """
    user_id = call.from_user.id
    if call.data.startswith("yes"):
        item_id = call.data[len("yes "):]
        LOGGER.info(f"add positive reward for user {user_id} item {item_id}")
    elif call.data.startswith("no"):
        item_id = call.data[len("no "):]
        LOGGER.info(f"add negativw reward for user {user_id} item {item_id}")
    bot.answer_callback_query(call.id, "Учли ваш выбор")
    product = get_product_for_user(user_id)
    send_user_recommendation(call.message.chat.id, product)


bot.infinity_polling()
