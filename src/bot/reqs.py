"""file to send requests to bandits"""
# pylint: disable=global-variable-not-assigned

import os
import random
from collections import defaultdict
from datetime import datetime

import requests

from src.logger import LOGGER  # pylint: disable=import-error

port = int(os.environ["FAST_API_PORT"])
session = requests.Session()
USER_RECOMMEND = defaultdict(list)
USERS_TO_REWARD = defaultdict(int)
USER_REWARD_FILE = "./data/rewards.txt"
INTERACTION_DATETIME = datetime.now()


class ItemInfo:
    """ItemInfo to keep info"""

    def __init__(self, category, description, price):
        self.category = category
        self.description = description
        self.price = price

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            f"category={self.category}, description={self.description}, "
            f"price={self.price}"
        )


class Item:
    """Item to keep info about an item"""

    def __init__(self, item_id, image_link, link, info):
        self.item_id = item_id
        self.image_link = image_link
        self.info = info
        self.link = link

    def save_image(self):
        """save image locally by image link"""

    def __repr__(self):
        return (
            f"Item(name={self.item_id}, item_id={self.image_link}, "
            f"link={self.info.category}, cat1={self.info.description})"
        )


def escape_description(text: str):
    """ "
    Format string to use in markdownv2

    :param text: string to format
    :return formated text for markdownv2
    """
    symbols = [
        "_",
        "*",
        "[",
        "]",
        "(",
        ")",
        "~",
        "`",
        ">",
        "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
        ":",
    ]
    for symbol in symbols:
        text = text.replace(symbol, "\\" + symbol)
    return text


def get_product_for_user(user_id: str) -> Item:
    """
    A simple version of a bandit to get item

    :param user_id: user_id get id of user to get a recommendations
    :return recommendation for a user
    """
    global USER_RECOMMEND, INTERACTION_DATETIME, USER_REWARD_FILE, USERS_TO_REWARD
    if len(USER_RECOMMEND[user_id]) != 0:
        item = USER_RECOMMEND[user_id][0]
        USER_RECOMMEND[user_id] = USER_RECOMMEND[user_id][1:]
        return item

    time = datetime.now()
    if (time - INTERACTION_DATETIME).total_seconds() > 15:
        INTERACTION_DATETIME = time
        users = update_interactions(os.path.abspath("./data/interactions.csv"))
        with open(USER_REWARD_FILE, "a", encoding="utf-8") as f2:
            for user in users:
                USERS_TO_REWARD[user] += 1
                f2.writelines([f"{user_id} 0\n"])
    LOGGER.info(f"get a product for user {user_id}")

    url = f"http://localhost:{port}/recommend/"

    data = {
        "use_llm": False,
        "user_id": user_id,
    }

    if random.random() < 0.9:
        data["use_llm"] = True

    response = session.post(url, json=data, timeout=10)

    LOGGER.info(f"Status Code {response.status_code} JSON responce {response.json()} ")

    items = [
        Item(
            item["item_id"],
            item["img_url"],
            item["url"],
            ItemInfo(item["cat2"], item["title"], item["price"]),
        )
        for item in response.json()["recommendations"]
    ]
    USER_RECOMMEND[user_id] = items[1:]
    return items[0]


def update_interactions(path: str):
    """ "
    Update interactions

    :param path: path to interactions
    """
    LOGGER.info(f"interactions by path {path}")

    url = f"http://localhost:{port}/update/"

    data = {
        "interactions_path": path,
    }

    response = session.post(url, json=data, timeout=10)

    LOGGER.info(f"Status Code {response.status_code} JSON responce {response.json()} ")
    return response.json()["rewarded_users"]
