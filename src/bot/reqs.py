""" file to send requests to bandits """

import os
import requests
from src.logger import LOGGER  # pylint: disable=import-error

port = int(os.environ["FAST_API_PORT"])


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
    """
    A simple version of a bandit to get item

    :param user_id: user_id get id of user to get a recommendations
    :return recommendation for a user
    """

    LOGGER.info(f"get a product for user {user_id}")

    url = f"http://localhost:{port}/recommend/"

    data = {
        "use_llm": False,
    }

    response = requests.post(url, json=data, timeout=10)

    LOGGER.info(f"Status Code {response.status_code} JSON responce {response.json()} ")

    items = [Item(item["item_id"], item["image_link"],
                  item["cat1"], item["description"]) for item in response.json()["recommendations"]]
    return items[0]
