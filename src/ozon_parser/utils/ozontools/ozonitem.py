"""
Class for item from OZON website
"""

import re

BASE_ITEM = {
    "title": "unknown",
    "price": "unknown",
    "initprice": "unknown",
    "discount": "unknown",
    "rating": "unknown",
    "reviews": "unknown",
    "url": "url",
    "imgUrl": "url",
}


class OzonItem(dict):
    """
    Class for item from OZON website
    """

    def __init__(self):
        """
        Init method
        """
        super().__init__()
        self.update(BASE_ITEM)

    @staticmethod
    def re_title(title: str) -> str:
        """
        Method for formating item title

        Args:
            title (str): title to format

        Returns:
            (str): formated title
        """
        title = title.replace(";", "")
        title = title.replace("  ", " ")

        return title

    @staticmethod
    def re_price(price: str) -> float:
        """
        Method for formating item price

        Args:
            price (str): price to format

        Returns:
            (float): formated price
        """

        price = "".join(re.findall(r"\d+", price))
        return float(price)

    @staticmethod
    def re_discount(discount: str) -> int:
        """
        Method for formating item discount

        Args:
            discount (str): discount to format

        Returns:
            (int): formated discount
        """

        discount = "".join(re.findall(r"\d+", discount))
        if discount == "":
            return -999
        return int(discount)

    @staticmethod
    def re_rating(rating: str) -> float:
        """
        Method for formating item rating

        Args:
            rating (str): rating to format

        Returns:
            (float): formated rating
        """

        rating = "".join(re.findall(r"\d.\d", rating))
        if rating == "":
            return -999
        return float(rating)

    @staticmethod
    def re_reviews(reviews: str) -> int:
        """
        Method for formating item reviews

        Args:
            reviews (str): reviews to format

        Returns:
            (int): formated reviews
        """

        reviews = "".join(re.findall(r"\d+", reviews))
        if reviews == "":
            return -999
        return int(reviews)
