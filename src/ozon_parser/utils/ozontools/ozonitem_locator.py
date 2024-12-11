"""
Class with CSS_SELECTORs for
OZON website parsing
"""

from dataclasses import dataclass
from selenium.webdriver.common.by import By


@dataclass
class OzonItemLocator:
    """Selectors for Ozon Cats"""

    # btn "Посмотреть все" в подкатегориях
    MORE_BTN = (
        By.CSS_SELECTOR,
        "button.tsBody500Medium.ga120-a",
    )
    SUBCATEGORIES = (
        By.CSS_SELECTOR,
        "a.d0r_10.tsBody500Medium",
    )
    SUBSUBCATEGORIES = (
        By.CSS_SELECTOR,
        "a.d0r_10.tsBody500Medium",
    )

    ITEMS = (
        By.CSS_SELECTOR,
        ".j6s_23",
    )

    TITLE = (
        By.CSS_SELECTOR,
        ".tsBody500Medium",
    )

    PRICE = (
        By.CSS_SELECTOR,
        ".c3022-a1",
    )

    INIT_PRICE = (
        By.CSS_SELECTOR,
        ".c3022-a1.tsBodyControl400Small.c3022-b.c3022-a6",
    )

    DISCOUNT = (
        By.CSS_SELECTOR,
        ".tsBodyControl400Small.c3022-a6.c3022-b4",
    )

    ITEM_URL = (
        By.CSS_SELECTOR,
        "a",
    )

    ITEM_RATING_DIV = (
        By.CSS_SELECTOR,
        "div.j5q_23.j7q_23",
    )

    ITEM_RATING_SPAN = (
        By.CSS_SELECTOR,
        "span[style='color:rgba(7, 7, 7, 1);']",
    )

    ITEM_REVIEWS = (
        By.CSS_SELECTOR,
        "span[style='color:rgba(0, 26, 52, 0.6);']",
    )

    IMG_URL = (
        By.CSS_SELECTOR,
        "img",
    )
