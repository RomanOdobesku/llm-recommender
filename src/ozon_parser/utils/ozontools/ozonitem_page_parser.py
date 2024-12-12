"""
Current project needs data to work with,
so this module was made to parse data from
OZON website
"""

from pathlib import Path
import time
from typing import Optional
from dataclasses import dataclass

import pandas as pd

from selenium.common.exceptions import (
    NoSuchElementException,
    NoSuchAttributeException,
)
from selenium.webdriver.remote.webelement import WebElement

from src.logger import LOGGER

from .baseparser import BaseParser
from .ozonitem_locator import OzonItemLocator
from .ozonitem import OzonItem


@dataclass
class OzonItemPageParser(BaseParser):
    """
    Class for parsing pages of OZON website
    (Extracting items properties)
    """

    def __extract_item(self, element: WebElement) -> OzonItem:
        """
        Method to process WebElement for ozon item
        Args:
            element (WebElement): element to extract info

        Returns:
            (OzonItem): OzonItem
        """
        item = OzonItem()

        try:
            # Get product title
            item["title"] = item.re_title(
                element.find_element(
                    *OzonItemLocator.TITLE,
                ).text,
            )

        except NoSuchElementException:
            LOGGER.exception(
                "NoSuchElementException: Title not found",
            )

        try:
            # Get product price
            item["price"] = item.re_price(
                element.find_element(*OzonItemLocator.PRICE).text,
            )

        except NoSuchElementException:
            LOGGER.exception(
                "NoSuchElementException: Price not found",
            )
            item["price"] = -999.0

        try:
            # Get product init_price
            item["initprice"] = item.re_price(
                element.find_element(
                    *OzonItemLocator.INIT_PRICE,
                ).text,
            )

        except NoSuchElementException:
            LOGGER.exception(
                "NoSuchElementException: InitPrice not found",
            )
            item["initprice"] = -999.0

        try:
            # Get product discount
            item["discount"] = item.re_discount(
                element.find_element(
                    *OzonItemLocator.DISCOUNT,
                ).text,
            )

        except NoSuchElementException:
            LOGGER.exception(
                "NoSuchElementException: Discount not found",
            )
            item["discount"] = -999

        try:
            # Get product url
            item["url"] = element.find_element(
                *OzonItemLocator.ITEM_URL,
            ).get_attribute("href")

        except NoSuchElementException:
            LOGGER.exception(
                "NoSuchElementException: Link not found",
            )
        except NoSuchAttributeException:
            LOGGER.exception(
                "NoSuchAttributeException: Link not found",
            )

        try:
            # Get product rating
            item["rating"] = item.re_rating(
                element.find_element(
                    *OzonItemLocator.ITEM_RATING_DIV,
                )
                .find_element(
                    *OzonItemLocator.ITEM_RATING_SPAN,
                )
                .text
            )

        except NoSuchElementException:
            LOGGER.exception(
                "NoSuchElementException: Rating not found",
            )
            item["rating"] = -999.0

        try:
            # Get product reviews
            item["reviews"] = item.re_reviews(
                element.find_element(
                    *OzonItemLocator.ITEM_REVIEWS,
                ).text
            )

        except NoSuchElementException:
            LOGGER.exception(
                "NoSuchElementException: Reviews not found",
            )
            item["reviews"] = -999

        try:
            # Get product image URL
            item["imgUrl"] = element.find_element(
                *OzonItemLocator.IMG_URL,
            ).get_attribute("src")

            # IF YOU NEED TO DOWNLOAD IMGS

            # # Extract image filename from URL
            # img_filename = img_url.split("/")[-1]
            # file.write(img_filename + ",")
            #
            # # Download and save the image with the same filename
            # in the 'images' folder
            # image_data = requests.get(img_url).content
            # with open(f"images/{img_filename}", "wb") as img_file:
            #     img_file.write(image_data)
            # file.write(img_filename + "\n")

        except NoSuchElementException:
            LOGGER.exception(
                "NoSuchElementException: Image not found",
            )
        except NoSuchAttributeException:
            LOGGER.exception(
                "NoSuchAttributeException: Image not found",
            )

        return item

    def __parse_page(
            self,
            parse_url: str,
            csv_path: str,
            csv_cat_info: str
    ) -> None:
        """
        Class method to extract items properties

        Args:
            parseURL (str): url to parse
            csv_path (str): path to csv file to save info
            csvCatInfo (str): string with category, sub_category and sub_sub_category info

        Returns:
            (None):
        """

        try:

            self.driver.get(parse_url)

            # Get all the product elements on the page
            elements = self.driver.find_elements(*OzonItemLocator.ITEMS)

            with open(csv_path, mode="a", encoding="utf-8") as file:
                # Extract data for each product
                for element in elements[:10]:

                    item = self.__extract_item(element)

                    file.write(
                        f"{item['title']};"
                        + f"{item['price']};"
                        + f"{item['initprice']};"
                        + f"{item['discount']};"
                        + f"{item['rating']};"
                        + f"{item['reviews']};"
                        + f"{item['url']};"
                        + f"{item['imgUrl']};"
                        + f"{csv_cat_info}\n"
                    )

        except Exception as e:
            LOGGER.exception(e)
            raise

    def parse(self, ozon_links_csv_path: Optional[str] = None) -> None:
        """
        Method to start extracting items' properties

        Args:
            ozon_links_csv_path (Optional[str]): csv with links to parse
                with following structure:
                    columns = [
                    "cat_name",
                    "cat_url",
                    "sub_cat_name",
                    "sub_cat_url",
                    "sub_sub_cat_name",
                    "sub_sub_cat_url",
                ]
        """

        try:

            Path(self.csv_path).mkdir(parents=True, exist_ok=True)

            if not ozon_links_csv_path:
                ozon_links_csv_path = self.csv_path + "ozon_links.csv"

            df = pd.read_csv(ozon_links_csv_path, sep=";")

            cat_names = df["cat_name"].unique()
            for cat_name in cat_names:

                LOGGER.info(f"Extracting category {cat_name}")

                csv_path = self.csv_path + f"items_{cat_name}.csv"

                # INIT csv FILE
                with open(csv_path, mode="w", encoding="utf-8") as file:
                    file.write(
                        "title;"
                        + "price;"
                        + "initprice;"
                        + "discount;"
                        + "rating;"
                        + "reviews;"
                        + "url;"
                        + "img_url;"
                        + "cat;"
                        + "sub_cat;"
                        + "sub_sub_cat"
                        + "\n"
                    )

                for _, row in df[df["cat_name"] == cat_name].iterrows():

                    LOGGER.info(f"Extracting {row['sub_sub_cat_url']}")

                    self.__parse_page(
                        f"{row["sub_sub_cat_url"]}",
                        csv_path,
                        f'{row["cat_name"]};'
                        + f'{row["sub_cat_name"]};'
                        + f'{row["sub_sub_cat_name"]}',
                    )

                    LOGGER.info(f"All items for category {cat_name} EXTRACTED!")

                time.sleep(0.25)

                LOGGER.info(f"Extracting category {cat_name} DONE!")

            self.driver.close()

        except Exception as e:
            LOGGER.error(e)
            raise
