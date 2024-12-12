"""
Class for extracting links of categories
and sub-\\subsub-categories from OZON
"""

from pathlib import Path
import time
from typing import Iterator, Tuple
from dataclasses import dataclass

from selenium.common.exceptions import NoSuchElementException

from src.logger import LOGGER

from .baseparser import BaseParser
from .ozonitem_locator import OzonItemLocator
from .ozon_categories import OZON_CATEGORIES


@dataclass
class OzonItemLinkExtractor(BaseParser):
    """
    Current project needs data to work with,
    so this module was made to parse data from
    OZON website
    """

    def __parse_page_subcategory(
        self,
        parse_url: str,
    ) -> Iterator[Tuple[str, str]]:
        """
        Class method for parsing sub_category
        (Extracting sub sub category urls)

        Args:
            parseURL (str): url to parse

        Returns:
            (Iterator[tuple[list[str], list[str]]]):
        """

        try:
            self.driver.switch_to.new_window("tab")
            self.driver.get(parse_url)

            self.driver.implicitly_wait(5)

            try:
                time.sleep(0.25)
                self.driver.find_element(*OzonItemLocator.MORE_BTN).click()
                time.sleep(0.25)
            except NoSuchElementException:
                LOGGER.exception(
                    "NoSuchElementException: no MORE Button on the page",
                )

            subcategories = self.driver.find_elements(*OzonItemLocator.SUBSUBCATEGORIES)
            sub_cat_names = []
            sub_cat_urls = []
            for subcat in subcategories[1:]:

                sub_cat_names.append(f"{subcat.text}")
                sub_cat_urls.append(f"{subcat.get_attribute('href')}")

            self.driver.close()

            return zip(sub_cat_names, sub_cat_urls)

        except NoSuchElementException:
            LOGGER.exception(
                "NoSuchElementException: subsubcategories not found",
            )
            sub_cat_names = []
            sub_cat_urls = []
            return zip(sub_cat_names, sub_cat_urls)

        except Exception as e:
            LOGGER.error(e)
            raise

    def __parse_page_category(self, parse_url: str, csv_path: str) -> None:
        """
        Class method for parsing category
        (Extracting sub categories urls)

        Args:
            parseURL (str): url to parse

        Returns:
            (Iterator[tuple[list[str], list[str]]]):
        """

        try:

            self.driver.get(parse_url)

            self.driver.implicitly_wait(5)

            # Store the ID of the original window
            original_window = self.driver.current_window_handle

            try:
                self.driver.find_element(*OzonItemLocator.MORE_BTN).click()
                time.sleep(0.25)

            except NoSuchElementException:
                LOGGER.exception(
                    "NoSuchElementException: no MORE Button on the page",
                )

            subcategories = self.driver.find_elements(*OzonItemLocator.SUBCATEGORIES)

            cat_name = subcategories[0].text
            cat_url = subcategories[0].get_attribute("href")

            with open(csv_path, mode="a", encoding="utf-8") as file:

                for subcat in subcategories[1:]:

                    sub_cat_name = subcat.text
                    sub_cat_url = subcat.get_attribute("href")

                    for (
                        sub_sub_cat_name,
                        sub_sub_cat_url,
                    ) in self.__parse_page_subcategory(f"{sub_cat_url}"):
                        file.write(
                            f"{cat_name};"
                            + f"{cat_url};"
                            + f"{sub_cat_name};"
                            + f"{sub_cat_url};"
                            + f"{sub_sub_cat_name};"
                            + f"{sub_sub_cat_url}"
                            + "\n"
                        )
                        self.driver.switch_to.window(original_window)

        except NoSuchElementException:
            LOGGER.exception(
                "NoSuchElementException: no subcategories on the page",
            )
        except IndexError:
            LOGGER.exception(
                "IndexError: subcategories ended on this page page",
            )
        except Exception as e:
            LOGGER.error(e)
            raise

    def parse(self):
        """
        Class method to extract links from categories

        Args:
            categories (dict): dict where keys are names of categories to parse
                and values are URLs of these categories

        Returns:
            (None):
        """

        try:

            Path(self.csv_path).mkdir(parents=True, exist_ok=True)

            for category, url in OZON_CATEGORIES.items():

                LOGGER.info(f"Extracting links for category: {category}")

                csv_path = self.csv_path + f"links_to_check_{category}.csv"

                # Init file to write urls
                with open(csv_path, mode="w", encoding="utf-8") as file:
                    file.write(
                        "cat_name;cat_url;"
                        + "sub_cat_name;sub_cat_url;"
                        + "sub_sub_cat_name;sub_sub_cat_url"
                        + "\n"
                    )

                self.__parse_page_category(url, csv_path)

                LOGGER.info("Links EXTRACTED")

            self.driver.close()

        except Exception as e:
            LOGGER.error(e)
            raise
