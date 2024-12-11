"""
Module with abstract class BaseParser
Depends on Undetected Chromium
"""

from abc import ABC
from typing import Optional
from dataclasses import dataclass

import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options


@dataclass
class BaseParser(ABC):
    """
    Abstract class BaseParser
    Depends on Undetected Chromium
    """

    csv_path: str
    user_agent: Optional[str]
    driver: uc.Chrome = uc.Chrome()

    def __init__(
        self,
        csv_path: str,
        user_agent: Optional[str] = None,
    ) -> None:
        """
        Init BaseParser

        Args:
            csv_path (str): the path to the folder where the links should be placed
            user_agent (Optional[str]): USER_AGENT

        Returns:
            (None):
        """

        self.csv_path = csv_path
        self.user_agent = user_agent

        # Default options for driver
        options = Options()

        if self.user_agent is not None:
            # add the custom User Agent to Chrome Options
            options.add_argument(f"--user-agent={self.user_agent}")

        self.driver = uc.Chrome(options=options)

    def set_up_driver(self, options: Options):
        """
        Class method to setup driver
        """
        self.driver = uc.Chrome(options=options)

    def parse(self):
        """
        Class method to start parsing data
        """
