"""
Parser Running Script
"""

import configparser
from pathlib import Path
from src.logger import LOGGER

from .utils.ozontools.ozonitem_link_extractor import OzonItemLinkExtractor
from .utils.ozontools.ozonitem_page_parser import OzonItemPageParser
from .utils.datatools.concat_csvs import concat_csvs_in_dir

if __name__ == "__main__":

    Path("./data/").mkdir(parents=True, exist_ok=True)

    try:
        LOGGER.info("Reading CONFIG")

        config = configparser.ConfigParser()
        config.read("./src/ozon_parser/settings.ini")

        LOGGER.info("CONFIG read")

    except Exception as e:
        LOGGER.error(e)
        raise

    try:

        LOGGER.info("Running LinkExtractor...")

        OzonItemLinkExtractor(
            csv_path=config["OZON"]["ozon_links_to_check_path"],
            user_agent=config["OZON"]["USER_AGENT"],
        ).parse()

        LOGGER.info("Links Extracted")

        # By default we want ot concat all csvs
        # into one file
        LOGGER.info("Concating Links...")
        concat_csvs_in_dir(
            dir_path=config["OZON"]["ozon_links_to_check_path"],
            output_path=config["OZON"]["ozon_links_csv_path"],
        )
        LOGGER.info("Links concatenated!")

    except Exception as e:
        LOGGER.error(e)
        raise

    try:

        LOGGER.info("Running PageParser...")

        OzonItemPageParser(
            csv_path=config["OZON"]["ozon_items_path"],
            user_agent=config["OZON"]["USER_AGENT"],
        ).parse()

        LOGGER.info("All Extracted")

        # By default we want ot concat all csvs
        # into one file
        LOGGER.info("Concating Links...")
        concat_csvs_in_dir(
            dir_path=config["OZON"]["ozon_items_path"],
            output_path=config["OZON"]["ozon_items_csv_path"],
        )
        LOGGER.info("Links concatenated!")

    except Exception as e:
        LOGGER.error(e)
        raise
