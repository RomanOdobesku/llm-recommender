"""
Class for a hierarchical bandit-based recommendation system.
"""

import random

import pandas as pd
from mab2rec import BanditRecommender, LearningPolicy

from src.logger import LOGGER  # pylint: disable=import-error


class Recommender:
    """
    A recommender system using a hierarchical bandit-based approach.
    Inherits from BanditRecommender and extends its functionality.
    """

    def __init__(self, items_data_path, interactions_data_path, top_k=1):
        """
        Initialize the recommender with item and interaction data,
        and inherit from BanditRecommender.

        :param items_data_path: Path to the CSV file containing item data.
        :param interactions_data_path: Path to the CSV file containing interaction data.
        :param top_k: Number of top recommendations to return.
        """
        self.items_df = self.load_data(items_data_path)
        self.interactions_df = self.load_data(interactions_data_path)
        self.hierarchical_categories = self.extract_categories()
        self.rec = BanditRecommender(LearningPolicy.ThompsonSampling(), top_k=top_k)

    def load_data(self, filename):
        """
        Load data from a CSV file into a pandas DataFrame.

        :param filename: Path to the CSV file.
        :return: A pandas DataFrame containing the loaded data.
        :raises FileNotFoundError: If the file does not exist.
        """
        try:
            df = pd.read_csv(filename)
            LOGGER.info(f"Dataset loaded from {filename}")
            return df
        except FileNotFoundError:
            LOGGER.info(f"Error: File {filename} not found.")
            return pd.DataFrame()

    def extract_categories(self):
        """
        Extract hierarchical categories from the items DataFrame.

        :return: A dictionary representing the hierarchical category structure.
        """
        hierarchical_categories = {}
        for _, row in self.items_df.iterrows():
            cat1, cat2, cat3 = row["cat1"], row["cat2"], row["cat3"]
            hierarchical_categories.setdefault(cat1, {}).setdefault(cat2, []).append(
                cat3
            )

        for cat1, cat2_dict in hierarchical_categories.items():
            for cat2 in cat2_dict:
                cat2_dict[cat2] = list(set(cat2_dict[cat2]))

        LOGGER.info(hierarchical_categories)
        return hierarchical_categories

    def get_random_cat2(self):
        """
        Randomly select a second-level category.

        :return: A randomly selected second-level category.
        """
        cat2_list = [
            cat2
            for cat1, cat2_dict in self.hierarchical_categories.items()
            for cat2 in cat2_dict
        ]
        return random.choice(cat2_list)

    def get_llm_selected_cat2(self):
        """
        Select a category using an LLM (placeholder).

        :return: A selected second-level category.
        """
        return self.get_random_cat2()

    def filter_items_by_cat2(self, category):
        """
        Filter items by a second-level category.

        :param category: The category to filter items by.
        :return: A pandas DataFrame containing filtered items.
        """
        filtered_df = self.items_df[self.items_df["cat2"] == category]
        return filtered_df

    def fit(self):
        """
        Train the recommender using initial interaction data.

        :return: None
        """
        if self.interactions_df.empty:
            LOGGER.info("No interactions data to fit.")
            return

        decisions = self.interactions_df["item_id"].astype(str).tolist()
        rewards = self.interactions_df["interaction"].tolist()

        self.rec.fit(decisions=decisions, rewards=rewards)

    def predict(self, use_llm=False):
        """
        Make recommendations based on the specified category selection method.

        :param use_llm: Whether to use an LLM for category selection.
        :return: List of recommended items, or None if no items found.
        """
        cat2 = self.get_llm_selected_cat2() if use_llm else self.get_random_cat2()
        filtered_items = self.filter_items_by_cat2(cat2)
        filtered_arms = filtered_items["item_id"].astype(str).tolist()
        if not filtered_arms:
            LOGGER.info("No items found for the selected category.")
            return None
        self.rec.set_arms(filtered_arms)
        recommendations = self.rec.recommend()
        LOGGER.info(f"Number of used arms: {len(self.rec.mab.arms)}")
        return recommendations
