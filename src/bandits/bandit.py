"""
Class for a hierarchical bandit-based recommendation system.
"""

import os
import pickle
import random
from datetime import datetime

import pandas as pd
from mab2rec import BanditRecommender, LearningPolicy

from src.logger import LOGGER


class Recommender:
    """
    A recommender system using a hierarchical bandit-based approach.
    Inherits from BanditRecommender and extends its functionality.
    """

    def __init__(
        self,
        items_data_path,
        interactions_data_path,
        predicted_categories_path,
        top_k=1,
    ):
        """
        Initialize the recommender with item and interaction data,
        and inherit from BanditRecommender.

        :param items_data_path: Path to the CSV file containing item data.
        :param interactions_data_path: Path to the CSV file containing interaction data.
        :param top_k: Number of top recommendations to return.
        """
        self.items_df = self.load_data(items_data_path)
        self.interactions_df = self.load_data(interactions_data_path)
        self.predicted_categories_map = self.extract_predicted_categories(
            predicted_categories_path
        )
        self.hierarchical_categories = self.extract_hierarchical_categories()
        self.available_categories = self.extract_available_categories()
        self.rec = BanditRecommender(LearningPolicy.ThompsonSampling(), top_k=top_k)

    def load_data(self, filename):
        """
        Load data from a CSV file into a pandas DataFrame.

        :param filename: Path to the CSV file.
        :return: A pandas DataFrame containing the loaded data.
        :raises FileNotFoundError: If the file does not exist.
        """
        try:
            df = pd.read_csv(filename, sep=",")
            LOGGER.info(f"Dataset loaded from {filename}")
            return df
        except FileNotFoundError:
            LOGGER.info(f"Error: File {filename} not found.")
            return pd.DataFrame()

    def extract_predicted_categories(self, predicted_categories_path):
        """
        Extract predicted categories.
        :return: None
        """
        df = self.load_data(predicted_categories_path)

        predicted_categories_map = {
            (row["user_category_1"], row["user_category_2"]): row["predicted_category"]
            for _, row in df.iterrows()
        }
        return predicted_categories_map

    def extract_hierarchical_categories(self):
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

    def extract_available_categories(self):
        """
        Extract as list all second-level categories available within the hierarchical categories structure.

        :return: A list of second-level categories available in the hierarchical structure.
        """
        available_categories = [
            cat2
            for _, cat2_dict in self.hierarchical_categories.items()
            for cat2 in cat2_dict
        ]
        return available_categories

    def get_random_cat(self, n=1):
        """
        Randomly select `n` second-level categories.

        :param n: Number of categories to select.
        :return: A list of randomly selected second-level categories.
        """
        if n > len(self.available_categories):
            LOGGER.error(
                f"Requested number of categories (n={n}) exceeds available categories ({len(self.available_categories)}). Adjusting to available maximum."
            )
            n = min(n, len(self.available_categories))
        return random.sample(self.available_categories, n)

    def get_last_liked_categories(self, user_id, n=2):
        """
        Retrieve the last `n` liked categories for a given user.

        :param user_id: The ID of the user.
        :param n: Number of last liked categories to retrieve.
        :return: A list of the last `n` liked categories, supplemented with random categories if needed.
        """
        user_interactions_df = self.interactions_df[
            (self.interactions_df["user_id"] == user_id)
            & (self.interactions_df["interaction"] == 1)
        ]

        user_interactions_df = user_interactions_df.sort_values(
            by="time", ascending=False
        )

        items_mapping = self.items_df.set_index("item_id")["cat2"]
        user_interactions_df["cat2"] = user_interactions_df["item_id"].map(
            items_mapping
        )

        last_liked_categories = user_interactions_df["cat2"].dropna().head(n).tolist()

        if len(last_liked_categories) < n:
            additional_categories = self.get_random_cat(n - len(last_liked_categories))
            last_liked_categories.extend(additional_categories)

        return last_liked_categories

    def get_llm_selected_cat(self, user_id):
        """
        Select a category using an LLM generated predictions.

        :return: A selected second-level category.
        """
        C1, C2 = self.get_last_liked_categories(user_id)
        CL = self.predicted_categories_map.get((C1, C2), self.get_random_cat(n=1)[0])
        return CL

    def filter_items_by_cat(self, category):
        """
        Filter items by a second-level category.

        :param category: The category to filter items by.
        :return: A pandas DataFrame containing filtered items.
        """
        LOGGER.info(f"Filtered by {category}")
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
        self.save()

        LOGGER.info(f"Fit completed with {len(decisions)} interactions.")

    def partial_fit(self, interactions_data_path):
        """
        Perform incremental training using new interaction data.
        Only considers interactions that occurred after the latest time
        in the current interactions data.

        :param interactions_data_path: Path to the CSV file containing interaction data.
        :return: None
        """
        new_interactions_df = self.load_data(interactions_data_path)
        if new_interactions_df.empty:
            LOGGER.info("No new interactions data to fit.")
            return

        new_interactions_df["time"] = pd.to_datetime(new_interactions_df["time"])

        if not self.interactions_df.empty and "time" in self.interactions_df.columns:
            self.interactions_df["time"] = pd.to_datetime(self.interactions_df["time"])
            latest_time = self.interactions_df["time"].max()
            new_interactions_df = new_interactions_df[
                new_interactions_df["time"] > latest_time
            ]

        if new_interactions_df.empty:
            LOGGER.info("No new interactions after the latest time.")
            return

        self.interactions_df = pd.concat(
            [self.interactions_df, new_interactions_df], ignore_index=True
        )
        decisions_new = new_interactions_df["item_id"].astype(str).tolist()
        rewards_new = new_interactions_df["interaction"].tolist()
        self.rec.partial_fit(decisions_new, rewards_new)
        self.save()

        LOGGER.info(
            f"Partial fit completed with {len(decisions_new)} new interactions."
        )

    def save(self):
        """
        Save the trained model to the models directory with a timestamped filename.

        :return: None
        """
        models_dir = "src/bandits/models"
        os.makedirs(models_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bandit_{timestamp}.pkl"
        filepath = os.path.join(models_dir, filename)

        try:
            with open(filepath, "wb") as fp:
                pickle.dump(self, fp)
            LOGGER.info(f"Model saved to {filepath}")
        except (OSError, IOError, pickle.PickleError) as e:
            LOGGER.error(f"Failed to save the model: {e}")

    def predict(self, user_id, use_llm=False):
        """
        Make recommendations based on the specified category selection method.

        :param use_llm: Whether to use an LLM for category selection.
        :return: List of recommended items, or None if no items found.
        """
        cat = (
            self.get_llm_selected_cat(user_id)
            if use_llm
            else self.get_random_cat(n=1)[0]
        )
        filtered_items = self.filter_items_by_cat(cat)
        filtered_arms = filtered_items["item_id"].astype(str).tolist()

        if not filtered_arms:
            LOGGER.info("No items found for the selected category.")
            return None

        self.rec.set_arms(filtered_arms)
        LOGGER.info(f"Number of arms: {len(self.rec.mab.arms)}")
        recommendations = self.rec.recommend()

        if not recommendations:
            LOGGER.info("No recommendations generated.")
            return None

        recommended_ids = [int(item) for item in recommendations]
        recommended_items = self.items_df[
            self.items_df["item_id"].isin(recommended_ids)
        ]

        LOGGER.info(f"Recommended items: {recommended_items.item_id.unique()}")
        return recommended_items
