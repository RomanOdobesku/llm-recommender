"""
Class for a hierarchical bandit-based recommendation system.
"""

import json
import os
import pickle
import random
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from mab2rec import BanditRecommender, LearningPolicy

from src.logger import LOGGER


class RecommenderConfig:
    """
    Configuration class for the Recommender system.

    Attributes:
        top_k (int): Number of top recommendations to consider.
        reward_interactions (int): Number of reward interactions to take into account.
        categories_n (int): Number of categories to use in recommendations.
        use_all_categories (bool): Whether to use all categories or not.
    """

    def __init__(
        self,
        top_k: int,
        reward_interactions: int,
        categories_n: int,
        use_all_categories: bool = False,
    ) -> None:
        """
        Initializes the configuration for the Recommender system.

        Args:
            top_k (int): Number of top recommendations to consider.
            reward_interactions (int): Number of reward interactions to take into account.
            categories_n (int): Number of categories to use in recommendations.
            use_all_categories (bool, optional): Whether to use all categories or not. Defaults to False.
        """
        self.top_k = top_k
        self.reward_interactions = reward_interactions
        self.categories_n = categories_n
        self.use_all_categories = use_all_categories

    def __str__(self) -> str:
        """
        Provides a user-friendly string representation of the configuration.

        Returns:
            str: A concise summary of the configuration.
        """
        return (
            f"RecommenderConfig: {self.top_k} recommendations, "
            f"{self.reward_interactions} interactions, "
            f"{self.categories_n} categories, "
            f"using all categories: {self.use_all_categories}"
        )

    def __repr__(self) -> str:
        """
        Provides a detailed and unambiguous string representation of the configuration.

        Returns:
            str: A full detailed representation for debugging.
        """
        return (
            f"RecommenderConfig(top_k={self.top_k}, "
            f"reward_interactions={self.reward_interactions}, "
            f"categories_n={self.categories_n}, "
            f"use_all_categories={self.use_all_categories})"
        )


class Recommender:
    """
    A recommender system using a hierarchical bandit-based approach.
    Inherits from BanditRecommender and extends its functionality.
    """

    def __init__(
        self,
        items_data_path: str,
        interactions_data_path: str,
        predicted_categories_path: str,
        config: RecommenderConfig,
    ) -> None:
        """
        Initialize the recommender with item and interaction data,
        and inherit from BanditRecommender.

        :param items_data_path: Path to the CSV file containing item data.
        :param interactions_data_path: Path to the CSV file containing interaction data.
        :param config: Configuration object for recommender settings.
        """
        self.items_df: pd.DataFrame = self.load_data(items_data_path)
        self.interactions_df: pd.DataFrame = self.load_data(interactions_data_path)

        items_mapping = self.items_df.set_index("item_id")["cat2"]
        self.interactions_df["cat2"] = self.interactions_df["item_id"].map(
            items_mapping
        )

        self.predicted_categories_map: Dict[tuple, str] = (
            self.extract_predicted_categories(predicted_categories_path)
        )
        self.hierarchical_categories: Dict[str, Dict[str, List[str]]] = (
            self.extract_hierarchical_categories()
        )
        self.available_categories: List[str] = self.extract_available_categories()
        self.reward_interactions: int = config.reward_interactions
        self.categories_n = config.categories_n
        self.use_all_categories = config.use_all_categories
        self.rec: BanditRecommender = BanditRecommender(
            LearningPolicy.ThompsonSampling(), top_k=config.top_k
        )

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from a CSV file into a pandas DataFrame.

        :param filename: Path to the CSV file.
        :return: A pandas DataFrame containing the loaded data.
        :raises FileNotFoundError: If the file does not exist.
        """
        try:
            df = pd.read_csv(filename, sep=";")
            LOGGER.info(f"Dataset loaded from {filename}")
            return df
        except FileNotFoundError:
            LOGGER.info(f"Error: File {filename} not found.")
            return pd.DataFrame()

    def get_reward_users(
        self, old_users_statistics: pd.Series, new_users_statistics: pd.Series
    ) -> List[int]:
        """
        Compare old and new user statistics to find users whose interaction counts have changed,
        considering a normalization factor of self.reward_interactions.

        :return: A list of users who will get reward.
        """
        old_stats_df = old_users_statistics.rename("old_count").reset_index()
        new_stats_df = new_users_statistics.rename("new_count").reset_index()
        comparison_df = pd.merge(old_stats_df, new_stats_df, on="user_id", how="outer")
        comparison_df.fillna(0, inplace=True)
        comparison_df["old_count"] = comparison_df["old_count"].astype(int)
        comparison_df["new_count"] = comparison_df["new_count"].astype(int)
        comparison_df["old_normalized"] = (
            comparison_df["old_count"] // self.reward_interactions
        )
        comparison_df["new_normalized"] = (
            comparison_df["new_count"] // self.reward_interactions
        )
        changed_users = comparison_df[
            comparison_df["new_normalized"] > comparison_df["old_normalized"]
        ]
        return changed_users["user_id"].tolist()

    def extract_predicted_categories(
        self, predicted_categories_path: str
    ) -> Dict[tuple, str]:
        """
        Extract predicted categories.

        :return: A dictionary representing the mapping of 2 user's categories to predicted one.
        """
        predicted_categories_map = {}

        with open(predicted_categories_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        for prediction in predictions:
            predicted_categories_map[(prediction[0], prediction[1])] = prediction[3]

        return predicted_categories_map

    def extract_hierarchical_categories(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract hierarchical categories from the items DataFrame.

        :return: A dictionary representing the hierarchical category structure.
        """

        hierarchical_categories: Dict[str, Dict[str, List[str]]] = {}
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

    def extract_available_categories(self) -> List[str]:
        """
        Extract as list all second-level categories available within
        the hierarchical categories structure.

        :return: A list of second-level categories available in the hierarchical structure.
        """
        available_categories: List[str] = [
            cat2
            for _, cat2_dict in self.hierarchical_categories.items()
            for cat2 in cat2_dict
        ]
        return available_categories

    def get_random_cat(self, n: int = 1) -> List[str]:
        """
        Randomly select `n` second-level categories.

        :param n: Number of categories to select.
        :return: A list of randomly selected second-level categories.
        """
        if n > len(self.available_categories):
            LOGGER.error(
                f"Requested number of categories (n={n}) exceeds \
                available categories ({len(self.available_categories)}). \
                Adjusting to available maximum."
            )
            n = min(n, len(self.available_categories))
        return random.sample(self.available_categories, n)

    def get_last_liked_categories(self, user_id: int, n: int = 2) -> List[str]:
        """
        Retrieve the last `n` liked categories for a given user.

        :param user_id: The ID of the user.
        :param n: Number of last liked categories to retrieve.
        :return: A list of the last `n` liked categories,
        supplemented with random categories if needed.
        """
        user_interactions_df = self.interactions_df[
            (self.interactions_df["user_id"] == user_id)
            & (self.interactions_df["interaction"] == 1)
        ]

        user_interactions_df = user_interactions_df.sort_values(
            by="time", ascending=False
        )

        last_liked_categories = (
            user_interactions_df["cat2"].dropna().unique().tolist()[:n]
        )

        LOGGER.info(f"last_liked_categories: {last_liked_categories}")

        if len(last_liked_categories) < n:
            additional_categories = self.get_random_cat(n - len(last_liked_categories))
            last_liked_categories.extend(additional_categories)

        LOGGER.info(f"Final User's liked categories: {last_liked_categories}")
        return last_liked_categories

    def get_llm_selected_cat(self, user_id: int, n: int = 1) -> List[str]:
        """
        Select `n` categories using an LLM generated predictions.

        :param user_id: The ID of the user.
        :param n: Number of categories to select.
        :return: A list of selected second-level categories.
        """
        LOGGER.info(f"Getting last liked categories for: {user_id}")
        user_cat21, user_cat22 = self.get_last_liked_categories(user_id, n=2)[:2]
        predicted_cats = self.predicted_categories_map.get((user_cat21, user_cat22))

        if predicted_cats:
            selected_cats = predicted_cats[:n]
        else:
            selected_cats = []

        LOGGER.info(f"Categories selected by LLM: {selected_cats}")

        valid_cats = []
        for cat in selected_cats:
            if cat in self.available_categories:
                valid_cats.append(cat)
            else:
                LOGGER.error(f"Predicted category {cat} not in available categories")

        if len(valid_cats) < n:
            additional_cats = self.get_random_cat(n - len(valid_cats))
            valid_cats.extend(additional_cats)

        return list(set(valid_cats))

    def filter_items_by_cats(self, categories: List[str]) -> pd.DataFrame:
        """
        Filter items by a list of second-level categories.

        :param categories: List of categories to filter items by.
        :return: A pandas DataFrame containing filtered items.
        """
        LOGGER.info(f"Filtered by categories: {categories}")
        filtered_df = self.items_df[self.items_df["cat2"].isin(categories)]
        return filtered_df

    def fit(self) -> None:
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

    def partial_fit(self, interactions_data_path: str) -> Optional[List[int]]:
        """
        Perform incremental training using new interaction data.
        Only considers interactions that occurred after the latest time
        in the current interactions data.

        :param interactions_data_path: Path to the CSV file containing interaction data.
        :return: None
        """
        new_interactions_df = self.load_data(interactions_data_path)

        if new_interactions_df.empty:
            LOGGER.info("No new interactions data.")
            return None

        old_users_statistics = self.interactions_df.groupby("user_id").size()
        new_users_statistics = new_interactions_df.groupby("user_id").size()

        LOGGER.info(f"Reward for old_users_statistics {old_users_statistics}")
        LOGGER.info(f"Reward for new_users_statistics {new_users_statistics}")

        new_interactions_df["time"] = pd.to_datetime(new_interactions_df["time"])

        if not self.interactions_df.empty and "time" in self.interactions_df.columns:
            self.interactions_df["time"] = pd.to_datetime(self.interactions_df["time"])
            latest_time = self.interactions_df["time"].max()
            new_interactions_df = new_interactions_df[
                new_interactions_df["time"] > latest_time
            ]

        if new_interactions_df.empty:
            LOGGER.info("No new interactions after the latest time.")
            return None

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

        reward_users = self.get_reward_users(old_users_statistics, new_users_statistics)
        LOGGER.info(f"Rewarded users: {reward_users}")
        return reward_users

    def save(self) -> None:
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

    def predict(self, user_id: int, use_llm: bool = False) -> Optional[pd.DataFrame]:
        """
        Make recommendations based on the specified category selection method.

        :param use_llm: Whether to use an LLM for category selection.
        :return: List of recommended items, or None if no items found.
        """
        if use_llm:
            categories = self.get_llm_selected_cat(user_id, n=self.categories_n)
        else:
            categories = self.get_random_cat(n=self.categories_n)

        if not self.use_all_categories:
            filtered_items = self.filter_items_by_cats(categories)
            filtered_arms = filtered_items["item_id"].tolist()
        else:
            filtered_arms = self.items_df["item_id"].tolist()

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

        LOGGER.info(f"Recommended items: {recommended_items.item_id.tolist()}")
        return recommended_items
