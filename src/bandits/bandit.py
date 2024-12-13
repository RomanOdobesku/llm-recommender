"""
Class for a hierarchical bandit-based recommendation system.
"""

import os
import pickle
import random
from datetime import datetime
from typing import Dict, List, Optional

from dataclasses import dataclass

import pandas as pd
from mab2rec import BanditRecommender, LearningPolicy

from src.logger import LOGGER


@dataclass
class RecommenderConfig:
    """
    Configuration class for the Recommender system.
    """

    top_k: int
    reward_interactions: int
    items_data_path: str
    interactions_data_path: str
    predicted_categories_path: str


class Recommender:
    """
    A recommender system using a hierarchical bandit-based approach.
    Inherits from BanditRecommender and extends its functionality.
    """

    def __init__(
        self,
        config: RecommenderConfig,
    ) -> None:
        """
        Initialize the recommender with item and interaction data,
        and inherit from BanditRecommender.

        :param items_data_path: Path to the CSV file containing item data.
        :param interactions_data_path: Path to the CSV file containing interaction data.
        :param config: Configuration object for recommender settings.
        """
        self.items_df: pd.DataFrame = self.load_data(config.items_data_path)
        self.interactions_df: pd.DataFrame = self.load_data(
            config.interactions_data_path
        )

        # items_mapping = self.items_df.set_index("item_id")["cat2"]
        # self.interactions_df["cat2"] = self.interactions_df["item_id"].map(
        #     items_mapping
        # )
        self.interactions_df = self.interactions_df.merge(
            self.items_df[["item_id", "cat2"]], on="item_id", how="left"
        )

        self.predicted_categories_map: Dict[tuple, str] = (
            self.extract_predicted_categories(config.predicted_categories_path)
        )
        self.hierarchical_categories: Dict[str, Dict[str, List[str]]] = (
            self.extract_hierarchical_categories()
        )
        self.available_categories: List[str] = self.extract_available_categories()
        self.reward_interactions: int = (
            config.reward_interactions
        )  # get_reward_interactions()
        self.rec: BanditRecommender = BanditRecommender(
            LearningPolicy.ThompsonSampling(), top_k=config.top_k  # get_top_k()
        )

    def load_data(self, filename: str) -> pd.DataFrame:
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

    def get_reward_users(
        self,
        old_users_statistics: pd.Series,
        new_users_statistics: pd.Series,
    ) -> List[int]:
        """
        Compare old and new user statistics to find users whose interaction counts have changed,
        considering a normalization factor of self.reward_interactions.

        :return: A list of user IDs who will get a reward.
        """

        # Efficiently merge and handle NaN values
        # (avoids creating unnecessary intermediate DataFrames)
        merged_df = (
            pd.merge(
                old_users_statistics.rename("old_count").reset_index(),
                new_users_statistics.rename("new_count").reset_index(),
                on="user_id",
                how="outer",
            )
            .fillna(0)
            .astype(int)
        )

        # Directly calculate normalized counts and identify changed users (more efficient)
        merged_df["old_normalized"] = merged_df["old_count"] // self.reward_interactions
        merged_df["new_normalized"] = merged_df["new_count"] // self.reward_interactions
        reward_users = merged_df.loc[
            merged_df["new_normalized"] > merged_df["old_normalized"], "user_id"
        ].tolist()

        return reward_users

    def extract_predicted_categories(
        self, predicted_categories_path: str
    ) -> Dict[tuple, str]:
        """
        Extract predicted categories.

        :return: A dictionary representing the mapping of 2 user's categories to predicted one.
        """
        df = self.load_data(predicted_categories_path)
        return df.set_index(["user_category_1", "user_category_2"])[
            "predicted_category"
        ].to_dict()

    def extract_hierarchical_categories(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract hierarchical categories from the items DataFrame.

        :return: A dictionary representing the hierarchical category structure.
        """

        try:
            hierarchical_categories = {}
            # Use .groupby for significantly better performance
            for cat1, group in self.items_df.groupby("cat1"):
                cat2_dict = {}
                for _, row in group.iterrows():
                    cat2 = row["cat2"]
                    cat3 = row["cat3"]
                    cat2_dict.setdefault(cat2, []).append(cat3)  # Efficiently appends

                for cat2, cat3_list in cat2_dict.items():
                    cat2_dict[cat2] = sorted(
                        list(set(cat3_list))
                    )  # Sort for consistency
                hierarchical_categories[cat1] = cat2_dict

            LOGGER.info(hierarchical_categories)
            return hierarchical_categories

        except KeyError as e:
            LOGGER.error(f"Error: Missing column(s) in items_df: {e}")
            return {}
        # except Exception as e:
        #     LOGGER.exception(f"An unexpected error occurred: {e}")
        #     return {}

        # hierarchical_categories: Dict[str, Dict[str, List[str]]] = {}
        # for _, row in self.items_df.iterrows():
        #     cat1, cat2, cat3 = row["cat1"], row["cat2"], row["cat3"]
        #     hierarchical_categories.setdefault(cat1, {}).setdefault(cat2, []).append(
        #         cat3
        #     )
        #
        # for cat1, cat2_dict in hierarchical_categories.items():
        #     for cat2 in cat2_dict:
        #         cat2_dict[cat2] = list(set(cat2_dict[cat2]))
        #
        # LOGGER.info(hierarchical_categories)
        # return hierarchical_categories

    def extract_available_categories(self) -> List[str]:
        """
        Extract as list all second-level categories available within
        the hierarchical categories structure.

        :return: A list of second-level categories available in the hierarchical structure.
        """
        available_categories = list(
            {
                cat2
                for cat2_dict in self.hierarchical_categories.values()
                for cat2 in cat2_dict
            }
        )
        return available_categories

    def get_random_cat(self, n: int = 1) -> List[str]:
        """
        Randomly select `n` second-level categories.

        :param n: Number of categories to select.
        :return: A list of randomly selected second-level categories.
        """
        if n <= 0:
            return []
        if n > len(self.available_categories):
            LOGGER.error(
                f"Requested number of categories (n={n}) exceeds \
                available categories ({len(self.available_categories)}). \
                Adjusting to available maximum."
            )
            return self.available_categories

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
        ].sort_values(by=["time"], ascending=False)

        # md need unique
        last_liked_categories = (
            user_interactions_df["cat2"].dropna().unique()[:n].tolist()
        )

        num_additional = max(0, n - len(last_liked_categories))
        additional_categories = self.get_random_cat(num_additional)
        last_liked_categories.extend(additional_categories)

        return last_liked_categories

    def get_llm_selected_cat(self, user_id: int) -> str:
        """
        Select a category using an LLM generated predictions.

        :return: A selected second-level category.
        """
        user_cat21, user_cat22 = self.get_last_liked_categories(user_id)
        predicted_cat = self.predicted_categories_map.get(
            (user_cat21, user_cat22), self.get_random_cat(n=1)[0]
        )
        return predicted_cat

    def filter_items_by_cat(self, category: str) -> pd.DataFrame:
        """
        Filter items by a second-level category.

        :param category: The category to filter items by.
        :return: A pandas DataFrame containing filtered items.
        """
        LOGGER.info(f"Filtered by {category}")
        filtered_df = self.items_df[self.items_df["cat2"] == category]
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
