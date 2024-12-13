"""
Class for a hierarchical bandit-based recommendation system.
"""

import json
import os
import pickle
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import gc

import numpy as np
import numpy.typing as npt
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
    bandit_top_k: int
    categories_n: int
    use_all_categories: bool = False


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
        self.items_df: pd.DataFrame = self.__load_data(config.items_data_path)
        self.interactions_df: pd.DataFrame = self.__load_data(
            config.interactions_data_path
        )
        self.interactions_df["time"] = pd.to_datetime(self.interactions_df["time"])

        self.items_mapping = self.items_df.set_index("item_id")["cat2"]
        self.interactions_df["cat2"] = self.interactions_df["item_id"].map(
            self.items_mapping
        )

        self.predicted_categories_map: Dict[tuple, str] = (
            self.extract_predicted_categories(config.predicted_categories_path)
        )
        self.hierarchical_categories: Dict[str, Dict[str, List[str]]] = (
            self.extract_hierarchical_categories()
        )
        self.available_categories: List[str] = self.extract_available_categories()
        self.reward_interactions: int = config.reward_interactions
        self.llm_cats_limit = config.categories_n
        self.use_all_categories = config.use_all_categories
        self.top_k = config.top_k
        self.rec: BanditRecommender = BanditRecommender(
            LearningPolicy.ThompsonSampling(), top_k=config.bandit_top_k
        )

        # Set limiter for categories from LLM
        self.categories_n = config.categories_n

        # DataFrame with columns=["cat2", "count"]
        self.top_popular_df = (
            self.interactions_df[self.interactions_df["interaction"] == 1]["items"]
            .value_counts()
            .to_frame()
            .reset_index()
        )

    def __update_top_popular_df(self):
        if not (self.interactions_df.empty and self.items_df.empty):
            self.top_popular_df = (
                self.interactions_df[self.interactions_df["interaction"] == 1]
                .join(
                    self.items_df,
                    on="item_id",
                    how="left",
                )["cat2"]
                .value_counts()
                .to_frame()
                .reset_index()
            )

    def __load_data(self, filename: str) -> pd.DataFrame:
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

    def __get_user_positive_interactions(self, user_id: int):
        """
        Method to get user positive interactions.
        Returns an empty DataFrame if no positive interactions are found.

        Args:
            user_id (int): user ID

        Returns:
            (pd.DataFrame): user positive interactions DataFrame,
            or an empty DataFrame if no interactions are found for the given user_id.
        """
        # Get users positive interactions
        user_interactions_df = self.interactions_df[
            (self.interactions_df["user_id"] == user_id)
            & (self.interactions_df["interaction"] == 1)
        ]

        if user_interactions_df.empty:
            return pd.DataFrame(columns=self.interactions_df.columns)

        return user_interactions_df

    def __get_user_last_liked_categories(
        self,
        user_positive_interactions: pd.DataFrame,
        n_last_liked: int,
    ):

        user_last_liked = (
            user_positive_interactions["cat2"].dropna().values[-n_last_liked:].tolist()
        )
        if not isinstance(user_last_liked, list):
            user_last_liked = []

        # LOGGER.info(f"last_liked_categories: {user_last_liked[::-1]}")

        add_n_cats = max(0, n_last_liked - len(user_last_liked))
        user_last_liked = self.get_random_cat(add_n_cats) + user_last_liked

        # LOGGER.info(f"Final User's liked categories: {user_last_liked[::-1]}")

        return user_last_liked[-n_last_liked:]

    def __get_user_top_popular_categories(
        self,
        user_interactions_df: pd.DataFrame,
        n_top: int,
    ) -> pd.DataFrame:
        """
        Method to get a user's top N popular categories.

        Args:
            user_interactions_df (pd.DataFrame): DataFrame containing user positive
                interactions. Must have a 'cat2' column.
            n_top (int): The number of top categories to retrieve.

        Returns:
            pd.DataFrame: pd.DataFrame(columns=["cat2", "weight"])
              - cat2 --- category names (strings).
              - weight --- corresponding weights (float).
                    Returns an empty tuple if input is invalid.
        """

        if n_top <= 0:
            return pd.DataFrame(data=[[], []], columns=["cat2", "weight"])

        try:
            user_top = (
                user_interactions_df["cat2"].value_counts().to_frame().reset_index()
            )
            user_top.rename(
                {
                    "count": "weight",
                }
            )
            user_top["weight"] = user_top["weights"] / user_interactions_df.shape[0]
            return user_top
        except (
            KeyError,
            AttributeError,
            ZeroDivisionError,
        ):  # Handle cases where 'cat2' column is missing
            return pd.DataFrame(data=[[], []], columns=["cat2", "weight"])

    def get_user_last_liked(
        self,
        user_id: int,
        n_last_liked: int,
    ) -> List[str]:
        """
        Method to get user's last interactions.

        Args:
            user_id (int): The ID of the user.
            n_last_liked (int): The number of last liked categories to retrieve.

        Returns:
            List[str]: A list of the user's last liked categories.
        """

        user_positive_interactions = self.__get_user_positive_interactions(user_id)

        user_last_liked: List[str] = self.__get_user_last_liked_categories(
            user_positive_interactions,
            n_last_liked,
        )

        return user_last_liked

    def get_user_top_popular(
        self,
        user_id: int,
        n_top: int,
    ) -> pd.DataFrame:
        """
        Method to get user's interaction statistics.

        Args:
            user_id: The ID of the user.
            n_top: The number of top popular categories to retrieve.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing two lists:
                - A list of the user's top N popular categories.
                - A list of the corresponding weights for those categories.
        """

        user_positive_interactions = self.__get_user_positive_interactions(user_id)

        user_top_popular = self.__get_user_top_popular_categories(
            user_positive_interactions,
            n_top,
        )

        return user_top_popular

    def get_llm_selected_cat(self, user_id: int, n: int = 1) -> List[str]:
        """
        Select `n` categories using an LLM generated predictions.

        :param user_id: The ID of the user.
        :param n: Limitation for number of categories to select.
        :return: A list of selected second-level categories.
        """

        LOGGER.info(f"Get statistics for: {user_id}")

        user_last_liked = self.get_user_last_liked(
            user_id=user_id,
            n_last_liked=2,
        )
        # user_cat21, user_cat22 = self.get_last_liked_categories(user_id, n=2)[:2]
        predicted_cats = self.predicted_categories_map.get(
            (user_last_liked[-2], user_last_liked[-1])
        )

        if predicted_cats:
            valid_cats = []
            for cat in predicted_cats:
                if cat in self.available_categories:
                    valid_cats.append(cat)
                else:
                    LOGGER.error(
                        f"Predicted category {cat} not in available categories"
                    )

            # if len(valid_cats) < n:
            #     additional_cats = self.get_random_cat(n - len(valid_cats))
            #     valid_cats.extend(additional_cats)

            # Ограничим количество предсказаний от LLM модели
            # Важно, чтобы было до n предсказаний
            valid_cats = random.sample(
                valid_cats,
                min(
                    n,
                    len(valid_cats),
                ),
            )
            LOGGER.info(f"Valid categories selected by LLM: {valid_cats}")

            return list(set(valid_cats))
        else:
            return []

    def filter_items_by_cats(self, categories: List[str]) -> npt.ArrayLike:
        """
        Filter items by a list of second-level categories.

        :param categories: List of categories to filter items by.
        :return: A pandas DataFrame containing filtered items.
        """
        LOGGER.info(f"Filtered by categories: {categories}")
        filtered_items = self.items_df[
            self.items_df["cat2"].isin(categories)
        ].values.tolist()

        return filtered_items

    def fit(self) -> None:
        """
        Train the recommender using initial interaction data.

        :return: None
        """
        if self.interactions_df.empty:
            LOGGER.info("No interactions data to fit.")
            return

        decisions: pd.Series = self.interactions_df["item_id"]
        rewards: pd.Series = self.interactions_df["interaction"]

        self.rec.fit(decisions=decisions, rewards=rewards)
        self.save()

        LOGGER.info(f"Fit completed with {len(decisions)} interactions.")

    def __get_users_statistics(self):
        return self.interactions_df["user_id"].value_counts()

    def partial_fit(self, interactions_data_path: str) -> Optional[List[int]]:
        """
        Perform incremental training using new interaction data.
        Only considers interactions that occurred after the latest time
        in the current interactions data.

        :param interactions_data_path: Path to the CSV file containing interaction data.
        :return: None
        """

        # Load new interactions DataFrame
        new_interactions_df = self.__load_data(interactions_data_path)

        if new_interactions_df.empty:
            LOGGER.info("No new interactions data.")
            return None

        # Check if there is new rows in new_interactions_df
        if new_interactions_df.shape[0] == self.interactions_df.shape[0]:
            LOGGER.info("No new interactions after the latest time.")
            return None

        # Get statistics from current interactions_df
        old_users_statistics = self.__get_users_statistics()
        LOGGER.info(f"Reward for old_users_statistics {old_users_statistics}")
        old_last_date = self.interactions_df["time"].max()

        # Set new interactions_df
        self.interactions_df = new_interactions_df
        gc.collect()

        # Parse column time to datetime format
        self.interactions_df["time"] = pd.to_datetime(self.interactions_df["time"])
        # Update global top popular categories
        self.__update_top_popular_df()

        # Get statistics from new interactions_df
        new_users_statistics = self.interactions_df["user_id"].value_counts()
        LOGGER.info(f"Reward for new_users_statistics {new_users_statistics}")

        # Get new records
        if "time" in self.interactions_df.columns:
            new_interactions_df = self.interactions_df[
                self.interactions_df["time"] > old_last_date
            ]

        # New items with interactions
        decisions_new: pd.Series = new_interactions_df["item_id"]
        rewards_new: pd.Series = new_interactions_df["interaction"]

        self.rec.partial_fit(decisions_new, rewards_new)
        LOGGER.info(
            f"Partial fit completed with {decisions_new.shape[0]} new interactions"
        )

        LOGGER.info("Saving new model...")
        self.save()
        LOGGER.info("Model saved!")

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

        # Get prediction for the next categories from LLM
        if use_llm:
            categories = self.get_llm_selected_cat(user_id, n=self.llm_cats_limit)
        else:
            categories = self.get_random_cat(n=self.llm_cats_limit)

        # Number of llm categories for future weights calculation
        num_llm_cats = len(categories)
        # Get weights for final sampling categories for
        # bandits

        # Get random samples of user top popular categories
        user_top_pop: pd.DataFrame = self.get_user_top_popular(
            user_id=user_id,
            n_top=self.categories_n * 2,
        )
        # initialize sapling weights for user top popular
        utp_weights = np.ones(self.categories_n)

        if user_top_pop.shape[0] > 0:

            flag_replace = False

            if user_top_pop.shape[0] < self.categories_n:
                flag_replace = True

            user_top_pop = user_top_pop.sample(self.categories_n, replace=flag_replace)

            # Get sapling weights for user top popular
            utp_weights = user_top_pop["weight"].values / np.linalg.norm(
                user_top_pop["weight"].values
            )
        else:
            utp_weights = np.array([])

        # sampling_weights[num_llm_cats : num_llm_cats + self.categories_n]
        categories.extend(user_top_pop["cat2"].tolist())

        # Get random samples of global top popular categories
        global_top_pop = self.top_popular_df.iloc[: self.categories_n * 2].sample(
            self.categories_n
        )
        gtp_weights = np.ones(shape=(self.categories_n,)) / self.categories_n

        sampling_weights = np.ones(
            shape=(num_llm_cats + utp_weights.shape[0] + gtp_weights.shape[0],)
        )
        sampling_weights[:num_llm_cats] = np.ones(shape=(num_llm_cats,))
        sampling_weights[num_llm_cats : num_llm_cats + utp_weights.shape[0]] = (
            utp_weights
        )
        sampling_weights[-gtp_weights.shape[0] :] = gtp_weights
        sampling_weights /= np.linalg.norm(sampling_weights)
        sampling_weights = sampling_weights.tolist()

        categories.extend(global_top_pop["cat2"].tolist())

        # Items to recommend
        recommendations = []

        for category in random.choices(
            categories, sampling_weights, k=self.categories_n
        ):

            # Get Items from predicted categories
            filtered_arms = self.filter_items_by_cats([category])

            if not filtered_arms:
                LOGGER.info("No items found for the selected category.")
                continue

            # Set arms for bandit_
            self.rec.set_arms(filtered_arms)
            LOGGER.info(
                f"Category: {category} Number of arms: {len(self.rec.mab.arms)}"
            )
            # Get predictions
            recommendations.extend(self.rec.recommend())

        if not recommendations:
            LOGGER.info("No recommendations generated.")
            return None

        LOGGER.info(f"recommendations: {recommendations}")
        recommended_ids = [int(item) for item in recommendations]
        recommended_items = self.items_df[
            self.items_df["item_id"].isin(recommended_ids)
        ]

        LOGGER.info(f"Recommended items: {recommended_items.item_id.tolist()}")
        return recommended_items
