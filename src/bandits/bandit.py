# pylint: disable=too-many-lines
"""
Class for a hierarchical bandit-based recommendation system.

For this project CSV files were used to store data
So code needs to be changed for using databases
"""

import json
import os
import pickle
import random
from datetime import datetime, timedelta
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

    Attributes:
        predict_n_items (int): Number of items to predict by default.
        reward_interactions (int): Number of interactions to reward (give token).
        bandit_top_k (int): Parameter for BanditRecommender (number of items to choose).
        categories_n (int): Number of categories to choose from.
        user_top_pop_n (int):   Number of categories included in user's top popular list
        global_top_pop_n (int):   Number of categories included in global top popular list
        items_data_path (str): Path to the CSV file containing items data.
        interactions_data_path (str): Path to the CSV file containing interaction data.
        predicted_categories_path (str): Path to the CSV file containing predicted
            categories data.
        weight_llm (float): weight for LLM categories for random sampling
        weight_utp: (float): weight for user top popular categories for random sampling
        weight_gtp: (float): weight for global top popular categories for random sampling
        weight_rand: (float): weight for RANDOM categories for random sampling
        timedelta_for_category_ban (timedelta): The time for which the category should disappear
            from the user's recommendations due to negative integration
        timedelta_for_item_ban (timedelta): The time for which the item should disappear
            from the user's recommendations due to the fact that it appeared recently
        models_dir: (str): directory models are saved to
        use_all_categories (bool): Use all users' categories or not. Default is False.
        llm_cats_limit (Optional[int]): Limit for LLM categories. Default is None.
    """

    predict_n_items: int
    reward_interactions: int
    bandit_top_k: int
    categories_n: int
    user_top_pop_n: int
    global_top_pop_n: int
    items_data_path: str
    interactions_data_path: str
    predicted_categories_path: str
    weight_llm: float = 0.7
    weight_utp: float = 0.05
    weight_gtp: float = 0.05
    weight_rand: float = 0.2
    timedelta_for_category_ban: timedelta = timedelta(minutes=1)
    timedelta_for_item_ban: timedelta = timedelta(minutes=5)
    models_dir: str = "src/bandits/models"
    use_all_categories: bool = False
    llm_cats_limit: Optional[int] = None


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
        Initializes the recommender system with item and interaction data.

        Args:
            config (RecommenderConfig): Configuration object for recommender settings,
                including file paths, bandit parameters, and other settings.
        """

        LOGGER.info("Loading items df...")
        # Loads items data into a pandas DataFrame.
        self.items_df: pd.DataFrame = self.__load_data(config.items_data_path)

        LOGGER.info("Loading interactions df...")
        # Loads items data into a pandas DataFrame.
        self.interactions_df: pd.DataFrame = self.__load_data(
            config.interactions_data_path
        )
        # Get predicted categories
        LOGGER.info("Loading predicted_categories...")
        # Loads a map of predicted categories based on user and item IDs.
        self.predicted_categories_map: Dict[tuple, str] = (
            self.__extract_predicted_categories(config.predicted_categories_path)
        )
        LOGGER.info("ALL data loaded!")

        # Set datetime format in table
        # Converts the 'time' column in the interactions DataFrame to datetime objects.
        self.interactions_df["time"] = pd.to_datetime(self.interactions_df["time"])
        # Merging category of 2nd kind to interactions dataframe
        # Merges category information ('cat2') from the items DataFrame
        # into the interactions DataFrame.
        self.interactions_df = pd.merge(
            left=self.interactions_df,
            right=self.items_df[["item_id", "cat2"]],
            how="left",
            on="item_id",
        )
        # Get hierarchy of categories
        # Extracts hierarchical relationships between categories.
        self.hierarchical_categories: Dict[str, Dict[str, List[str]]] = (
            self.__extract_hierarchical_categories()
        )
        # Get categories to operate with
        # Extracts the list of available categories to be used in the recommender.
        self.available_categories: List[str] = self.__extract_available_categories()

        # ------------------------------
        # Set default params from config
        # ------------------------------
        # Sets the number of interactions to reward.
        self.reward_interactions: int = config.reward_interactions
        # Determines whether to use all user categories.
        self.use_all_categories: bool = config.use_all_categories
        # Sets the number of items to predict.
        self.predict_n_items: int = config.predict_n_items
        # Sets the total number of categories.
        self.categories_n: int = config.categories_n
        # Sets path to dir where models are stored
        self.models_dir: str = config.models_dir
        # Sets n for top popular
        self.user_top_pop_n: int = config.user_top_pop_n
        self.global_top_pop_n: int = config.global_top_pop_n
        # Set limiter for categories from LLM
        # Sets a limit on the number of categories to consider from the LLM.
        self.llm_cats_limit: int = (
            config.llm_cats_limit if config.llm_cats_limit else config.categories_n
        )
        # Set default params from config
        # Initializes the BanditRecommender with a specified learning policy and top_k value.
        self.rec: BanditRecommender = BanditRecommender(
            LearningPolicy.ThompsonSampling(), top_k=config.bandit_top_k
        )
        # Set weights
        self.weight_llm: float = config.weight_llm
        self.weight_utp: float = config.weight_utp
        self.weight_gtp: float = config.weight_gtp
        self.weight_rand: float = config.weight_rand

        self.timedelta_for_category_ban: timedelta = config.timedelta_for_category_ban
        self.timedelta_for_item_ban: timedelta = config.timedelta_for_item_ban
        # Set global top popular DataFrame
        self.top_popular_df = pd.DataFrame()
        self.__update_top_popular_df()

    def __update_top_popular_df(self) -> None:
        """
        Updates the DataFrame containing the most popular categories and their weights.
        """

        # DataFrame with columns=["cat2", "count"]
        # Creates a DataFrame of the most popular categories and their weights.
        if not (self.interactions_df.empty and self.items_df.empty):
            self.top_popular_df = (
                self.interactions_df[self.interactions_df["interaction"] == 1]["cat2"]
                .value_counts()  # Counts occurrences of each 'cat2' category
                .to_frame()  # Converts the Series to a DataFrame
                .reset_index()  # Resets the index to make 'cat2' a regular column
            ).rename(  # Renames the 'count' column to 'weight'
                columns={
                    "count": "weight",
                }
            )
            # Calculates the weights of each category
            # based on the number of positive interactions.
            # Normalizes the counts to get a weight
            # based on total number of positive interactions
            self.top_popular_df["weight"] = (
                self.top_popular_df["weight"]
                / self.interactions_df[self.interactions_df["interaction"] == 1].shape[
                    0
                ]
            )

    def __load_data(self, filename: str) -> pd.DataFrame:
        """
        Loads data from a CSV file into a pandas DataFrame.
        Notion: using a semicolon (`;`) for CSV file as a separator

        Args:
            filename (str): The path to the CSV file that needs to be loaded.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the data loaded from the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: If any other error occurs during file loading (e.g., file corruption).
        """
        try:
            # Attempt to read the CSV file into a DataFrame, using ';' as a separator.
            df = pd.read_csv(filename, sep=";")
            LOGGER.info(f"Dataset loaded from {filename}")
            return df
        except FileNotFoundError as e:
            # Handle the case where the specified file is not found.
            LOGGER.info(f"Error: File {filename} not found.")
            raise e
        except Exception as e:
            # Handle any other exceptions that occur during file loading.
            LOGGER.info(f"Error while loading file: {e}.")
            raise e

    def __get_reward_users(
        self, old_users_statistics: pd.Series, new_users_statistics: pd.Series
    ) -> List[int]:
        """
        Identifies users eligible for a reward based on changes in their interaction counts.

        This method compares the interaction counts of users between two time points,
        using a normalization factor to determine whether a user's interaction count has
        increased enough to qualify for a reward.

        Args:
            old_users_statistics (pd.Series): A pandas Series containing user IDs as index
                and their old interaction counts as values.
            new_users_statistics (pd.Series): A pandas Series containing user IDs as index
                and their new interaction counts as values.

        Returns:
            List[int]: A list of user IDs that are eligible for a reward.
        """

        # Convert series to DataFrames and reset index
        old_stats_df = old_users_statistics.rename("old_count").reset_index()
        new_stats_df = new_users_statistics.rename("new_count").reset_index()

        # Merge the old and new statistics DataFrames based on 'user_id'
        comparison_df = pd.merge(old_stats_df, new_stats_df, on="user_id", how="outer")

        # Fill any missing values (users that exist in only one of the series) with 0
        comparison_df.fillna(0, inplace=True)

        # Convert 'old_count' and 'new_count' to integers
        comparison_df["old_count"] = comparison_df["old_count"].astype(int)
        comparison_df["new_count"] = comparison_df["new_count"].astype(int)

        # Calculate the normalized counts by integer division by reward_interactions
        comparison_df["old_normalized"] = (
            comparison_df["old_count"] // self.reward_interactions
        )
        comparison_df["new_normalized"] = (
            comparison_df["new_count"] // self.reward_interactions
        )
        # Filter the DataFrame to find users whose normalized interaction count has increased
        changed_users = comparison_df[
            comparison_df["new_normalized"] > comparison_df["old_normalized"]
        ]

        # Extract the 'user_id' column and convert it to a list
        return changed_users["user_id"].tolist()

    def __extract_predicted_categories(
        self, predicted_categories_path: str
    ) -> Dict[Tuple[str, str], str]:
        """
        Extracts predicted categories from a JSON file.

        This function reads a JSON file containing predicted categories
        and constructs a dictionary mapping pairs of (user_id, item_id)
        to the predicted category.

        Args:
            predicted_categories_path (str): The path to the JSON file containing
                predicted categories.

        Returns:
            Dict[Tuple[str, str], str]: A dictionary where keys are tuples of (user_id, item_id)
                and values are the corresponding predicted category.

        Raises:
            FileNotFoundError: If the input file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
            Exception: For any other unexpected error.
        """
        predicted_categories_map = {}

        try:
            with open(predicted_categories_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)

            for prediction in predictions:
                predicted_categories_map[(prediction[0], prediction[1])] = prediction[3]

            return predicted_categories_map

        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found: {predicted_categories_path}"
            ) from None

        except json.JSONDecodeError as e:
            raise json.JSONDecodeError("Invalid JSON format", e.doc, e.pos) from None

    def __extract_hierarchical_categories(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Extracts hierarchy categories from the items DataFrame:
        category (cat1) -> sub_category (cat2) -> sub_sub_category (cat3)

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary representing the hierarchical
                category structure, or an empty dict if an error is encountered.
                Keys are 'cat1' values, nested dicts are 'cat2' values, and lists are sorted
                unique 'cat3' values.

        Raises:
            KeyError: If the required columns ('cat1', 'cat2', 'cat3')
                are missing from `self.items_df`.
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

            LOGGER.info("Hierarchical categories extracted")
            return hierarchical_categories

        except KeyError as e:
            LOGGER.error(f"Error: Missing column(s) in items_df: {e}")
            return {}
        except Exception as e:
            LOGGER.error(f"An unexpected error occurred: {e}")
            return {}

    def __extract_available_categories(self) -> List[str]:
        """
        Extracts all unique second-level categories ('cat2') from
        the hierarchical category structure.

        Returns:
            List[str]: A list of unique second-level categories available
                in the hierarchical structure.

        Raises:
            TypeError: If `self.hierarchical_categories` is not a dictionary.
            KeyError: If the structure of `self.hierarchical_categories` does not match
                the expected format (i.e. nested dictionaries).
        """

        try:

            available_categories: List[str] = [
                cat2
                for cat2_dict in self.hierarchical_categories.values()
                for cat2 in cat2_dict
            ]

            LOGGER.info(f"Extracted available categories: {available_categories}")
            return available_categories

        except (TypeError, KeyError) as e:
            LOGGER.error(f"Error while extracting available categories: {e}")
            return []
        except Exception as e:
            LOGGER.error(f"An unexpected error occurred: {e}")
            return []

    def __get_random_cat(self, n: int = 1) -> List[str]:
        """
        Randomly selects `n` second-level categories from the available categories.

        Args:
            n (int, optional): The number of categories to randomly select. Defaults to 1.

        Returns:
            List[str]: A list of randomly selected second-level categories.

        Raises:
            ValueError: If n is a negative number.
        """

        if n < 0:
            raise ValueError(f"n must be a positive number. Passed n = {n}")

        num_available = len(self.available_categories)

        if n > num_available:
            LOGGER.error(
                f"Requested number of categories (n={n}) exceeds "
                f"available categories ({num_available}). "
                "Adjusting to available maximum."
            )
            n = num_available  # Adjust n to the number of available categories

        # Use random.sample to ensure unique categories
        selected_categories = random.sample(self.available_categories, n)

        LOGGER.info(f"Selected random categories: {selected_categories}")

        return selected_categories

    def __get_user_interactions(
        self,
        user_id: int,
        interaction_type: Optional[int] = None,
        time_delta: Optional[timedelta] = None,
    ):
        """
        Retrieves user interactions from the interactions DataFrame
        based on specified filters.

        Args:
            user_id (int): The ID of the user whose interactions are to be retrieved.
            interaction_type (Optional[int], optional): Filters by interaction type:
                - None (default): Returns all interactions.
                - 1: Returns only positive interactions.
                - 0: Returns only negative interactions.
                Defaults to None.
            timedelta (Optional[timedelta], optional): Filters interactions within
                the specified time window before the most recent interaction
                of the user. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered user interactions,
            or an empty DataFrame if no matching interactions are found.

        Raises:
            TypeError: If user_id is not an integer.
            TypeError: If positive is not None or an int 0 or 1
        """

        if interaction_type not in (None, 0, 1):
            raise ValueError("positive must be None, 0 or 1")

        # Get interactions by user_id
        user_interactions_df = self.interactions_df[
            self.interactions_df["user_id"] == user_id
        ]

        if user_interactions_df.empty:
            LOGGER.info(f"No interactions found for user {user_id}.")
            return pd.DataFrame(columns=self.interactions_df.columns)

        # Filter by interaction type if specified
        if interaction_type is not None:
            user_interactions_df = user_interactions_df[
                (user_interactions_df["interaction"] == interaction_type)
            ]

        # Filter by time window if specified
        if time_delta is not None:
            max_user_date = user_interactions_df["time"].max()
            user_interactions_df = user_interactions_df[
                (user_interactions_df["time"] > max_user_date - time_delta)
            ]

        LOGGER.info(
            f"Returning {len(user_interactions_df)} interaction(s) for user {user_id}"
        )

        return user_interactions_df

    def __get_user_last_categories(
        self,
        user_id: int,
        interaction_type: Optional[int] = 1,
        n_last: Optional[int] = None,
        time_delta: Optional[timedelta] = None,
    ):
        """
        Retrieves the last `n_last` categories from user interactions
        based on a specified interaction type.

        Args:
            user_id (int): The ID of the user.
            interaction_type (Optional[int], optional): The type of interaction to filter by:
                - 1: Positive interactions.
                - 0: Negative interactions.
                Defaults to 1.
            n_last (int): The number of last categories to retrieve.
            timedelta (Optional[timedelta], optional): Filters interactions
                within the specified time window before the most recent interaction
                of the user. Defaults to None.

        Returns:
            List[str]: A list of the user's last categories based on interaction type.

        Raises:
            ValueError: If interaction_type is not None, 0 or 1.
            ValueError: If n_last is not a positive number.
        """

        if interaction_type not in [None, 0, 1]:
            raise ValueError("interaction_type must be None, 0 or 1")

        # Get user interactions
        user_interactions = self.__get_user_interactions(
            user_id,
            interaction_type,
            time_delta,
        )

        if n_last is None:
            user_last = user_interactions["cat2"].dropna().tolist()
            LOGGER.info(f"Returning all last user categories: {user_last}")
            return user_last

        if n_last <= 0:
            raise ValueError("n_last must be a positive number")

        # Extract all non-null 'cat2' values from the DataFrame
        user_categories = user_interactions["cat2"].dropna().tolist()

        # Get last n from the list of categories
        user_last = (
            user_categories[-n_last:]
            if len(user_categories) >= n_last
            else user_categories
        )
        if not isinstance(user_last, list):
            user_last = []

        # Calculate number of random categories to add for padding
        num_to_add = max(0, n_last - len(user_last))
        user_last = self.__get_random_cat(num_to_add) + user_last

        # Return last n categories
        user_last = user_last[-n_last:]

        LOGGER.info(
            f"Retrieved last {n_last} categories for user {user_id},"
            f"interaction_type: {interaction_type}: {user_last}"
        )
        return user_last

    def __get_user_top_popular_categories(
        self, user_id: int, n_top: int
    ) -> pd.DataFrame:
        """
        Retrieves the top `n_top` most popular categories for a user
        based on their interaction history.

        Args:
            user_id (int): The ID of the user.
            n_top (int): The number of top categories to return.

        Returns:
            pd.DataFrame: A DataFrame with columns "cat2" (category names)
                and "weight" (normalized frequency). Returns an empty DataFrame
                with columns "cat2" and "weight" if input is invalid
                or no positive interactions are present.

        Raises:
          TypeError: If n_top is not an integer.
          ValueError: If n_top is not a positive number.
        """

        try:

            if n_top <= 0:
                LOGGER.warning("n_top must be positive, returning an empty DataFrame")
                return pd.DataFrame(columns=["cat2", "weight"])

            # Get user's positive interactions
            user_interactions = self.__get_user_interactions(
                user_id=user_id,
                interaction_type=1,
            )

            # Check for empty DataFrame and prevent errors
            if user_interactions.empty:
                LOGGER.info("Input DataFrame is empty. Returning empty DataFrame")
                return pd.DataFrame(columns=["cat2", "weight"])

            # Calculate the value counts of categories and convert to a DataFrame
            user_top = (
                user_interactions["cat2"]
                .value_counts()
                .to_frame()
                .reset_index()
                .rename(
                    columns={
                        "count": "weight",
                    }
                )
            )

            # Normalize weights by dividing with the total number of user interactions
            user_top["weight"] = user_top["weight"] / user_interactions.shape[0]

            # Select the top n categories
            user_top_popular = user_top.iloc[:n_top]
            LOGGER.info(
                f"Returning top {n_top} categories for user {user_id}: {user_top_popular}"
            )

            return user_top_popular

        except KeyError as e:
            LOGGER.warning(
                f"KeyError: 'cat2' column missing in interactions for user {user_id}: {e}."
                "Returning empty DataFrame"
            )
            return pd.DataFrame(columns=["cat2", "weight"])

        except ZeroDivisionError as e:
            LOGGER.warning(
                f"ZeroDivisionError: No positive interactions found for user {user_id}"
                f"or division by zero: {e}. Returning empty DataFrame"
            )
            return pd.DataFrame(columns=["cat2", "weight"])
        except Exception as e:
            LOGGER.error(
                f"An unexpected error occurred: {e}. Returning empty DataFrame"
            )
            return pd.DataFrame(columns=["cat2", "weight"])

    def __get_llm_selected_cat(self, user_id: int, n: int = 1) -> List[str]:
        """
        Selects `n` categories using an LLM-generated prediction.

        Args:
            user_id (int): The ID of the user.
            n (int, optional): Limitation for the number of categories to select. Defaults to 1.

        Returns:
            List[str]: A list of selected second-level categories.

        Raises:
            TypeError: If n is not an integer.
            ValueError: If n is not a positive number.
        """

        LOGGER.info(f"Get statistics for: {user_id}")

        user_last_liked = self.__get_user_last_categories(
            user_id=user_id,
            n_last=2,
            interaction_type=1,
        )

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
                        f"Predicted category '{cat}' not in available categories."
                    )

            valid_cats = (
                random.sample(
                    valid_cats,
                    min(n, len(valid_cats)),
                )
                if valid_cats
                else []
            )

            LOGGER.info(
                f"LLM-selected valid categories for user {user_id}: {valid_cats}"
            )

            return list(set(valid_cats))

        return []

    def __filter_items_by_cats(self, categories: List[str]) -> npt.ArrayLike:
        """
        Filter items by a list of second-level categories.

        Args:
            categories (List[str]): A list of second-level categories to filter items by.

        Returns:
             npt.NDArray: A NumPy array containing the item IDs of the filtered items.
             Returns an empty array if no items match the specified
             categories or categories list is empty.
        """

        LOGGER.info(f"Filtered by categories: {categories}")
        filtered_items = self.items_df[self.items_df["cat2"].isin(categories)][
            "item_id"
        ].values.tolist()

        return filtered_items

    def fit(self) -> None:
        """
        Train the recommender using initial interaction data.

        Returns: None:
        """
        if self.interactions_df.empty:
            LOGGER.info("No interactions data to fit.")
            return

        decisions: pd.Series = self.interactions_df["item_id"]
        rewards: pd.Series = self.interactions_df["interaction"]

        self.rec.fit(decisions=decisions, rewards=rewards)
        self.__save_model()

        LOGGER.info(f"Fit completed with {len(decisions)} interactions.")

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

        # Get statistics (number of user interactions with each category)
        # from current interactions_df
        old_users_statistics = self.interactions_df["user_id"].value_counts()
        LOGGER.info(f"Reward for old_users_statistics {old_users_statistics}")
        old_last_date = self.interactions_df["time"].max()

        # Set new interactions_df
        self.interactions_df = pd.merge(
            left=new_interactions_df,
            right=self.items_df,
            how="left",
            on="item_id",
        )
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
        self.__save_model()
        LOGGER.info("Model saved!")

        reward_users = self.__get_reward_users(
            old_users_statistics, new_users_statistics
        )
        LOGGER.info(f"Rewarded users: {reward_users}")

        return reward_users

    def __save_model(self) -> None:
        """
        Save the trained model to the models directory with a timestamped filename.

        :return: None
        """
        models_dir = self.models_dir
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

    def __get_valid_categories(self, user_id: int, categories: npt.ArrayLike):
        """
        Ensures categories are valid and not present in recent user dislikes.

        Args:
            user_id (int): The ID of the user.
            categories (List[str]): A list of categories to validate.

        Returns:
           np.ndarray: An array of valid and unique categories.
        """
        categories = np.array(categories)  # Convert to numpy array for efficiency
        LOGGER.info(f"Validating categories for user {user_id}: {categories}")

        # Get disliked categories
        disliked_categories = self.__get_user_interactions(
            user_id,
            interaction_type=0,
            time_delta=self.timedelta_for_category_ban,
        )["cat2"].unique()

        is_in_dislike = np.isin(categories, disliked_categories)

        # Check if ther are dislikes
        while True:
            # Replace disliked categories
            if np.any(is_in_dislike):
                valid_cats = self.items_df[~self.items_df["cat2"].isin(categories)][
                    "cat2"
                ].unique()

                categories = np.where(
                    is_in_dislike,
                    np.random.choice(valid_cats) if valid_cats.size else categories,
                    categories,
                )

            # Check if values are unique
            unique_values, counts = np.unique(categories, return_counts=True)
            if np.all(counts == 1):
                break

            # Find the duplicates
            duplicates = unique_values[counts > 1]
            for duplicate_value in duplicates:
                duplicate_indices = np.where(categories == duplicate_value)[0]
                valid_cats = self.items_df[~self.items_df["cat2"].isin(categories)][
                    "cat2"
                ].unique()
                # Replace all duplicates with randoms
                # ZERO index never duplicates
                for idx in duplicate_indices[1:]:
                    categories[idx] = np.random.choice(valid_cats)

            # Recheck the categories in dislikes after replacement
            disliked_categories = self.__get_user_interactions(
                user_id,
                interaction_type=0,
                time_delta=self.timedelta_for_category_ban,
            )["cat2"].unique()

            is_in_dislike = np.isin(categories, disliked_categories)

        LOGGER.info(f"Validated categories for user {user_id}: {categories}")
        return categories

    def __check_item_in_last_user_interactions(
        self, user_id: int, item_id: int
    ) -> bool:
        """
        Check if item was in user's interactions in last N minutes
        """

        ui_interations = self.interactions_df[
            (self.interactions_df["user_id"] == user_id)
            & (self.interactions_df["item_id"] == item_id)
        ]

        if ui_interations.empty:
            return False

        if (
            ui_interations["time"]
            < ui_interations["time"].max() - self.timedelta_for_item_ban
        ):
            return True

        return False

    def predict(  # pylint: disable=too-many-locals
        self, user_id: int, use_llm: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Make recommendations based on the specified category selection method.

        :param user_id: User ID
        :param use_llm: Whether to use an LLM for category selection.
        :param predict_n_items: number of items we want to predict.

        :return: List of recommended items, or None if no items found.
        """

        # Get prediction for the next categories from LLM
        if use_llm:
            categories = self.__get_llm_selected_cat(user_id, self.llm_cats_limit)
        else:
            categories = self.__get_random_cat(self.llm_cats_limit)

        categories = np.array(categories)
        num_llm_cats = len(categories)

        # WORK with user top popular
        # Get random samples of user top popular categories
        user_top_pop: pd.DataFrame = self.__get_user_top_popular_categories(
            user_id=user_id,
            n_top=self.user_top_pop_n,
        )
        # initialize sapling weights for user top popular
        utp_weights = np.ones(self.categories_n)
        if user_top_pop.shape[0] > 0:
            flag_replace = user_top_pop.shape[0] < self.categories_n
            user_top_pop = user_top_pop.sample(self.categories_n, replace=flag_replace)
            # Get sapling weights for user top popular
            # MAIN IDEA: To save relations between user's top popular items weights
            utp_weights = (user_top_pop["weight"] / user_top_pop["weight"].sum()).values
        else:
            utp_weights = np.array([])

        # WORK with global top popular
        # Get random samples of global top popular categories
        global_top_pop = self.top_popular_df.iloc[: self.global_top_pop_n].sample(
            self.categories_n
        )
        # Get sapling weights for global top top_popular_df
        # MAIN IDEA: To save relations between global top popular items weights
        gtp_weights = (global_top_pop["weight"] / global_top_pop["weight"].sum()).values

        # WORK with random categories' choices
        rand_cats = random.choices(
            self.items_df["cat2"].unique(),
            k=self.categories_n,
        )
        # Get sapling weights for RAND
        rand_weights = np.ones(shape=(self.categories_n,)) / self.categories_n

        categories = np.array(categories)
        # SET ALL SAMPLING WEIGHTS
        sampling_weights = np.concatenate(
            (
                np.ones(shape=(num_llm_cats,)) * self.weight_llm,  # LLM weights
                utp_weights * self.weight_utp,  # UTP weights
                gtp_weights * self.weight_gtp,  # GTP weights
                rand_weights * self.weight_rand,  # RAND weights
            ),
            axis=0,
        )
        # Normalize weights
        sampling_weights /= sampling_weights.sum()
        sampling_weights = sampling_weights.tolist()

        # Combine and validate
        categories = np.concatenate(
            (
                categories,
                (
                    user_top_pop["cat2"].values
                    if not user_top_pop.empty
                    else np.empty(shape=0)
                ),
                (
                    global_top_pop["cat2"].values
                    if not global_top_pop.empty
                    else np.empty(shape=0)
                ),
                rand_cats,
            )
        )
        categories = self.__get_valid_categories(user_id, categories)

        # initialize recommendations (RESULT)
        recommendations = []

        flag_replace = len(categories) < self.predict_n_items
        for category in np.random.choice(
            a=categories,
            p=sampling_weights,
            replace=flag_replace,
            size=self.predict_n_items,
        ):
            # Get Items from predicted categories
            filtered_arms = self.__filter_items_by_cats([category])

            if not filtered_arms:
                LOGGER.info("No items found for the selected category.")
                continue
            # Set arms for bandit_
            self.rec.set_arms(filtered_arms)
            # Recommend
            subrec = self.rec.recommend()
            # Need only new interactions
            while subrec[0] in recommendations:
                counter = 0
                subrec = self.rec.recommend()
                while (
                    self.__check_item_in_last_user_interactions(user_id, subrec[0])
                    and counter < 5
                ):
                    counter += 1
                    subrec = self.rec.recommend()
            recommendations.extend(subrec)

        if not recommendations:
            LOGGER.info("No recommendations generated.")
            return None

        recommended_items = self.items_df[
            self.items_df["item_id"].isin(recommendations)
        ]

        LOGGER.info(f"Recommended items: {recommended_items.item_id.tolist()}")
        return recommended_items
