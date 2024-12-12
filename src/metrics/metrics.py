"""
Module for metrics calculations.
"""

import json

import numpy as np
import pandas as pd


def get_sessions(
    interactions_path: str, eps: int, start_time: str = None, end_time: str = None
) -> pd.DataFrame:
    """
    Get a DataFrame with session IDs assigned to each interaction based on a time threshold.

    :param interactions_path: Path to the CSV file containing user interactions data.
    :param eps: The time difference (in seconds) threshold to define a new session.
    :param start_time: Optional. Start of the time interval as a string.
    :param end_time: Optional. End of the time interval as a string.
    :return: A DataFrame with a new column "session_id" indicating
             the session each interaction belongs to.
    """
    interactions_df = pd.read_csv(interactions_path)

    if interactions_df.empty:
        print("No interactions data available.")
        return pd.DataFrame()

    interactions_df["time"] = pd.to_datetime(interactions_df["time"])

    # Filter by time range if start_time and end_time are provided
    if start_time:
        start_time = pd.to_datetime(start_time)
        interactions_df = interactions_df[interactions_df["time"] >= start_time]
    if end_time:
        end_time = pd.to_datetime(end_time)
        interactions_df = interactions_df[interactions_df["time"] <= end_time]

    interactions_df = interactions_df.sort_values(by=["user_id", "time"])

    # Calculate time differences between consecutive interactions for each user
    interactions_df["time_diff"] = (
        interactions_df.groupby("user_id")["time"].diff().dt.total_seconds()
    )

    # Identify new sessions based on the time difference threshold
    interactions_df["new_session"] = (interactions_df["time_diff"] > eps).fillna(True)

    # Assign a session ID for each interaction
    interactions_df["session_id"] = interactions_df.groupby("user_id")[
        "new_session"
    ].cumsum()

    return interactions_df


def merge_item_cat2(
    interactions_df: pd.DataFrame,
    items_path: str,
) -> pd.DataFrame:
    """
    Merge items from interactions_df with their item_id from items file.
    """
    # Load data
    items_df = pd.read_csv(items_path, sep=",")

    # Ensure 'cat2' is mapped from items.csv where missing
    interactions_df = interactions_df.merge(
        items_df[["item_id", "cat2"]],
        on="item_id",
        how="left",
        suffixes=("", "_from_items"),
    )

    # Fill missing 'cat2' from items.csv
    if "cat2_from_items" in interactions_df.columns:
        interactions_df["cat2"] = interactions_df["cat2"].combine_first(
            interactions_df["cat2_from_items"]
        )
        interactions_df.drop(columns=["cat2_from_items"], inplace=True, errors="ignore")

    return interactions_df


def coverage_metrics(interactions_path: str, items_path: str) -> float:
    """
    Calculate the coverage metric for recommendations.

    :param interactions_path: Path to the CSV file containing user interactions data.
    :param items_path: Path to the CSV file containing the catalog of all items.
    :return: Coverage metric as a float between 0 and 1, indicating the proportion
             of catalog items covered by the recommendations.
    """
    interactions_df = pd.read_csv(interactions_path)
    items_df = pd.read_csv(items_path)
    recommended_items = set(interactions_df["item_id"])
    total_items = set(items_df["item_id"])

    coverage = (
        len(recommended_items & total_items) / len(total_items) if total_items else 0
    )
    return coverage


def average_session_length(
    interactions_path: str, eps: int, start_time: str = None, end_time: str = None
) -> tuple:
    """
    Calculate the average session length based on interaction data.

    :param interactions_path: Path to the CSV file containing user interactions data.
    :param eps: The time difference (in seconds) threshold to define a new session.
    :param start_time: Optional. Start of the time interval as a string.
    :param end_time: Optional. End of the time interval as a string.
    :return: A tuple containing:
             - The average session length in terms of the number of interactions.
             - The average session duration in seconds.
    """
    interactions_df = get_sessions(interactions_path, eps, start_time, end_time)

    if interactions_df.empty:
        print("No data available for the specified time range.")
        return 0, 0

    session_lengths = interactions_df.groupby(["user_id", "session_id"]).size()
    avg_session_length = session_lengths.mean()

    # Calculate the session duration for each session
    session_times = interactions_df.groupby(["user_id", "session_id"])["time"].agg(
        ["min", "max"]
    )
    session_times["duration"] = (
        session_times["max"] - session_times["min"]
    ).dt.total_seconds()

    avg_session_time = session_times["duration"].mean()

    return avg_session_length, avg_session_time


def compute_mean_ratios(
    interactions_path: str, eps: int, start_time: str = None, end_time: str = None
) -> float:
    """
    Compute the mean ratio of likes to total interactions per session.

    :param interactions_path: Path to the CSV file containing user interactions data.
    :param eps: The time difference (in seconds) threshold to define a new session.
    :param start_time: Optional. Start of the time interval as a string.
    :param end_time: Optional. End of the time interval as a string.
    :return: The mean ratio of likes (likes / (likes + dislikes)) as a float.
    """
    interactions_df = get_sessions(interactions_path, eps, start_time, end_time)

    if interactions_df.empty:
        print("No data available for the specified time range.")
        return 0

    # Calculate the like ratio per session for each user
    session_ratios = (
        interactions_df.groupby(["user_id", "session_id"])["interaction"]
        .mean()
        .reset_index(name="like_ratio")
    )
    # Calculate mean ratio per user
    user_mean_ratios = (
        session_ratios.groupby("user_id")["like_ratio"]
        .mean()
        .reset_index(name="user_mean_ratio")
    )
    global_mean_ratio = user_mean_ratios["user_mean_ratio"].mean()

    return global_mean_ratio


def novelty_metrics(interactions_path: str) -> float:
    """
    Calculate the novelty metric for recommendations based on interaction data.

    :param interactions_path: Path to the CSV file containing user interactions data.
    :return: Novelty metric as a float, where a higher value indicates more novel
             recommendations based on item popularity.
    """

    interactions_df = pd.read_csv(interactions_path)

    if interactions_df.empty:
        print("No interaction data available for novelty calculation.")
        return 0

    # Compute item popularity based on interaction frequencies
    item_popularity = interactions_df["item_id"].value_counts(normalize=True)

    # Filter the popularity for recommended items
    recommended_popularities = item_popularity.reindex(
        set(interactions_df["item_id"]), fill_value=0
    )

    novelty = (
        -recommended_popularities.mean() * np.log(recommended_popularities.mean())
        if not recommended_popularities.empty
        else 0
    )

    return novelty


def uci_n(
    interactions_path: str,
    items_path: str,
    n_values: list,
    session_limit: int = 3,
    eps: int = 1800,
) -> dict:
    """
    Calculate the User Clustered Interest (UCI@N) metric
             based on a certain number of recent sessions.

    :param interactions_path: Path to the CSV file containing user interactions data.
                    The file should have columns 'user_id', 'interaction',
                                             'item_id', 'cat2', and 'time'.
    :param items_path: Path to the CSV file containing item metadata.
                This file should have columns 'item_id' and 'cat2'.
    :param n_values: List of N values for which to compute UCI@N.
    :param session_limit: Number of most recent sessions to consider per user.
    :param eps: The time difference (in seconds) threshold to define a new session.
    :return: A dictionary where keys are N values and values are the UCI@N metric.
    """
    # Load data
    interactions_df = get_sessions(
        interactions_path, eps
    )  # Add session IDs to interactions

    interactions_df = merge_item_cat2(interactions_df, items_path)

    # Filter interactions to only include "likes" (interaction == 1)
    liked_interactions = interactions_df[interactions_df["interaction"] == 1]

    # Select the most recent `session_limit` sessions for each user
    recent_sessions = (
        liked_interactions.groupby("user_id")
        .apply(
            lambda group: group[
                group["session_id"].isin(group["session_id"].nlargest(session_limit))
            ]
        )
        .reset_index(drop=True)
    )

    # Count unique clusters (cat2) for the selected sessions
    user_cluster_counts = (
        recent_sessions.groupby("user_id")["cat2"]
        .nunique()
        .reset_index(name="unique_clusters")
    )

    # Compute UCI@N for each N value
    uci_at_n = {}
    for n in n_values:
        # Count the number of users who interacted with at least N unique clusters
        users_with_n_clusters = user_cluster_counts[
            user_cluster_counts["unique_clusters"] >= n
        ]
        uci_at_n[n] = len(users_with_n_clusters)

    return uci_at_n


def precision(predictions_path: str, interactions_path: str):
    """
    Calculate the precision of predictions based on actual user interactions.

    Precision is defined as the ratio of true positive predictions to the total number
    of predictions made (true positives + false positives).

    :param predictions_path: Path to the JSON file containing predictions.
                             The file should be in the format:
                             [
                                 ["user_category_1", "user_category_2", "predicted_category"],
                                 ...
                             ]
                             where `predicted_category` is a semicolon-separated string.

    :param interactions_path: Path to the CSV file containing actual user interactions.
                              The file should contain columns:
                              - "user_category_1"
                              - "user_category_2"
                              - "user_category_3"
                              where `user_category_3` represents the ground truth categories
                              as a single value per row.

    :return: Precision as a float value [0, 1].
    """
    # Step 1: Load the JSON file
    with open(predictions_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Convert JSON into a DataFrame
    predicted_data = pd.DataFrame(
        json_data, columns=["user_category_1", "user_category_2", "predicted_category"]
    )

    actual_data = pd.read_csv(interactions_path, sep=",")

    # Step 3: Merge data on user_category_1 and user_category_2

    actual_test_data = pd.merge(
        actual_data,
        predicted_data,
        on=["user_category_1", "user_category_2"],
        how="inner",
    )
    # Convert user_category_3 into list for the same pairs
    actual_test_data = actual_test_data.groupby(
        ["user_category_1", "user_category_2"], as_index=False
    ).agg({"user_category_3": list, "predicted_category": "first"})

    # Convert semicolon-separated predicted_category into a list
    actual_test_data["predicted_category"] = actual_test_data[
        "predicted_category"
    ].apply(lambda x: x.split("; "))

    # Compare user_category_3 (list) and predicted_category (list)
    actual_test_data["is_correct"] = actual_test_data.apply(
        lambda row: any(
            pred in row["user_category_3"] for pred in row["predicted_category"]
        ),
        axis=1,
    )
    # Step 5: Compute Precision
    true_positives = actual_test_data["is_correct"].sum()
    false_positive = (~actual_test_data["is_correct"]).sum()

    # Avoid division by zero
    precision_value = (
        true_positives / (true_positives + false_positive)
        if (true_positives + false_positive) > 0
        else 0
    )

    return precision_value


def match_rate(predictions_path: str, cluster_descriptions_path: str) -> float:
    """
    Calculate the match rate of predictions with cluster descriptions.
    Match rate is calculated as the proportion of predicted categories
    that exist in the cluster descriptions.

    :param predictions_path: Path to a JSON file containing predictions in the specified format.
                             Format: [
                                 ["category_1", "category_2", "predicted_category; ..."],
                                 ...
                             ]
    :param cluster_descriptions_path: Path to a JSON file containing valid cluster descriptions.
                                       Format: ["description_1", "description_2", ...]
    :return: Match rate as a float (between 0 and 1).
    """
    # Load predictions
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Load cluster descriptions
    with open(cluster_descriptions_path, "r", encoding="utf-8") as f:
        cluster_descriptions = set(json.load(f))  # Convert to set for faster lookups

    # Initialize match counters
    total_matches = 0  # Total number of matches
    total_categories = 0  # Total number of predicted categories

    # Process predictions
    for prediction in predictions:
        # Extract the predicted categories and split by semicolon
        predicted_categories = prediction[2].split("; ")

        # Count matches and total predicted categories
        total_matches += sum(
            1 for pred in predicted_categories if pred in cluster_descriptions
        )
        total_categories += len(predicted_categories)

    # Calculate match rate
    return total_matches / total_categories if total_categories > 0 else 0.0
