"""
Module for metrics calculations.
"""

import numpy as np
import pandas as pd


def coverage_metric(interactions_path: str, items_path: str) -> float:
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
    Compute the global mean ratio of likes to total interactions per session.

    :param interactions_path: Path to the CSV file containing user interactions data.
    :param eps: The time difference (in seconds) threshold to define a new session.
    :param start_time: Optional. Start of the time interval as a string.
    :param end_time: Optional. End of the time interval as a string.
    :return: The global mean ratio of likes (likes / (likes + dislikes)) as a float.
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


def novelty_metric(interactions_path: str) -> float:
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


def match_rate(predictions: dict, cluster_descriptions_path: str) -> float:
    """
    Calculate the match rate of predictions with cluster descriptions.

    :param predictions: List of generated outputs from the LLM.
    :param cluster_descriptions: List of valid cluster descriptions.
    :return: Match rate as a float.
    """
    cluster_descriptions = pd.read_csv(cluster_descriptions_path, sep=";")
    cluster_descriptions = cluster_descriptions["sub_cat_name"].unique()
    matches = sum(1 for pred in predictions.values() if pred in cluster_descriptions)

    return matches / len(predictions) if predictions else 0


def recall(predictions: dict, interactions_path: str) -> float:
    """
    Calculate recall for user interest transitions.

    :param predictions: List of generated outputs from the LLM.
    :param interactions_path: List of successful user interest transitions.
    :return: Recall as a float.
    """
    interactions_df = pd.read_csv(interactions_path)
    actual_transitions = interactions_df[interactions_df["interaction"] == 1]
    matches = sum(1 for pred in predictions if pred in actual_transitions)

    return matches / len(actual_transitions) if actual_transitions else 0


def calculate_uci_at_n_sessions(
    interactions_path: str,
    n_values: list,
    session_limit: int = 3,
    eps: int = 1800,
) -> dict:
    """
    Calculate the User Clustered Interest (UCI@N)
      metric based on a certain number of recent sessions.

    :param interactions_path: Path to the CSV file containing user interactions data.
                    The file should have columns 'user_id', 'interaction', 'cat2', and 'time'.
    :param n_values: List of N values for which to compute UCI@N.
    :param session_limit: Number of most recent sessions to consider per user.
    :param eps: The time difference (in seconds) threshold to define a new session.
    :param start_time: Optional. Start of the time interval as a string.
    :param end_time: Optional. End of the time interval as a string.
    :return: A dictionary where keys are N values and values are the UCI@N metric.
    """
    # Get the interactions DataFrame with session IDs
    interactions_df = get_sessions(interactions_path, eps)

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
