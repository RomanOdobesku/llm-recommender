"""
Module for plotting graphs on historical data.
"""

import matplotlib.pyplot as plt
import pandas as pd
from metrics import compute_mean_ratios


def plot_like_rate(
    interactions_path: str,
    eps: int,
    config: dict = None,
):
    """
    Plot a smooth like-rate over time with custom minute intervals.

    :param interactions_path: Path to the CSV file containing user interactions data.
    :param eps: The time difference (in seconds) threshold to define sessions.
    :param config: Dictionary with configuration options:
                   - interval_minutes: Time interval in minutes (default: 30)
                   - start_time: Start of the time interval as a string (default: None)
                   - end_time: End of the time interval as a string (default: None)
                   - save_path: File path to save the generated plot (default: None)
    """
    # Default configuration
    config = config or {}
    interval_minutes = config.get("interval_minutes", 30)
    start_time = config.get("start_time")
    end_time = config.get("end_time")
    save_path = config.get("save_path")

    # Load the interactions data
    interactions = pd.read_csv(interactions_path)
    interactions["time"] = pd.to_datetime(interactions["time"], errors="coerce")
    interactions = interactions.dropna(subset=["time"])

    # Convert start_time and end_time to datetime if provided
    if start_time:
        start_time = pd.to_datetime(start_time, errors="coerce")
        interactions = interactions[interactions["time"] >= start_time]
    if end_time:
        end_time = pd.to_datetime(end_time, errors="coerce")
        interactions = interactions[interactions["time"] <= end_time]

    if interactions.empty:
        print("No valid data available for the specified time range.")
        return

    # Define minute-based time bins
    time_bins = _generate_time_bins(interactions, interval_minutes)

    # Compute like-rate for each time bin
    like_rates = _compute_like_rates(time_bins, interactions_path, eps)

    # Plot the like-rate over time
    _plot_like_rate(like_rates, eps, interval_minutes, save_path)


def _generate_time_bins(interactions, interval_minutes):
    min_time = interactions["time"].min()
    max_time = interactions["time"].max()
    return pd.date_range(start=min_time, end=max_time, freq=f"{interval_minutes}T")


def _compute_like_rates(time_bins, interactions_path, eps):
    like_rates = []
    for i in range(len(time_bins) - 1):
        time_bin_start = time_bins[i]
        time_bin_end = time_bins[i + 1]
        like_rate = compute_mean_ratios(
            interactions_path=interactions_path,
            eps=eps,
            start_time=str(time_bin_start),
            end_time=str(time_bin_end),
        )
        like_rates.append((time_bin_start, like_rate))
    return like_rates


def _plot_like_rate(like_rates, eps, interval_minutes, save_path):
    times, rates = zip(*like_rates)
    plt.figure(figsize=(12, 6))
    plt.plot(
        times,
        rates,
        label=f"Like Rate (eps={eps}s, interval={interval_minutes} minutes)",
        marker="o",
    )
    plt.xlabel("Time")
    plt.ylabel("Like Rate")
    plt.title("Smooth Like Rate Over Time (Minute Intervals)")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png")
        print(f"Graph saved at: {save_path}")
    else:
        plt.show()
