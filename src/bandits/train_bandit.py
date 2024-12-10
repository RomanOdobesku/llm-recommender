"""
Script to train and save the hierarchical bandit-based recommendation system.
"""

from src.bandits.bandit import Recommender

if __name__ == "__main__":
    recommender = Recommender(
        items_data_path="./data/items.csv",
        interactions_data_path="./data/interactions.csv",
    )
    recommender.fit()
