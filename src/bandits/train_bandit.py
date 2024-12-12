"""
Script to train and save the hierarchical bandit-based recommendation system.
"""

from src.bandits.bandit import Recommender, RecommenderConfig

if __name__ == "__main__":
    config = RecommenderConfig(top_k=3, reward_interactions=30)
    recommender = Recommender(
        items_data_path="./data/items.csv",
        interactions_data_path="./data/interactions.csv",
        predicted_categories_path="./data/predicted_categories.json",
        config=config,
    )
    recommender.fit()
