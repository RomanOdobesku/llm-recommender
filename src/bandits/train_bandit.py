"""
Script to train and save the hierarchical bandit-based recommendation system.
"""

from src.bandits.bandit import Recommender, RecommenderConfig

if __name__ == "__main__":
    recommender = Recommender(
        config=RecommenderConfig(
            top_k=1,
            reward_interactions=30,
            items_data_path="./data/items.csv",
            interactions_data_path="./data/interactions.csv",
            predicted_categories_path="./data/predicted_categories.csv",
        ),
    )
    recommender.fit()
