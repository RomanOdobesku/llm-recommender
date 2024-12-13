"""
Script to train and save the hierarchical bandit-based recommendation system.
"""

from src.bandits.bandit import Recommender, RecommenderConfig

if __name__ == "__main__":
    config = RecommenderConfig(
        top_k=5,
        reward_interactions=30,
        categories_n=3,
        bandit_top_k=1,
        use_all_categories=False,
    )
    recommender = Recommender(
        items_data_path="./data/items.csv",
        interactions_data_path="./data/interactions.csv",
        predicted_categories_path="./data/predicted_categories.json",
        config=config,
    )
    recommender.fit()
