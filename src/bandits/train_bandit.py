"""
Script to train and save the hierarchical bandit-based recommendation system.
"""

from datetime import timedelta
from src.bandits.bandit import Recommender, RecommenderConfig

if __name__ == "__main__":
    config = RecommenderConfig(
        predict_n_items=5,
        reward_interactions=30,
        categories_n=3,
        bandit_top_k=1,
        user_top_pop_n=10,
        global_top_pop_n=30,
        weight_llm=0.7,
        weight_utp=0.05,
        weight_gtp=0.05,
        weight_rand=0.2,
        timedelta_for_category_ban=timedelta(minutes=1),
        timedelta_for_item_ban=timedelta(minutes=5),
        models_dir="src/bandits/models",
        items_data_path="./data/items.csv",
        interactions_data_path="./data/interactions.csv",
        predicted_categories_path="./data/predicted_categories.json",
        use_all_categories=False,
    )
    recommender = Recommender(
        config=config,
    )
    recommender.fit()

    print("\n" * 10)

    print(
        recommender.predicted_categories_map[
            ("Фото и видеокамеры", "Компьютеры и периферия")
        ]
    )
    print(recommender.predict(user_id=932768504, use_llm=False))
    print(recommender.predict(user_id=932768504, use_llm=True))
