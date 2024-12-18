"""
Module for base test of bandit recommender
"""

import requests

from src.logger import LOGGER

BASE_URL = "http://127.0.0.1:8000"


def test_recommend():
    """Test the recommend endpoint with a sample user input."""
    response = requests.post(  # pylint: disable=missing-timeout
        f"{BASE_URL}/recommend/", json={"user_id": 87346237431, "use_llm": True}
    )
    print("Recommend Response:", response.json())
    assert response.status_code == 200, "Failed to get recommendations"
    assert (
        "recommendations" in response.json()
    ), "Response missing 'recommendations' key"


def test_update():
    """Test the update endpoint with a sample interaction data path."""
    response = requests.post(  # pylint: disable=missing-timeout
        f"{BASE_URL}/update/", json={"interactions_path": "data/interactions.csv"}
    )
    assert response.status_code == 200, "Failed to update the model"
    LOGGER.info(response.json())


if __name__ == "__main__":
    test_recommend()
    print("\n" * 10)
    test_update()
    print("\n" * 10)
    test_recommend()
