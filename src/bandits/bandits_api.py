"""
API for the hierarchical bandit-based recommendation system.
"""

import os
import pickle

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.bandits.bandit import Recommender

app = FastAPI()


def load_latest_model(models_dir: str = "src/bandits/models") -> Recommender:
    """Load the latest recommender model from the specified directory."""
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' does not exist.")
    model_files = sorted(
        (
            f
            for f in os.listdir(models_dir)
            if f.startswith("bandit_") and f.endswith(".pkl")
        ),
        key=lambda x: os.path.getmtime(os.path.join(models_dir, x)),
        reverse=True,
    )
    if not model_files:
        raise FileNotFoundError("No models found in the models directory.")
    latest_model_path = os.path.join(models_dir, model_files[0])
    with open(latest_model_path, "rb") as file:
        return pickle.load(file)


try:
    RECOMMENDER = load_latest_model()
except FileNotFoundError:
    RECOMMENDER = None


class RecommendRequest(BaseModel):
    """Request model for recommend endpoint."""

    user_id: str
    use_llm: bool = False


class UpdateRequest(BaseModel):
    """Request model for update endpoint."""

    interactions_path: str


@app.get("/")
async def root():
    """Root endpoint to confirm the API is running."""
    return {"message": "Bandit Recommender API is running."}


@app.post("/recommend/")
async def recommend(request: RecommendRequest):
    """Generate recommendations based on the current model."""
    if RECOMMENDER is None:
        raise HTTPException(status_code=500, detail="Recommender model not loaded.")
    try:
        recommendations = RECOMMENDER.predict(
            user_id=request.user_id, use_llm=request.use_llm
        )
        if recommendations is None or recommendations.empty:
            return {"recommendations": []}
        return {"recommendations": recommendations.to_dict(orient="records")}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/update/")
async def update(request: UpdateRequest):
    """Update the model with new interactions data."""
    if RECOMMENDER is None:
        raise HTTPException(status_code=500, detail="Recommender model not loaded.")
    try:
        RECOMMENDER.partial_fit(request.interactions_path)
        return {"message": "Recommender updated successfully."}

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
