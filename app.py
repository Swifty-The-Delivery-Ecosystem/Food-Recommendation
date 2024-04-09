from fastapi import FastAPI
import numpy as np
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import uvicorn

app = FastAPI()
df = pd.read_csv("indian_food.csv")
food_item_embeddings = np.load("recipe_embeddings.npy")
similarity_matrix = np.load("similarity_matrix.npy")

def get_index(food_item):
    return df[df.name == food_item].index.values[0]

def get_food_item(index):
    return df.iloc[index]["name"]

def get_food_item_recommendation(food_item):
    idx = get_index(food_item)
    similar_food_item_idx = np.argsort(similarity_matrix[idx])[::-1][1:6]
    similar_food_item = [get_food_item(i) for i in similar_food_item_idx]
    return similar_food_item

@app.post("/recommend")
def recommend(food_item: list[str]):
    if len(food_item) != 5:
        return {"error": "Please provide exactly 5 food items."}

    with ThreadPoolExecutor() as executor:
        recommendations = list(executor.map(get_food_item_recommendation, food_item))

    all_recommendations = [rec for recs in recommendations for rec in recs]
    recommendation_counts = Counter(all_recommendations)
    sorted_recommendations = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)

    return {"recommended_food_items": [rec[0] for rec in sorted_recommendations]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)