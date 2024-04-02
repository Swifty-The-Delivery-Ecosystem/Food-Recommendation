from fastapi import FastAPI
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load data and embeddings
df = pd.read_csv("indian_food.csv")
recipe_embeddings = np.load("recipe_embeddings.npy")
similarity_matrix = np.load("similarity_matrix.npy")

def get_index(recipe_name):
    return df[df.recipe_name == recipe_name].index.values[0]

def get_recipe(index):
    return df.iloc[index]["recipe_name"]

def get_recipe_recommendation(recipe_name):
    idx = get_index(recipe_name)
    similar_recipes_idx = np.argsort(similarity_matrix[idx])[::-1][1:6]  # Exclude self
    similar_recipes = [get_recipe(i) for i in similar_recipes_idx]
    return similar_recipes

@app.get("/recommend/{recipe_name}")
def recommend(recipe_name: str):
    recommendations = get_recipe_recommendation(recipe_name)
    return {"recommended_recipes": recommendations}
