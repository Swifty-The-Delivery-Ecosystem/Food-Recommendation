from fastapi import FastAPI
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load the dataset
dataset_url = "https://raw.githubusercontent.com/nehaprabhavalkar/indian-food-101/main/indian_food.csv"
df = pd.read_csv(dataset_url)

# Initialize SentenceTransformer model
bert_model = SentenceTransformer("bert-base-nli-mean-tokens")


# Function to calculate embeddings for recipes
def calculate_embeddings(data):
    embeddings = bert_model.encode(data).tolist()
    return embeddings


# Function to recommend similar recipes
def recommend_similar_recipes(recipe_name, embeddings):
    recipe_index = df[df["name"] == recipe_name].index[0]
    similarities = cosine_similarity([embeddings[recipe_index]], embeddings)[0]
    similar_indices = np.argsort(similarities)[::-1][1:11]  # Exclude self
    similar_recipes = df.iloc[similar_indices]["name"].tolist()
    return similar_recipes


# API endpoint to recommend similar recipes
@app.get("/recommend/{recipe_name}")
async def recommend_recipe(recipe_name: str):
    # Calculate embeddings for recipe ingredients
    embeddings = calculate_embeddings(df["ingredients"])
    # Recommend similar recipes
    recommended_recipes = recommend_similar_recipes(recipe_name, embeddings)
    return {"recommended_recipes": recommended_recipes}
