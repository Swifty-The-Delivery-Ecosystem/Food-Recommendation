from fastapi import FastAPI
import numpy as np
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key not set. Please set the GROQ_API_KEY in the .env file.")

client = Groq(api_key=api_key)

food_list = [
    "Burger", "Cold Coffee", "Hot Coffee", "Veg Pizza", "Pav Bhaji", "Aloo Matar", "Samosa",
    "Dum Aloo", "Doodhpak", "Chak Hao Kheer", "Kheer", "Dosa", "Idli", "Aloo Paratha",
    "Chole Kulche", "Bhel", "Vada Pav", "Litti Chokha", "Mocktail", "Chocolate Pastries"
]

def get_groq_recommendations(food_item):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""I will provide a food item from the following list:
                                ["Burger", "Cold Coffee", "Hot Coffee", "Veg Pizza", "Pav Bhaji", "Aloo Matar", "Samosa",
                                "Dum Aloo", "Doodhpak", "Chak Hao Kheer", "Kheer", "Dosa", "Idli", "Aloo Paratha",
                                "Chole Kulche", "Bhel", "Vada Pav", "Litti Chokha", "Mocktail", "Chocolate Pastries"].

                                The food item I’m providing is "{food_item}".

                                Based on this item, return an array of other similar items from this list only.
                                Only include items from the provided list, and ensure they are relevant to "{food_item}".
                                Only give me the array no other text or anything just array. Only give response as array"""
                },
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=50,
            top_p=1,
            stream=False,
            stop=None,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error fetching recommendations for {food_item}: {e}")
        return []

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

@app.post("/v2/recommend")
def recommendV2(food_items: list[str]):
    if len(food_items) >= 3:
        food_items = food_items[:3]
    all_recommendations = []

    for item in food_items:
        recommendations = get_groq_recommendations(item)
        
        try:
            rec_list = eval(recommendations)
            all_recommendations.extend(rec_list)
        except:
            continue

    recommendation_counts = Counter(all_recommendations)
    sorted_recommendations = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)

    return {"recommended_food_items": [rec[0] for rec in sorted_recommendations]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)