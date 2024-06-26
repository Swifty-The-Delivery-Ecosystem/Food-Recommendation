import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from metaflow import FlowSpec, Parameter, step
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class FoodRecommendationPipeline(FlowSpec):

    # Define parameters if needed
    DATA_FILE = Parameter(
        "data_file",
        "https://raw.githubusercontent.com/Swifty-The-Delivery-Ecosystem/Food-Recommendation/master/indian_food.csv",
        help="Path to the food data CSV file",
    )

    @step
    def start(self):
        self.next(self.prepare_dataset)

    @step
    def prepare_dataset(self):
        # Read the dataset
        self.data = pd.read_csv(self.DATA_FILE)

        # Impute missing values for 'state' and 'region'
        self.data.iloc[[7, 94, 109, 115], [7, 8]] = ["Delhi", "North"]
        self.data.iloc[[9, 117], [7, 8]] = ["Uttar Pradesh", "North"]
        self.data.iloc[[10, 96, 145, 149, 154, 158, 164], [7, 8]] = [
            "Andhra Pradesh",
            "South",
        ]
        self.data.iloc[[12, 98], [7, 8]] = ["Gujarat", "West"]
        self.data.iloc[[111, 128, 130, 144, 156, 161, 162, 231, 248], [7, 8]] = [
            "Tamil Nadu",
            "South",
        ]

        # Replace nan for 'region'
        self.data["region"] = self.data["region"].fillna(self.data["region"].mode()[0])

        # Replace '-1' for 'flavor_profile' with 'unique' word
        self.data["flavor_profile"].replace("-1", "unique", regex=True, inplace=True)

        # Replace prep_time and cook_time with respective median values
        self.data["prep_time"].replace(
            -1, self.data["prep_time"].median(), regex=True, inplace=True
        )
        self.data["cook_time"].replace(
            -1, self.data["cook_time"].median(), regex=True, inplace=True
        )

        # Proceed to generate embeddings
        self.next(self.generate_embeddings)

    @step
    def generate_embeddings(self):
        # Plot recipes by Diet, Course, and Flavor
        df_diet = self.data["diet"].value_counts()
        df_course = self.data["course"].value_counts()
        df_flavor = self.data["flavor_profile"].value_counts()

        plt.figure(figsize=(15, 8))

        plt.subplot(1, 3, 1)
        plt.pie(df_diet, labels=df_diet.index, autopct="%1.1f%%")
        plt.title("Recipes by Diet")

        plt.subplot(1, 3, 2)
        plt.pie(df_course, labels=df_course.index, autopct="%1.1f%%")
        plt.title("Recipes by Course")

        plt.subplot(1, 3, 3)
        plt.pie(
            df_flavor,
            labels=df_flavor.index,
            autopct="%1.1f%%",
            explode=[0.01] * len(df_flavor),
        )
        plt.title("Recipes by Flavor")

        plt.show()

         # Use SentenceTransformer to generate embeddings for recipe ingredients
        bert_model = SentenceTransformer("bert-base-nli-mean-tokens")
        recipe_embeddings = bert_model.encode(self.data["ingredients"]).tolist()
        similarity_matrix = cosine_similarity(recipe_embeddings)

        # Save embeddings and similarity matrix to the repository
        embeddings_file = os.path.join(os.getcwd(), "recipe_embeddings.npy")
        similarity_matrix_file = os.path.join(os.getcwd(), "similarity_matrix.npy")
        np.save(embeddings_file, recipe_embeddings)
        np.save(similarity_matrix_file, similarity_matrix)

        # Proceed to final recommendation step
        self.next(self.end)


    @step
    def end(self):
        # This step is the endpoint of the pipeline
        # Perform any final processing or output results
        print("Pipeline completed successfully!")


if __name__ == "__main__":
    FoodRecommendationPipeline()
