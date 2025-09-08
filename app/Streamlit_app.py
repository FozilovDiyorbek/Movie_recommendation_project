import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from src.hybrid_model import HybridRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender

movies = pd.read_csv("C:/projects_2/movie_recommendation_project/data/movies.csv")
ratings = pd.read_csv("C:/projects_2/movie_recommendation_project/data/ratings.csv")

content_model = ContentBasedRecommender(movies)
collab_model = CollaborativeFilteringRecommender(ratings)
hybrid_model = HybridRecommender(movies, ratings)

content_model.train()
collab_model.train()
hybrid_model.train()

st.title("🎬 Movie Recommendation System (Hybrid Model)")

user_id = st.number_input("User ID:", min_value=1, max_value=int(ratings["userId"].max()), value=1)
movie_list = movies['title'].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

model_choice = st.radio("Select Recommendation Type:", ["Content-Based", "Collaborative", "Hybrid"])

if st.button("Recommend"):
    if model_choice == "Content-Based":
        recs = content_model.recommend(selected_movie, top_n=10)
    elif model_choice == "Collaborative":
        recs = collab_model.recommend(user_id, movies, top_n=10)
    else:
        recs = hybrid_model.recommend(user_id, selected_movie, top_n=10, alpha=0.6)

    st.write("Top Recommendations:")
    for i, row in enumerate(recs.itertuples(), 1):
        st.write(f"{i}. {row.title}")
