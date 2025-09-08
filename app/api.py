from fastapi import FastAPI
import pandas as pd
from src.hybrid_model import HybridRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender

app = FastAPI()

# Ma'lumotlarni yuklash
movies = pd.read_csv("C:/projects_2/movie_recommendation_project/data/movies.csv")
ratings = pd.read_csv("C:/projects_2/movie_recommendation_project/data/ratings.csv")

# Modellarni yaratish
content_model = ContentBasedRecommender(movies)
collab_model = CollaborativeFilteringRecommender(ratings)
hybrid_model = HybridRecommender(movies, ratings)

# Train
content_model.train()
collab_model.train()
hybrid_model.train()

@app.get("/")
def home():
    return {"message": "Movie Recommendation API with Content, Collaborative, and Hybrid Models"}

@app.get("/recommend/content/{movie_name}")
def content_recommend(movie_name: str, top_n: int = 10):
    recs = content_model.recommend(movie_name, top_n)
    return {"recommendations": recs["title"].tolist()}

@app.get("/recommend/collaborative/{user_id}")
def collab_recommend(user_id: int, top_n: int = 10):
    recs = collab_model.recommend(user_id, movies, top_n)
    return {"recommendations": recs["title"].tolist()}

@app.get("/recommend/hybrid/{user_id}/{movie_name}")
def hybrid_recommend(user_id: int, movie_name: str, top_n: int = 10, alpha: float = 0.5):
    recs = hybrid_model.recommend(user_id, movie_name, top_n, alpha)
    return {"recommendations": recs["title"].tolist()}
