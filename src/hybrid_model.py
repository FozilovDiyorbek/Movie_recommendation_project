import pandas as pd
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender

class HybridRecommender:
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.content_model = ContentBasedRecommender(movies)
        self.collab_model = CollaborativeFilteringRecommender(ratings)
        self.movies = movies

    def train(self):
        self.content_model.train()
        rmse = self.collab_model.train()
        return rmse

    def recommend(self, user_id: int, movie_title: str, top_n: int = 10, alpha: float = 0.5):
        """
        Hybrid recommendation: weighted sum of content and collaborative
        """
        content_recs = self.content_model.recommend(movie_title, top_n * 2)
        collab_recs = self.collab_model.recommend(user_id, self.movies, top_n * 2)

        merged = pd.concat([content_recs, collab_recs]).drop_duplicates("movieId")
        merged["score"] = (
            merged["movieId"].apply(lambda x: 1 if x in content_recs["movieId"].values else 0) * alpha
            + merged["movieId"].apply(lambda x: 1 if x in collab_recs["movieId"].values else 0) * (1 - alpha)
        )
        merged = merged.sort_values("score", ascending=False).head(top_n)
        return merged
