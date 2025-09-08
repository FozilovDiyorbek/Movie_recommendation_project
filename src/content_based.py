import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


class ContentBasedRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.cosine_sim = None

    def train(self, save_path: str = "..models/"):
        """
        Train TF-IDF and cosine similarity matrix
        """
        tfidf_matrix = self.tfidf.fit_transform(self.movies["genres"].fillna(""))
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(self.tfidf, f)
        with open(os.path.join(save_path, "cosine_sim_matrix.pkl"), "wb") as f:
            pickle.dump(self.cosine_sim, f)

    def recommend(self, movie_title: str, top_n: int = 10):
        """
        Recommend similar movies by content
        """
        indices = pd.Series(self.movies.index, index=self.movies["title"]).drop_duplicates()
        if movie_title not in indices:
            return f"Movie '{movie_title}' not found in database."

        idx = indices[movie_title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies.iloc[movie_indices][["movieId", "title", "genres"]]