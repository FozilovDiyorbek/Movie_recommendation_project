import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

class CollaborativeFilteringRecommender:
    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings
        self.model = None

    def train(self):
        """
        Train SVD collaborative filtering model
        """
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings[["userId", "movieId", "rating"]], reader)
        trainset, testset = train_test_split(data, test_size=0.2)
        self.model = SVD()
        self.model.fit(trainset)
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=True)
        return rmse

    def recommend(self, user_id: int, movies: pd.DataFrame, top_n: int = 10):
        """
        Recommend top-N movies for a given user
        """
        all_movie_ids = movies["movieId"].unique()
        user_movies = self.ratings[self.ratings["userId"] == user_id]["movieId"].tolist()
        movie_candidates = [m for m in all_movie_ids if m not in user_movies]

        predictions = [(movie_id, self.model.predict(user_id, movie_id).est) for movie_id in movie_candidates]
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
        recommended_ids = [p[0] for p in predictions]
        return movies[movies["movieId"].isin(recommended_ids)][["movieId", "title", "genres"]]
