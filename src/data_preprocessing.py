import pandas as pd
import os

def load_data(movie_path = "..data/movies.csv", ratings_path = "..data/ratings.csv"):
    """
    Load movies and ratings datasets
    """
    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


def preprocess_movies(movies: pd.DataFrame):
    """
    Basic preprocessing on movies dataset
    """
    movies.dropna(inplace=True)
    return movies


def preprocess_ratings(ratings: pd.DataFrame):
    """
    Basic preprocessing on ratings dataset
    """
    ratings.dropna(inplace=True)
    ratings["rating"] = ratings["rating"].astype(float)
    return ratings