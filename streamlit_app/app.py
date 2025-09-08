import streamlit as st
import pandas as pd
import pickle
from pathlib import Path


MOVIES_CSV = Path("../data/movies.csv")
SIM_MATRIX_PKL = Path("../models/cosine_sim_matrix.pkl")
SVD_MODEL_PKL = Path("../models/svd_model.pkl")

@st.cache_resource
def load_models():
    df = pd.read_csv(MOVIES_CSV)

    with open(SIM_MATRIX_PKL, "rb") as f:
        cosine_sim = pickle.load(f)

    with open(SVD_MODEL_PKL, "rb") as f:
        svd_model = pickle.load(f)

    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return df, cosine_sim, indices, svd_model

df, cosine_sim, indices, svd_model = load_models()

# Content-based recommendation
def recommend_content(title, top_n=10):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][["movieId", "title"]]

# Hybrid recommendation
def recommend_hybrid(user_id, title, top_n=10):
    content_df = recommend_content(title, top_n=20)
    content_df["cf_score"] = content_df["movieId"].apply(
        lambda x: svd_model.predict(user_id, x).est
    )
    content_df["final_score"] = (content_df.index.to_series().rank(ascending=False) + content_df["cf_score"]) / 2
    return content_df.sort_values("final_score", ascending=False).head(top_n)

# Streamlit UI
st.title("🎥 Hybrid Movie Recommendation System")
user_id = st.number_input("User ID kiriting:", min_value=1, value=1)
selected_movie = st.selectbox("Filmni tanlang:", sorted(df["title"].unique()))


if st.button("Tavsiyalarni ko‘rish"):
    recommendations = recommend_hybrid(user_id, selected_movie, top_n=10)
    st.subheader(f"'{selected_movie}' uchun tavsiyalar:")
    for i, row in recommendations.iterrows():
        st.write(f"🎬 {row['title']} (CF Score: {row['cf_score']:.2f})")

