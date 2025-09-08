# 🎬 Movie Recommendation System

This project is a **Movie Recommendation System** built using the **MovieLens dataset**.  
It implements **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid Approach** to recommend movies.  
Additionally, the project provides both a **Streamlit web app** and a **FastAPI backend API** for deployment.  

---

## 🚀 Features
- **Content-Based Filtering** → Recommends movies similar to a given movie (based on TF-IDF + cosine similarity).
- **Collaborative Filtering** → Recommends movies based on user ratings (using SVD from `surprise`).
- **Hybrid Model** → Combines both approaches with a weighted score (`alpha` parameter).
- **Web App (Streamlit)** → User-friendly UI to test recommendations.
- **REST API (FastAPI)** → Provides endpoints for integration with other apps.

---

## 📂 Project Structure
