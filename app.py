import requests
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TMDB_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("AI Movie Recommendation System")
st.write("Search for a movie and get similar movie recommendations.")


def get_api_key():
    return st.secrets["TMDB_API_KEY"]


def search_movie(movie_name):
    url = f"{TMDB_BASE_URL}/search/movie"

    params = {
        "api_key": get_api_key(),
        "query": movie_name
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data["results"]


def get_recommendations(movie_id):
    url = f"{TMDB_BASE_URL}/movie/{movie_id}/recommendations"

    params = {
        "api_key": get_api_key()
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data["results"]


def get_poster_url(poster_path):
    if poster_path:
        return IMAGE_BASE_URL + poster_path
    return None


def ai_rank_movies(selected_movie, recommended_movies):
    movie_rows = []

    movie_rows.append({
        "title": selected_movie["title"],
        "overview": selected_movie.get("overview", ""),
        "release_date": selected_movie.get("release_date", "")
    })

    for movie in recommended_movies:
        movie_rows.append({
            "title": movie["title"],
            "overview": movie.get("overview", ""),
            "release_date": movie.get("release_date", "")
        })

    df = pd.DataFrame(movie_rows)

    df["features"] = (
        df["title"].fillna("") + " " +
        df["overview"].fillna("") + " " +
        df["release_date"].fillna("")
    )

    cv = CountVectorizer(stop_words="english")
    vectors = cv.fit_transform(df["features"])

    similarity = cosine_similarity(vectors)

    scores = list(enumerate(similarity[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    ranked_movies = []

    for index, score in scores[1:6]:
        ranked_movies.append(recommended_movies[index - 1])

    return ranked_movies


movie_name = st.text_input("Enter a movie name", placeholder="Example: Interstellar")

if movie_name:
    try:
        search_results = search_movie(movie_name)

        if len(search_results) == 0:
            st.warning("No movie found.")
        else:
            selected_movie = search_results[0]

            st.subheader(selected_movie["title"])

            col1, col2 = st.columns([1, 2])

            with col1:
                poster_url = get_poster_url(selected_movie.get("poster_path"))
                if poster_url:
                    st.image(poster_url)

            with col2:
                st.write(selected_movie.get("overview", "No description available."))
                st.write("Rating:", selected_movie.get("vote_average", "N/A"))
                st.write("Release Date:", selected_movie.get("release_date", "N/A"))

            recommended_movies = get_recommendations(selected_movie["id"])

            if len(recommended_movies) == 0:
                st.info("No recommendations found.")
            else:
                ranked_movies = ai_rank_movies(selected_movie, recommended_movies)

                st.subheader("Recommended Movies")

                cols = st.columns(5)

                for i, movie in enumerate(ranked_movies):
                    with cols[i]:
                        poster_url = get_poster_url(movie.get("poster_path"))

                        if poster_url:
                            st.image(poster_url)

                        st.write(movie["title"])
                        st.caption(f"Rating: {movie.get('vote_average', 'N/A')}")

    except Exception as e:
        st.error(f"Something went wrong: {e}")