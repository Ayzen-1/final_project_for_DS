import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1516890751715-4d295e5a5b7f');
        background-size: cover;
        background-position: center;
        color: white;
    }
    .main-title {
        color: #FFDDC1;
        text-align: center;
        font-weight: bold;
        font-size: 2.5em;
        margin: 20px 0;
    }
    .recommendation-header {
        color: #FF4B4B;
        font-size: 24px;
        text-align: center;
        margin: 20px 0;
    }
    .st-table {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_data():
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    tags = pd.read_csv('ml-latest-small/tags.csv')
    return movies.copy(), ratings.copy(), tags.copy()

movies, ratings, tags = load_data()

ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})

movies['genres'] = movies['genres'].str.replace('|', ' ')
tags['tag'] = tags['tag'].fillna('')
tags_merged = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies = movies.merge(tags_merged, on='movieId', how='left')
movies['tag'] = movies['tag'].fillna('')
movies['metadata'] = movies['genres'] + ' ' + movies['tag']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['metadata'])

nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
nn_model.fit(tfidf_matrix)

def get_candidate_movies(movie_title, n_candidates=100):
    movie_genre = movies[movies['title'] == movie_title]['genres'].values[0]
    candidates = movies[movies['genres'].str.contains(movie_genre)]
    candidates = candidates.sort_values(by='movieId', ascending=False).head(n_candidates)
    return candidates

def get_hybrid_recommendations(userId, movie_title, n_candidates=100):
    candidates = get_candidate_movies(movie_title, n_candidates)
    candidate_indices = candidates.index
    query_index = movies[movies['title'] == movie_title].index[0]
    
    distances, indices = nn_model.kneighbors(tfidf_matrix[query_index], n_neighbors=n_candidates)
    
    similar_movie_ids = [movies['movieId'].iloc[i] for i in indices.flatten() if i in candidate_indices]
    return similar_movie_ids[:10]

user_id = st.sidebar.selectbox("Выберите ID пользователя", ratings['user_id'].unique())
movie_title = st.sidebar.selectbox("Выберите название фильма", movies['title'].unique())

if st.sidebar.button("Получить рекомендации"):
    recommendations = get_hybrid_recommendations(user_id, movie_title)
    
    st.markdown("<div class='recommendation-header'>Рекомендуемые фильмы</div>", unsafe_allow_html=True)
    
    recommendation_data = pd.DataFrame(
        {
            "Название": [movies[movies['movieId'] == movie_id]['title'].values[0] for movie_id in recommendations],
            "Жанры": [movies[movies['movieId'] == movie_id]['genres'].values[0] for movie_id in recommendations]
        }
    )
    
    st.table(recommendation_data)
st.sidebar.markdown(
    """
    <hr>
    <small>Разработано с помощью Streamlit. Код доступен на GitHub.</small>
    <br>
    <small>Фон: <a href="https://unsplash.com/photos/2vTQZp5e0H0" target="_blank" style="color: #FFDDC1;">Unsplash</a></small>
    """,
    unsafe_allow_html=True
)
