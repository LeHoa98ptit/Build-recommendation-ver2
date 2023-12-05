from flask import Flask, request, jsonify, make_response, render_template
import pandas as pd
from surprise import SVD, KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

# Load cleaned dataset
song_df = pd.read_csv("cleaned_song_dataset.csv")

# Create a temporary user id for new users
temp_user_id = 'temp_user'

# Collaborative Filtering Model
reader = Reader(rating_scale=(0, song_df['play_count'].max()))
data = Dataset.load_from_df(song_df[['user', 'title', 'play_count']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
collab_model = SVD()
collab_model.fit(trainset)

# Content-Based Model (using TF-IDF on song titles)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(song_df['title'])

# Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', songs=song_df['title'])


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    # Get selected songs from the request

    song_ids = request.get_json().get('selected_songs', [])

    # Lọc dữ liệu từ song_df
    selected_songs = song_df[song_df['song'].isin(song_ids)]['title'].values.tolist()

    # Get hybrid recommendations
    hybrid_recommendations = get_hybrid_recommendations(temp_user_id, selected_songs)

    return make_response(jsonify({
        "recommendations": hybrid_recommendations
    }))


# Content-Based Filtering (TF-IDF on song titles)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(song_df['title'])


# Function to get collaborative filtering recommendations
def get_collab_recommendations(user_id, df, songs_listened, top_n=5):
    temp_user_df = pd.DataFrame({'user': [temp_user_id] * len(songs_listened), 'title': songs_listened,
                                 'play_count': [1] * len(songs_listened)})

    song_df1 = pd.concat([df, temp_user_df])

    # Collaborative Filtering (SVD)
    reader = Reader(rating_scale=(0, song_df1['play_count'].max()))
    data = Dataset.load_from_df(song_df1[['user', 'title', 'play_count']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

    # Build and train the collaborative filtering model (SVD)
    collab_model = SVD()
    collab_model.fit(trainset)

    all_songs = song_df1['title'].unique()
    songs_user_has_listened = set(songs_listened)
    songs_to_recommend = list(set(all_songs) - songs_user_has_listened)

    testset_user = [(user_id, song, 0) for song in songs_to_recommend]
    predictions = collab_model.test(testset_user)

    top_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]
    return [(recommendation.iid, recommendation.est) for recommendation in top_recommendations]


# Function to get content-based filtering recommendations
def get_content_recommendations(songs_listened, df, top_n=5, min_listen_threshold=5):
    all_songs = df['title'].unique()
    songs_user_has_listened = set(songs_listened)
    songs_to_recommend = list(set(all_songs) - songs_user_has_listened)

    # Fit the TF-IDF vectorizer on the titles of the songs the user has listened to
    tfidf_vectorizer.fit(songs_listened)

    # Transform the titles of all songs to TF-IDF vectors
    tfidf_matrix_user = tfidf_vectorizer.transform(songs_listened)
    tfidf_matrix_all = tfidf_vectorizer.transform(df['title'])

    # Compute similarity scores using cosine similarity
    content_scores = linear_kernel(tfidf_matrix_user, tfidf_matrix_all).mean(axis=0)

    content_predictions = list(enumerate(content_scores))

    # Sort content-based predictions by score in descending order
    sorted_indices = sorted(content_predictions, key=lambda x: x[1], reverse=True)

    # Get the top recommended songs
    top_recommendations = [list(df['title'])[idx] for idx, _ in sorted_indices[:top_n]]
    return [(song, content_scores[df[df['title'] == song].index[0]]) for song in top_recommendations]


# Function to get hybrid recommendations
def get_hybrid_recommendations(user_id, songs_listened, top_n=5, min_listen_threshold=5):
    # Collaborative filtering predictions
    collab_predictions = get_collab_recommendations(user_id, song_df, songs_listened, top_n=5)

    # Content-based filtering predictions
    content_recommendations = get_content_recommendations(songs_listened, song_df, top_n, min_listen_threshold)

    # Combine predictions from both models (simple average here)
    hybrid_predictions = collab_predictions + content_recommendations

    # Filter out songs the user has already listened to
    songs_user_has_listened = set(songs_listened)
    hybrid_predictions = [(song, score) for song, score in hybrid_predictions if song not in songs_user_has_listened]

    # Get the top recommended songs
    top_recommendations = sorted(hybrid_predictions,
                                 key=lambda x: (float('-inf') if isinstance(x[1], str) else float(x[1])),
                                 reverse=True)[
                          :top_n]

    return top_recommendations


@app.route('/get_songs')
def get_songs():
    songs = song_df[['song', 'title']].drop_duplicates()
    return jsonify({
        'songs': songs['song'].tolist(),
        'title': songs['title'].tolist()
    })


if __name__ == '__main__':
    app.run(debug=True)
