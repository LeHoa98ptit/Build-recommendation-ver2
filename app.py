from flask import Flask, request, jsonify, make_response, render_template
import pandas as pd
from surprise import SVD, dump
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

# Load cleaned dataset
song_df = pd.read_csv("clean_song_dataset_0_log.csv")

# Create a temporary user id for new users
temp_user_id = 'temp_user'

# Collaborative Filtering Model - Load the model
collab_model = dump.load("model/collab_model.pkl")[1]

# Content-Based Model (using TF-IDF on song titles)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(song_df[['title', "artist_name", 'release']])


# Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', songs=song_df['title'])


@app.route('/get_songs')
def get_songs():
    songs = song_df[['song', 'title', 'artist_name', 'release']].drop_duplicates()
    return jsonify({
        'songs': songs.to_dict(orient='records')
    })


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    # Get selected songs from the request
    song_ids = request.get_json().get('selected_songs', [])

    # check number of songs that user selected, if number_song < threshold => return top 10 most populer song
    num_song = len(song_ids)
    if num_song < 5:
        top_10_songs = get_top_song()
        top_songs_info = song_df[song_df["song"].isin(top_10_songs.index)][['title', 'artist_name']].drop_duplicates()
        top_songs_info["Score"] = "Top_10_Most_Popular_Song"
        return make_response(jsonify({
            "recommendations": top_songs_info.to_dict(orient='records')
        }))

    # Get hybrid recommendations
    hybrid_recommendations = get_hybrid_recommendations(song_ids)
    df_hybrid = pd.DataFrame(hybrid_recommendations, columns=["song", "Score"])
    # Merge/join the two DataFrames on the 'song_id' column
    result_df = pd.merge(df_hybrid, song_df, on='song')[['title', 'artist_name', "Score"]].drop_duplicates()

    return make_response(jsonify({
        "recommendations": result_df.to_dict(orient='records')
    }))


# Function to get collaborative filtering recommendations
def get_collab_recommendations(songs_listened):
    all_songs = song_df['song'].unique()
    songs_user_has_listened = set(songs_listened)
    songs_to_recommend = list(set(all_songs) - songs_user_has_listened)

    recommendations = [(item, collab_model.predict(temp_user_id, item).est) for item in songs_to_recommend]
    top_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

    return [(recommendation[0], recommendation[1]) for recommendation in top_recommendations]


# Function to get content-based filtering recommendations
def get_content_recommendations(songs_listened, df):
    songs_listened = pd.DataFrame(songs_listened, columns=["song"])
    # Merge songs_listened với song_df để lấy ra data các bài hát mà người dùng đã nghe
    user_listened_data = pd.merge(songs_listened, song_df, on='song', how='inner')[['title', "artist_name", 'release']]

    # Fit the TF-IDF vectorizer on the titles of the songs the user has listened to
    tfidf_vectorizer.fit(user_listened_data[['title', "artist_name", 'release']])

    # Transform the titles of all songs to TF-IDF vectors
    tfidf_matrix_user = tfidf_vectorizer.transform(user_listened_data[['title', "artist_name", 'release']])
    tfidf_matrix_all = tfidf_vectorizer.transform(df[['title', "artist_name", 'release']])

    # Compute similarity scores using cosine similarity
    content_scores = linear_kernel(tfidf_matrix_user, tfidf_matrix_all).mean(axis=0)

    content_predictions = list(enumerate(content_scores))

    # Sort content-based predictions by score in descending order
    sorted_indices = sorted(content_predictions, key=lambda x: x[1], reverse=True)

    # Get the top recommended songs
    top_recommendations = [list(df['song'])[idx] for idx, _ in sorted_indices[:10]]
    return [(song, content_scores[df[df['song'] == song].index[0]]) for song in top_recommendations]


# Function to get hybrid recommendations
def get_hybrid_recommendations(songs_listened, top_n=10):
    # Collaborative filtering predictions
    collab_predictions = get_collab_recommendations(songs_listened)

    # Content-based filtering predictions
    content_recommendations = get_content_recommendations(songs_listened, song_df)

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


def get_top_song():
    # Calculate the total number of listens for each song
    total_listen_count = song_df.groupby('song')['play_count'].sum()

    # sorted by total number of listens in descending order
    most_listened_songs = total_listen_count.sort_values(ascending=False)

    # Displays the 10 most listened to songs
    top_10_songs = pd.DataFrame(most_listened_songs.nlargest(10))

    return top_10_songs


if __name__ == '__main__':
    app.run(debug=True)
