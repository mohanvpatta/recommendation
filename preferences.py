import json
import numpy as np
import pandas as pd
from pprint import pprint

with open("data/movies.json", encoding="utf8") as f:
    movies = json.load(f)
    movies_actors = {}
    for movie in movies:
        movies_actors[movie["id"]] = movie["actors"]

u_data = np.loadtxt("100k/u.data", dtype=int)

num_users = np.max(u_data[:, 0])
num_items = np.max(u_data[:, 1])

movie_thumbnails = {
    "1": [
        [3, 2, 4],
        [31],
        [31, 4],
        [193],
        [14327],
    ]
}

with open("data/popular_actors.txt", "r") as f:
    popular_actors = f.read().splitlines()
    popular_actors = [int(actor) for actor in popular_actors]
    popular_actors = set(popular_actors)


def get_matrix(df, rows, cols):
    user_item_matrix = np.zeros((rows, cols))
    for row in df.itertuples():
        if row.rating != 0:
            user_item_matrix[row.user_id - 1, row.movie_id - 1] = row.rating

    user_item_matrix = np.insert(user_item_matrix, 0, 0, axis=0)
    user_item_matrix = np.insert(user_item_matrix, 0, 0, axis=1)

    return user_item_matrix


user_item_matrix_train = get_matrix(
    pd.read_csv(
        f"100k/u1.base",
        sep="	",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    ),
    num_users,
    num_items,
)


def get_user_actor_preferences(user_id, user_item_matrix):
    user_ratings = user_item_matrix[user_id, :]
    rated_movies = np.where(user_ratings != 0)[0]
    user_actor_preferences = {}

    for movie in rated_movies:
        if movie in movies_actors:
            for actor in movies_actors[movie]:
                actor_id = actor["id"]
                if actor_id in user_actor_preferences:
                    user_actor_preferences[actor_id].append(
                        int(user_item_matrix[user_id, movie])
                    )
                else:
                    user_actor_preferences[actor_id] = [
                        int(user_item_matrix[user_id, movie])
                    ]

    return user_actor_preferences


def get_thumbnail_weights(movie, movie_thumbnails, popular_actors, positive_actors):
    thumbnails = movie_thumbnails[str(movie)]

    POPULARITY_WEIGHT = 1
    POSITIVE_EXPOSURE_WEIGHT = 2

    thumbnail_weights = []

    for thumbnail in thumbnails:
        weights = []
        for person in thumbnail:
            if person in popular_actors:
                weights.append(POPULARITY_WEIGHT)
                print("popular", person)
            if person in positive_actors:
                weights.append(POSITIVE_EXPOSURE_WEIGHT)
                print("positive", person)
            else:
                weights.append(0.25)

        weights.sort(reverse=True)

        adjusted_weights = 0
        for i in range(len(weights)):
            adjusted_weights += weights[i] / (i + 1)

        thumbnail_weights.append(adjusted_weights)

    return thumbnail_weights


def get_positive_preferences(user_actor_preferences):
    positive_actors = set()
    for actor in user_actor_preferences:
        if not (
            np.mean(user_actor_preferences[actor]) < 2.5
            and len(user_actor_preferences[actor]) >= 5
        ):
            positive_actors.add(actor)

    return positive_actors


user_actor_preferences = get_user_actor_preferences(1, user_item_matrix_train)
positive_actors = get_positive_preferences(user_actor_preferences)

thumbnail_weights = get_thumbnail_weights(
    1, movie_thumbnails, popular_actors, positive_actors
)

pprint(thumbnail_weights)
