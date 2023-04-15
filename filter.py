import pandas as pd
import numpy as np
import numpy.ma as ma
from datetime import datetime
import warnings


u_data = np.loadtxt("100k/u.data", dtype=int)

num_users = np.max(u_data[:, 0])
num_items = np.max(u_data[:, 1])


def get_matrix(df, rows, cols):
    user_item_matrix = np.zeros((rows, cols))
    for row in df.itertuples():
        if row.rating != 0:
            user_item_matrix[row.user_id - 1, row.movie_id - 1] = row.rating

    user_item_matrix = np.insert(user_item_matrix, 0, 0, axis=0)
    user_item_matrix = np.insert(user_item_matrix, 0, 0, axis=1)

    return user_item_matrix


def calculate_rmse_mae(predicted_ratings, actual_ratings):
    nz_indices = actual_ratings.nonzero()
    nz_actual = actual_ratings[nz_indices]
    nz_predicted = predicted_ratings[nz_indices]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mse = np.mean((nz_predicted - nz_actual) ** 2)
        rmse = np.sqrt(mse)

        mae = np.mean(np.abs(nz_predicted - nz_actual))

        return [rmse, mae]


def predict_ratings(user_id, user_item_matrix, user_similarity_matrix):
    top_n_users = 400

    # For each item, calculate the weighted sum of ratings by similar users
    predicted_ratings = np.zeros(user_item_matrix.shape[1])
    for item in range(1, user_item_matrix.shape[1]):
        # Retrieve the ratings for the item by all users
        ratings = user_item_matrix[:, item]

        # Identify the similar users to the target user based on the similarity matrix
        similarities = user_similarity_matrix[user_id, :]
        similar_users = np.argsort(similarities)[::-1][1:top_n_users]

        # Calculate the weighted sum of ratings by similar users for the item
        masked_ratings = ma.masked_array(ratings, mask=(ratings == 0))
        masked_similarities = ma.masked_array(
            similarities[similar_users], mask=(ratings[similar_users] == 0)
        )

        weighted_ratings = ma.dot(masked_similarities, masked_ratings[similar_users])
        similarity_weights = np.sum(masked_similarities)

        # Predict the rating for the item by the target user
        if similarity_weights > 0:
            predicted_rating = weighted_ratings / similarity_weights
        else:
            predicted_rating = 0
        predicted_ratings[item] = predicted_rating

    return predicted_ratings


def triangle_similarity(ratings_m, ratings_n):
    # Identify the items rated by both users
    nz_indices_m = ratings_m.nonzero()
    nz_indices_n = ratings_n.nonzero()
    nz_indices = np.intersect1d(nz_indices_m, nz_indices_n)

    # Calculate the similarity between the two users
    if len(nz_indices) < 0:
        return 0
    ratings_m = ratings_m[nz_indices]
    ratings_n = ratings_n[nz_indices]

    numerator = np.sum(np.subtract(ratings_m, ratings_n) ** 2) ** 0.5
    denominator = np.sum((ratings_m) ** 2) ** 0.5 + np.sum((ratings_n) ** 2) ** 0.5

    if denominator == 0:
        return 0

    return 1 - (numerator / denominator)


def calculate_user_similarity_matrix(user_item_matrix, index):
    user_similarity_matrix = np.zeros(
        (user_item_matrix.shape[0], user_item_matrix.shape[0])
    )

    # normalize the user-item matrix by subtracting the mean rating of each user
    row_means = user_item_matrix.mean(axis=1)
    row_stdevs = user_item_matrix.std(axis=1, ddof=1)
    row_stdevs[row_stdevs == 0] = 1
    user_item_matrix_centered = user_item_matrix - row_means.reshape(-1, 1)
    user_item_matrix_normalized = user_item_matrix_centered / row_stdevs.reshape(-1, 1)

    x_min = np.min(user_item_matrix_normalized)
    x_max = np.max(user_item_matrix_normalized)

    user_item_matrix = (user_item_matrix_normalized - x_min) / (x_max - x_min)

    for i in range(1, user_item_matrix.shape[0]):
        for j in range(i + 1, user_item_matrix.shape[0]):
            user_similarity_matrix[i, j] = triangle_similarity(
                user_item_matrix[i], user_item_matrix[j]
            )
            user_similarity_matrix[j, i] = user_similarity_matrix[i, j]

    print(f"Completed calculating similarity matrix for u{index}")

    # save the matrix in a csv file
    pd.DataFrame(user_similarity_matrix).to_csv(
        f"similarity_matrix_u{index}.csv", index=False, header=False
    )

    return user_similarity_matrix


def calculate_res(res, label):
    rmse = np.mean(res["rmse"])
    mae = np.mean(res["mae"])

    print(f"{label} Results:")
    print(f"RMSE: {np.mean(res['rmse'])}")
    print(f"MAE: {np.mean(res['mae'])}")
    print(f"Time taken: {res['end'] - res['start']}")

    return [rmse, mae]


def run_benchmark():
    total_res = {
        "rmse": [],
        "mae": [],
        "start": datetime.now(),
        "end": None,
    }

    for index in range(1, 6):
        [base, test] = [
            pd.read_csv(
                f"100k/u{index}.base",
                sep="	",
                header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
                engine="python",
            ),
            pd.read_csv(
                f"100k/u{index}.test",
                sep="	",
                header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
                engine="python",
            ),
        ]

        user_similarity_matrix = calculate_user_similarity_matrix(
            get_matrix(
                base,
                num_users,
                num_items,
            ),
            index,
        )

        user_item_matrix_train = get_matrix(
            base,
            num_users,
            num_items,
        )

        user_item_matrix_test = get_matrix(
            test,
            num_users,
            num_items,
        )

        sample_res = {
            "rmse": [],
            "mae": [],
            "start": datetime.now(),
            "end": None,
        }

        for user in range(1, user_item_matrix_test.shape[0]):
            predicted_ratings = predict_ratings(
                user, user_item_matrix_train, user_similarity_matrix
            )
            [rmse, mae] = calculate_rmse_mae(
                predicted_ratings, user_item_matrix_test[user]
            )

            if not np.isnan(rmse) and not np.isnan(mae):
                sample_res["rmse"].append(rmse)
                sample_res["mae"].append(mae)

        sample_res["end"] = datetime.now()
        [sample_rmse_avg, sample_mae_avg] = calculate_res(sample_res, f"u{index}")

        total_res["rmse"].append(sample_rmse_avg)
        total_res["mae"].append(sample_mae_avg)

    total_res["end"] = datetime.now()

    calculate_res(total_res, "Total")


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
