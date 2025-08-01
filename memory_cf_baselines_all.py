
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, pairwise_distances

def jaccard_similarity_matrix(binary_matrix):
    return 1 - pairwise_distances(binary_matrix, metric='jaccard')

def cosine_similarity_matrix(matrix):
    return cosine_similarity(matrix)

def bhattacharyya_similarity_matrix(rating_matrix, alpha=0.75):
    n_users = rating_matrix.shape[0]
    sim_matrix = np.zeros((n_users, n_users))

    norm_ratings = rating_matrix / (np.linalg.norm(rating_matrix, axis=1, keepdims=True) + 1e-10)
    for i in range(n_users):
        for j in range(i + 1, n_users):
            bc = np.sum(np.sqrt(norm_ratings[i] * norm_ratings[j]))
            penalty = np.minimum(1, alpha * (np.count_nonzero(rating_matrix[i]) + np.count_nonzero(rating_matrix[j])) / (2 * rating_matrix.shape[1]))
            sim = bc * penalty
            sim_matrix[i, j] = sim_matrix[j, i] = sim
    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix

def predict_ratings(user_index, sim_matrix, rating_matrix, k=10):
    similar_users = np.argsort(sim_matrix[user_index])[::-1]
    similar_users = [u for u in similar_users if u != user_index][:k]
    weights = sim_matrix[user_index, similar_users]
    neighbors_ratings = rating_matrix[similar_users]
    weighted_sum = np.dot(weights, neighbors_ratings)
    sim_sum = np.sum(weights)
    if sim_sum > 0:
        return weighted_sum / sim_sum
    else:
        return np.zeros(rating_matrix.shape[1])

def evaluate_cf_baseline(rating_matrix, sim_func, k=10, **kwargs):
    n_users, n_items = rating_matrix.shape
    preds = []
    trues = []
    for user_idx in range(n_users):
        user_vector = rating_matrix[user_idx].copy()
        rated_items = np.where(user_vector > 0)[0]
        if len(rated_items) < 2:
            continue
        test_item = np.random.choice(rated_items)
        true_rating = user_vector[test_item]
        user_vector[test_item] = 0
        new_matrix = rating_matrix.copy()
        new_matrix[user_idx] = user_vector
        sim_matrix = sim_func(new_matrix, **kwargs)
        pred_vector = predict_ratings(user_idx, sim_matrix, new_matrix, k)
        preds.append(pred_vector[test_item])
        trues.append(true_rating)
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    return mae, rmse

def run_all_baselines(rating_matrix, k=10):
    bin_matrix = (rating_matrix > 0).astype(int)
    results = {}

    mae, rmse = evaluate_cf_baseline(rating_matrix.values, cosine_similarity_matrix, k)
    results['Cosine'] = {'MAE': mae, 'RMSE': rmse}

    mae, rmse = evaluate_cf_baseline(bin_matrix.values, jaccard_similarity_matrix, k)
    results['Jaccard'] = {'MAE': mae, 'RMSE': rmse}

    mae, rmse = evaluate_cf_baseline(rating_matrix.values, bhattacharyya_similarity_matrix, k, alpha=0.75)
    results['Penalized_BC'] = {'MAE': mae, 'RMSE': rmse}

    return pd.DataFrame(results).T
