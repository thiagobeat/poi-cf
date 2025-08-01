from core import config, utils
from poi import poi as poi
from similatiry import measures as sm

from scipy.sparse import csr_matrix
import datetime as dt
from scipy import spatial
from scipy import sparse
import numpy as np
import pandas as pd
import jenkspy
import random

from recommender_system import my_jenks_breaks as jb, my_quantile_transform as qt, my_kmeans_breaks as kb
from recommender_system import preprocessing, evaluation_metrics, evaluation_plots

from sklearn.neighbors import NearestNeighbors
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity

import math
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# import sys
# this_module = sys.modules[__name__]

save_similatiries_file = True


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def adjusted_cosine(np_ratings, nb_users, dataset_name, ratings):
    similarities = np.zeros(shape=(nb_users, nb_users))
    similarities.fill(-1)

    def _progress(count):
        sys.stdout.write('\rComputing similarities. Progress status : %.1f%%' % (float(count / nb_users) * 100.0))
        sys.stdout.flush()

    users = sorted(ratings.user_id.unique())
    for i in users[:-1]:
        for j in users[i + 1:]:
            scores = np_ratings[(np_ratings[:, 1] == i) | (np_ratings[:, 1] == j), :]
            vals, count = np.unique(scores[:, 0], return_counts=True)
            scores = scores[np.isin(scores[:, 0], vals[count > 1]), :]

            if scores.shape[0] > 2:
                x = scores[scores[:, 1].astype('float') == i, 4]
                y = scores[scores[:, 1].astype('float') == j, 4]
                w = cosine(x, y)

                similarities[i, j] = w
                similarities[j, i] = w
        _progress(i)
    _progress(nb_users)

    # get neighbors by their neighbors in decreasing order of similarities
    neighbors = np.flip(np.argsort(similarities), axis=1)

    # sort similarities in decreasing order
    similarities = np.flip(np.sort(similarities), axis=1)

    # save similarities to disk
    if save_similatiries_file:
        save_similarities(similarities, neighbors, dataset_name=dataset_name, base_dir='user2user')

    return similarities, neighbors


def calculate_similarity_matrix(ratings_matrix, n_neighbors, ratings, metric='bhattacharyya', use_penalization=False):
    data, col, row = [], [], []

    def _progress(count):
        sys.stdout.write('\rComputing similarities. Progress status : %.1f%%' % (float(count / ratings.user_id.nunique()) * 100.0))
        sys.stdout.flush()

    users = sorted(range(ratings_matrix.shape[0]))
    #users = sorted(ratings.user_id.unique())

    for i in users:
        for j in users:

            if i == j:
                continue

                # print("values for for i:{} x j:{}".format(i,j))

            scores_i = ratings_matrix[i, :]
            scores_j = ratings_matrix[j, :]

            w = 0

            if metric == 'bhattacharyya':
                if use_penalization:
                    w = sm.get_SBCF_adjustment_factor_similarity_array(scores_i, scores_j)
                else:
                    w = sm.get_SBCF_similarity_array(scores_i, scores_j)
            elif metric == 'SBCF_adj':
                w = sm.SBCF_adj(scores_i, scores_j)
            elif metric == 'correlation':
                w = sm.pearson_array(scores_i, scores_j, compress_items=True)
            elif metric == 'jaccard':
                w = sm.jacc_array(scores_i, scores_j)
            elif metric == 'cosine_adj':
                w = sm.CSA_adj(scores_i, scores_j)

            if np.isnan(w):
                w = 0

            row.append(i)
            col.append(j)
            data.append(w)

        _progress(i)
    _progress(ratings.user_id.nunique())

    from scipy import sparse
    sim_matrix = sparse.coo_matrix((data, (row, col)), shape=(len(users), len(users))).toarray()

    return sim_matrix


def find_n_similarities(bhatta_similarities_matrix, n):
    if bhatta_similarities_matrix.shape[0] <= n:
        n = bhatta_similarities_matrix.shape[0]

    # order = np.argsort(bhatta_similarities_matrix.values, axis=1)[:, :n]
    df_similarities = bhatta_similarities_matrix.apply(lambda x: pd.Series(x.sort_values(ascending=False)
                                                                           .iloc[:n].values[0],
                                                                           index=['{}'.format(i) for i in
                                                                                  range(1, n + 1)]), axis=1)

    return df_similarities


def find_n_neighbours(bhatta_similarities_matrix, n):
    if bhatta_similarities_matrix.shape[0] <= n:
        n = bhatta_similarities_matrix.shape[0]

    # order = np.argsort(df.values, axis=1)[:, :n]
    df_neighbours = bhatta_similarities_matrix.apply(lambda x: pd.Series(x.sort_values(ascending=False)
                                                                         .iloc[:n].index,
                                                                         index=['{}'.format(i) for i in
                                                                                range(1, n + 1)]), axis=1)

    return df_neighbours


def find_candidate_items(user_id, n, neighbors, ratings):
    """
    Find candidate items for an active user

    :param user_id : active user
    :param n : number of candidates to evaluate
    :param neighbors : users similar to the active user
    :param ratings: ratings to get similar candidates
    :return candidates : top n of candidate items
    """
    user_neighbors = neighbors[user_id]
    activities = ratings.loc[ratings.user_id.isin(user_neighbors)]

    # sort items in decreasing order of frequency (put the most popular items on top)
    frequency = activities.groupby('item_id')['rating'].count().reset_index(name='count').sort_values(['count'],
                                                                                                      ascending=False)
    # get the item ids for the items in the group of similar users to the current user
    group_user_items = frequency.item_id
    # get the ratings made by the current user
    active_items = ratings.loc[ratings.user_id == user_id].item_id.to_list()
    # use only the items that are not in the active group of items for the current user
    candidates = np.setdiff1d(group_user_items, active_items, assume_unique=True)[:n]

    return candidates


def predict(user_id, item_id, neighbors, similarities, np_ratings, mean):
    """
    predict what score user_id would have given to item_id.

    :param
        - user_id : user id for which we want to make prediction
        - item_id : item id on which we want to make prediction

    :return
        - r_hat : predicted rating of user user_id on item item_id
    """
    user_similarities = similarities[user_id]
    user_neighbors = neighbors[user_id]
    # get mean rating of user user_id
    user_mean = mean[user_id]

    # find users who rated item 'item_id'
    iratings = np_ratings[np_ratings[:, 1].astype('int') == item_id]

    # find similar users to 'user_id' who rated item 'item_id'
    suri = iratings[np.isin(iratings[:, 0], user_neighbors)]

    # similar users who rated current item (surci)
    normalized_ratings = suri[:, 4]
    indexes = [np.where(user_neighbors == uid)[0][0] for uid in suri[:, 0].astype('int')]
    sims = user_similarities[indexes]

    num = np.dot(normalized_ratings, sims)
    den = np.sum(np.abs(sims))

    if num == 0 or den == 0:
        return user_mean

    r_hat = user_mean + np.dot(normalized_ratings, sims) / np.sum(np.abs(sims))

    return r_hat


def user2userPredictions(user_id, pred_path, max_num_recommendations, neighbors, similarities, np_ratings, mean,
                         ratings):
    """
    Make rating prediction for the active user on each candidate item and save in file prediction.csv

    :param
        - user_id : id of the active user
        - pred_path : where to save predictions
    """
    # find candidate items for the active user
    candidates = find_candidate_items(user_id, max_num_recommendations, neighbors, ratings)

    # loop over candidates items to make predictions
    for item_id in candidates:
        # prediction for user_id on item_id
        r_hat = predict(user_id, item_id, neighbors, similarities, np_ratings, mean)

        if save_similatiries_file:
            # save predictions
            with open(pred_path, 'a+') as file:
                line = '{},{},{}\n'.format(user_id, item_id, r_hat)
                file.write(line)


def user2userRecommendation(user_id, user_encoder, item_encoder, pois, return_pois_merge=False):
    """
    """
    # encode the user_id
    uid = user_encoder.transform([user_id])[0]
    saved_predictions = 'predictions.csv'

    predictions = pd.read_csv(saved_predictions, sep=',', names=['user_id', 'item_id', 'predicted_rating'])
    predictions = predictions[predictions.user_id == uid]
    predictions_list = predictions.sort_values(by=['predicted_rating'], ascending=False)

    predictions_list.user_id = user_encoder.inverse_transform(predictions_list.user_id.tolist())
    predictions_list.item_id = item_encoder.inverse_transform(predictions_list.item_id.tolist())

    if return_pois_merge:
        predictions_list = pd.merge(predictions_list, pois, on='item_id', how='inner')

    return predictions_list


def user2userCF(ratings, max_num_recommendations, neighbors, similarities, np_ratings, mean):
    """
    Make predictions for each user in the database.
    """
    # get list of users in the database
    users = ratings.user_id.unique()

    def _progress(count):
        sys.stdout.write('\rRating predictions. Progress status : %.1f%%' % (float(count / len(users)) * 100.0))
        sys.stdout.flush()

    saved_predictions = 'predictions.csv'
    if os.path.exists(saved_predictions):
        os.remove(saved_predictions)

    for count, user_id in enumerate(users):
        # make rating predictions for the current user
        user2userPredictions(user_id, saved_predictions, max_num_recommendations, neighbors, similarities, np_ratings,
                             mean, ratings)
        _progress(count)


def save_similarities(similarities, neighbors, dataset_name, base_dir='item2item'):
    save_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    similarities_file_name = os.path.join(save_dir, 'similarities.npy')
    neighbors_file_name = os.path.join(save_dir, 'neighbors.npy')
    try:
        np.save(similarities_file_name, similarities)
        np.save(neighbors_file_name, neighbors)
    except ValueError as error:
        print(f"An error occured when saving similarities, due to : \n ValueError : {error}")


def load_similarities(dataset_name, k=20, base_dir='item2item'):
    save_dir = os.path.join(base_dir, dataset_name)
    similarities_file = os.path.join(save_dir, 'similarities.npy')
    neighbors_file = os.path.join(save_dir, 'neighbors.npy')
    similarities = np.load(similarities_file)
    neighbors = np.load(neighbors_file)
    return similarities[:, :k], neighbors[:, :k]


def create_simple_nearest_neighbors_model(rating_matrix, k=20, metric="cosine"):
    """
    :param rating_matrix:
    :param k : number of nearest neighbors to return
    :param metric: metric to be used to the model
    :return model : our knn model
    """
    model = NearestNeighbors(metric=metric, n_neighbors=k + 1, algorithm='brute')
    model.fit(rating_matrix)
    return model


def get_nearest_neighbors(rating_matrix, model):
    """
    :param rating_matrix : rating matrix of shape (nb_users, nb_items)
    :param model : nearest neighbors model
    :return
        - similarities : distances of the neighbors from the referenced user
        - neighbors : neighbors of the referenced user in decreasing order of similarities
    """
    similarities, neighbors = model.kneighbors(rating_matrix)
    return similarities[:, 1:], neighbors[:, 1:]


def build_user_ratings_matrix_OLD(df_ids, df_stay_points, df_pois):
    calculate_ratings = False

    df_ratings = pd.DataFrame(columns=config.HEADER_RECOMMENDER_SYSTEM_RATINGS)

    df_stay_points['cut_jenks'] = 0

    if calculate_ratings:
        # Builds the user profile accordingly to the available points of interest
        for user_id in df_ids[df_ids.directory.isin(df_stay_points.user_id)].directory.sort_values(
                ascending=True).unique():

            print("Building rating matrix - User {} --- started in {}".format(user_id, dt.datetime.now()))

            # Here we iterate over all the pois in order to flag if the current user has or not visited it
            for poi_id, cluster_id in \
                    zip(df_pois[df_pois.poi_id.isin(df_stay_points[(df_stay_points.user_id == user_id)].poi_id)][
                            ['poi_id', 'cluster_id']].sort_values(by='poi_id', ascending=True).poi_id.unique(),
                        df_pois[df_pois.poi_id.isin(df_stay_points[(df_stay_points.user_id == user_id)].poi_id)][
                            ['poi_id', 'cluster_id']].sort_values(by='poi_id', ascending=True).cluster_id.unique()):

                # print("POI {} --- started in {}".format(poi_id, dt.datetime.now()))

                if False:
                    boolean_utility_matrix[user_id, poi_id] = len(df_stay_points[(
                            (df_stay_points.user_id == user_id) & (df_stay_points.cluster_id == cluster_id))]) > 0

                # Here we put the weigh as rating for the user and the POI following the formula:
                # (user_quantity_visits_poi * user_time_spent_poi) / (total_visits_poi * total_stay_time_poi)
                # by using this formula we penalize those users which stay fewer time and make less visits.
                user_visits_poi = len(
                    df_stay_points[((df_stay_points.user_id == user_id) & (df_stay_points.cluster_id == cluster_id))])
                user_time_visits_poi = df_stay_points[((df_stay_points.user_id == user_id) & (
                        df_stay_points.cluster_id == cluster_id))].stay_time.sum().total_seconds()

                # Here we enforce at least quantity_visits time to the poi as the user had to be there at least for a second for it to count
                if user_time_visits_poi == 0:
                    user_time_visits_poi = user_visits_poi

                total_visits_user = len(
                    df_stay_points[(df_stay_points.user_id == user_id)])  # .stay_time.sum().total_seconds()
                total_visits_poi = len(df_stay_points[(df_stay_points.cluster_id == cluster_id)])

                user_id_x_poi_id_x_rating = sm.get_UIR(user_visits_poi, user_time_visits_poi, total_visits_user,
                                                       total_visits_poi)

                df_stay_points.loc[((df_stay_points.cluster_id == cluster_id) & (
                        df_stay_points.user_id == user_id)), 'rating'] = user_id_x_poi_id_x_rating

                if False:
                    mat_csr[user_id, poi_id] = user_id_x_poi_id_x_rating

                # df_ratings = \
                #    df_ratings.append(pd.DataFrame([[user_id, poi_id, user_id_x_poi_id_x_rating]],
                #                                   columns=config.HEADER_RECOMMENDER_SYSTEM_RATINGS), ignore_index=True, sort=False)

            if False:
                # loop through the set of all stay points related to the given user a get the cluster_id related to the poi
                for cluster_id in sorted(df_stay_points[df_stay_points.user_id == user_id].cluster_id.unique()):
                    # here we get the cluster_id related to the poi that is part of the set of stay points and get the weighted average of the user x poi
                    user_profile.append(up.UserProfile(user_id=user_id
                                                       , stay_points=df_stay_points[
                            ((df_stay_points.user_id == user_id) & (df_stay_points.cluster_id == cluster_id))]
                                                       , poi=df_pois[(df_pois.cluster_id == cluster_id)]))
                print(user_profile)

            # Once we have done the calculations, we normalize the ratings and use the cut_jenks algorithm to organize the ratings like stars

            # Now let's normalize the data before using the natural breaks
            if len(set(df_stay_points[(df_stay_points.user_id == user_id)]['rating'].to_list())) > 1:
                df_stay_points[(df_stay_points.user_id == user_id)]['rating'] = \
                    (df_stay_points[(df_stay_points.user_id == user_id)]['rating'] -
                     df_stay_points[(df_stay_points.user_id == user_id)]['rating'].min()) / \
                    (df_stay_points[(df_stay_points.user_id == user_id)]['rating'].max() -
                     df_stay_points[(df_stay_points.user_id == user_id)]['rating'].min())

            # Here we obtain the number of different ratings for the given user
            nb_stars = len(set(df_stay_points[(df_stay_points.user_id == user_id)]['rating'].to_list())) - 1
            if nb_stars >= 2:
                if nb_stars > 5:
                    nb_stars = 5

                nb_bins = jenkspy.jenks_breaks(df_stay_points[(df_stay_points.user_id == user_id)].rating.values,
                                               nb_class=nb_stars)
                nb_bins = list(set(nb_bins))
                nb_bins.sort()

                nb_labels = list(np.arange(0, len(list(set(nb_bins)))))
                while len(nb_labels) >= len(nb_bins):
                    nb_labels = list(np.arange(0, len(nb_bins) - 1))

                df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'] = \
                    pd.cut(df_stay_points[(df_stay_points.user_id == user_id)]['rating'],
                           bins=nb_bins,
                           duplicates='drop',
                           labels=nb_labels,
                           include_lowest=True)

                df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'] = pd.to_numeric(
                    df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'])

                # Now let's normalize the data after using the natural breaks
                df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'] = \
                    (df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'] - \
                     df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'].min()) / \
                    (df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'].max() - \
                     df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'].min())

    else:
        df_stay_points = pd.read_pickle('C:\\MAPi\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\df_stay_points_all.df')

    for user_id in df_stay_points.user_id.unique():

        # Here we obtain the number of different ratings for the given user
        nb_stars = len(set(df_stay_points[(df_stay_points.user_id == user_id)]['rating'].to_list())) - 1
        if nb_stars >= 2:
            if nb_stars > 5:
                nb_stars = 5

            nb_bins = jenkspy.jenks_breaks(df_stay_points[(df_stay_points.user_id == user_id)].rating.values,
                                           nb_class=nb_stars)
            nb_bins = list(set(nb_bins))
            nb_bins.sort()

            nb_labels = list(np.arange(0, len(list(set(nb_bins)))))
            while len(nb_labels) >= len(nb_bins):
                nb_labels = list(np.arange(0, len(nb_bins) - 1))

            df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'] = \
                pd.cut(df_stay_points[(df_stay_points.user_id == user_id)]['rating'],
                       bins=nb_bins,
                       duplicates='drop',
                       labels=nb_labels,
                       include_lowest=True)

            df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'] = pd.to_numeric(
                df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'])

            # Now let's normalize the data after using the natural breaks
            df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'] = \
                (df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'] - \
                 df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'].min()) / \
                (df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'].max() - \
                 df_stay_points[(df_stay_points.user_id == user_id)]['cut_jenks'].min())

        for cluster_id in df_stay_points[df_stay_points.user_id == user_id].cluster_id.sort_values().unique():
            mask_user_cluster = ((df_stay_points.cluster_id == cluster_id) & (df_stay_points.user_id == user_id))

            user_id_x_poi_id_x_rating = df_stay_points.loc[mask_user_cluster, 'rating']
            user_id_x_poi_id_x_cut_jenks = df_stay_points.loc[mask_user_cluster, 'cut_jenks']
            df_ratings = \
                df_ratings.append(
                    pd.DataFrame([[user_id, cluster_id, user_id_x_poi_id_x_rating, user_id_x_poi_id_x_cut_jenks]],
                                 columns=config.HEADER_RECOMMENDER_SYSTEM_RATINGS), ignore_index=True, sort=False)

    for cluster_id in df_stay_points[df_stay_points.user_id == user_id].cluster_id.sort_values().unique():
        df_pois.loc[(df_pois.cluster_id == cluster_id), 'rating'] = sum(
            df_stay_points[(df_stay_points.cluster_id == cluster_id)]['rating']) / sum(df_stay_points['rating'])
        df_pois.loc[(df_pois.cluster_id == cluster_id), 'cut_jenks'] = sum(
            df_stay_points[(df_stay_points.cluster_id == cluster_id)]['cut_jenks']) / sum(df_stay_points['cut_jenks'])

    df_ratings['cut_jenks'] = pd.to_numeric(df_ratings['cut_jenks'])
    df_pois['cut_jenks'] = pd.to_numeric(df_pois['cut_jenks'])

    # df_stay_points_merged = pd.DataFrame.merge(df_stay_points, df_pois[['poi_id', 'category', 'name']], on='poi_id', how="inner")

    # We save the processed data
    df_ratings.to_pickle('C:\\MAPi\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\df_ratings_all.df')
    df_pois.to_pickle('C:\\MAPi\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\df_pois_all_ratings.df')
    df_stay_points.to_pickle('C:\\MAPi\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\df_stay_points_all_ratings.df')

    return df_ratings, df_stay_points, df_pois


def build_user_ratings_matrix(df_ids, df_stay_points, df_pois, save_dataset_path):
    calculate_ratings = True
    generate_df_ratings = True

    is_to_normalize_ratings = True
    is_to_cut_jenks_transform = True
    is_to_quantile_transform = True
    nb_stars_ratings = 3

    df_ratings = pd.DataFrame(columns=config.HEADER_RECOMMENDER_SYSTEM_RATINGS)

    df_stay_points['cut_jenks'] = -1.
    df_stay_points['rating_kmeans'] = -1.

    if calculate_ratings:

        # Builds the user profile accordingly to the available points of interest
        # for user_id in df_ids[df_ids.directory.isin(df_stay_points.user_id)].directory.sort_values(ascending=True).unique():
        for user_id in df_stay_points.user_id.unique():

            # print("Building rating matrix - User {} --- started in {}".format(user_id, dt.datetime.now()))

            mask_user_df_stay_points = (df_stay_points.user_id == user_id)

            # Here we loop trhough all pairs of user x location in order to calculate the ratings
            for cluster_id in df_stay_points[mask_user_df_stay_points].cluster_id.sort_values().unique():

                # print("Checking User {} x POI {} --- started in {}".format(user_id, cluster_id, dt.datetime.now()))

                mask_user_x_cluster_df_stay_points = (
                        (df_stay_points.cluster_id == cluster_id) & (df_stay_points.user_id == user_id))

                user_visits_poi = len(df_stay_points[mask_user_x_cluster_df_stay_points])
                if df_stay_points[mask_user_x_cluster_df_stay_points].stay_time.sum() != 0:
                    user_time_visits_poi = pd.to_timedelta(df_stay_points[
                        mask_user_x_cluster_df_stay_points].stay_time).sum().total_seconds()
                    # Here we enforce at least quantity_visits time to the poi as the user had to be there at least for a second for it to count
                    if user_time_visits_poi == 0:
                        user_time_visits_poi = user_visits_poi ** 2
                else:
                    user_time_visits_poi = user_visits_poi ** 2

                total_visits_user = len(df_stay_points[mask_user_df_stay_points])
                total_visits_poi = len(df_stay_points[(df_stay_points.cluster_id == cluster_id)])

                num_visited_places_u = df_stay_points[mask_user_df_stay_points].cluster_id.nunique()

                # Ratio between all the users who have visited this location divided by the number of users
                Fp = df_stay_points[
                         (df_stay_points.cluster_id == cluster_id)].user_id.nunique() / df_stay_points.user_id.nunique()

                user_id_x_poi_id_x_rating = sm.get_UIR(user_visits_poi, user_time_visits_poi, total_visits_user,
                                                       total_visits_poi, num_visited_places_u)
                user_id_x_poi_id_x_rating_sqrt = sm.get_UIR_sqrt(user_visits_poi, user_time_visits_poi,
                                                                 total_visits_user, total_visits_poi,
                                                                 num_visited_places_u, Fp)
                user_id_x_poi_id_x_rating_brais = sm.get_UIR_Brais(user_visits_poi, user_time_visits_poi,
                                                                   total_visits_user, total_visits_poi,
                                                                   num_visited_places_u, Fp)
                # user_id_x_poi_id_x_rating_brais_sig = get_UIR_Brais_sig(user_visits_poi, user_time_visits_poi, total_visits_user, total_visits_poi, num_visited_places_u, Fp)

                df_stay_points.loc[mask_user_x_cluster_df_stay_points, 'visits'] = total_visits_poi
                df_stay_points.loc[mask_user_x_cluster_df_stay_points, 'rating'] = user_id_x_poi_id_x_rating
                df_stay_points.loc[mask_user_x_cluster_df_stay_points, 'rating_sqrt'] = user_id_x_poi_id_x_rating_sqrt
                df_stay_points.loc[mask_user_x_cluster_df_stay_points, 'rating_brais'] = user_id_x_poi_id_x_rating_brais
                # df_stay_points.loc[mask_user_x_cluster_df_stay_points, 'rating_brais_sig'] = user_id_x_poi_id_x_rating_brais

            if is_to_normalize_ratings:
                if len(set(df_stay_points.loc[mask_user_df_stay_points, 'rating'].to_list())) > 1:
                    df_stay_points.loc[mask_user_df_stay_points, 'rating_norm'] = \
                        (df_stay_points.loc[mask_user_df_stay_points, 'rating'] -
                         df_stay_points.loc[mask_user_df_stay_points, 'rating'].min()) / \
                        (df_stay_points.loc[mask_user_df_stay_points, 'rating'].max() -
                         df_stay_points.loc[mask_user_df_stay_points, 'rating'].min())

            df_stay_points.loc[mask_user_df_stay_points] = kb.get_k_means_rating(
                df_stay_points.loc[mask_user_df_stay_points], nb_stars_ratings)
            if is_to_cut_jenks_transform:
                df_stay_points.loc[mask_user_df_stay_points, 'rating_cut_jenks'] = df_stay_points.loc[
                    mask_user_df_stay_points, 'rating_norm']
                # df_stay_points.loc[mask_user_df_stay_points] = jb.get_cut_jenks_transform(df_stay_points.loc[mask_user_df_stay_points], nb_stars_ratings)
            if False:
                df_stay_points.loc[mask_user_df_stay_points, 'rating_my_cut_jenks'] = jn.get_my_jenks_breaks(
                    df_stay_points.loc[mask_user_df_stay_points], nb_stars_ratings)
            if is_to_quantile_transform:
                df_stay_points.loc[mask_user_df_stay_points] = qt.get_quantile_transform(
                    df_stay_points.loc[mask_user_df_stay_points])

        df_stay_points.to_pickle(save_dataset_path + 'df_stay_points_all_ratings.df')

    else:
        df_stay_points = pd.read_pickle(save_dataset_path + 'df_stay_points_all_ratings.df')

    if generate_df_ratings:
        for user_id in df_stay_points.user_id.unique():

            for cluster_id in df_stay_points[df_stay_points.user_id == user_id].cluster_id.sort_values().unique():
                mask_user_cluster = ((df_stay_points.cluster_id == cluster_id) & (df_stay_points.user_id == user_id))

                user_id_x_poi_id_x_rating = df_stay_points.loc[mask_user_cluster, 'rating_norm'].mean()
                user_id_x_poi_id_x_cut_jenks = df_stay_points.loc[mask_user_cluster, 'cut_jenks'].mean()

                if False:
                    df_ratings = \
                        df_ratings.append(
                            pd.DataFrame([[user_id, cluster_id, user_id_x_poi_id_x_rating, user_id_x_poi_id_x_cut_jenks]],
                                         columns=config.HEADER_RECOMMENDER_SYSTEM_RATINGS), ignore_index=True, sort=False)

                df_ratings = pd.concat([df_ratings, pd.DataFrame([[user_id, cluster_id, user_id_x_poi_id_x_rating, user_id_x_poi_id_x_cut_jenks]],
                                     columns=config.HEADER_RECOMMENDER_SYSTEM_RATINGS)], axis=0, ignore_index=True)

        # df_ratings.fillna(0)
        for cluster_id in df_stay_points[df_stay_points.user_id == user_id].cluster_id.sort_values().unique():
            df_pois.loc[(df_pois.cluster_id == cluster_id), 'rating'] = sum(
                df_stay_points[(df_stay_points.cluster_id == cluster_id)]['rating_norm']) / sum(
                df_stay_points['rating_norm'])
            df_pois.loc[(df_pois.cluster_id == cluster_id), 'cut_jenks'] = sum(
                df_stay_points[(df_stay_points.cluster_id == cluster_id)]['cut_jenks']) / sum(
                df_stay_points['cut_jenks'])

        df_ratings['cut_jenks'] = pd.to_numeric(df_ratings['cut_jenks'])
        df_pois['cut_jenks'] = pd.to_numeric(df_pois['cut_jenks'])

        # df_stay_points_merged = pd.DataFrame.merge(df_stay_points, df_pois[['poi_id', 'category', 'name']], on='poi_id', how="inner")

        # We save the processed data
        df_ratings.to_pickle(save_dataset_path + 'df_ratings_all.df')
        df_pois.to_pickle(save_dataset_path + 'df_pois_all_ratings.df')
        df_stay_points.to_pickle(save_dataset_path + 'df_stay_points_all_ratings.df')

    else:
        df_ratings = pd.read_pickle(save_dataset_path + 'df_ratings_all.df')
        df_pois = pd.read_pickle(save_dataset_path + 'df_pois_all_ratings.df')
        df_stay_points = pd.read_pickle(save_dataset_path + 'df_stay_points_all_ratings.df')

    return df_ratings, df_stay_points, df_pois


def get_dataset_stats(dataset_name, dataset_purpose, df_stay_points, df_pois, df_ratings):
    num_users = df_stay_points.user_id.nunique()
    num_item = df_pois.poi_id.nunique()
    num_rating = len(df_ratings)
    k_ratio = (num_rating * 100) / (num_users * num_item)
    # k_ratio = 1 - df_ratings.shape[0] / (df_ratings.user_id.nunique() * df_ratings.poi_id.nunique())
    rating_domain = df_ratings['cut_jenks'].sort_values(ascending=True).unique().tolist()

    print(
        'Dataset: {} - Purpose: {} - #User (M): {} - #Item (N): {} - #Rating (R): {} - k ratio (R*100 / M*N): {} - Rating domain: {}'.format(
            dataset_name, dataset_purpose, num_users, num_item, num_rating, k_ratio, rating_domain))


def get_metrics(df_ratings):
    import recmetrics
    import matplotlib.pyplot as plt

    if False:
        fig = plt.figure(figsize=(15, 7))
        recmetrics.long_tail_plot(df=df_ratings,
                                  item_id_column="poi_id",
                                  interaction_type="POI ratings",
                                  percentage=0.5,
                                  x_labels=False)

    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import train_test_split

    reader = Reader(rating_scale=(0, df_ratings.cut_jenks.max()))
    data = Dataset.load_from_df(df_ratings[['user_id', 'poi_id', 'cut_jenks']], reader)
    # reader = Reader(rating_scale=(0, 1))
    # data = Dataset.load_from_df(df_ratings[['user_id', 'poi_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)

    algo_svd = get_svd_algo(trainset)

    test = algo_svd.test(testset)
    test = pd.DataFrame(test)
    test.drop("details", inplace=True, axis=1)
    test.columns = ['user_id', 'poi_id', 'actual', 'svd_predictions']
    print('SVD test: {}'.format(test.head()))

    print("MSE: ", recmetrics.mse(test.actual, test.svd_predictions))
    print("RMSE: ", recmetrics.rmse(test.actual, test.svd_predictions))

    algo_knn = get_knn_algo(trainset)

    test_knn = algo_knn.test(testset)
    test_knn = pd.DataFrame(test_knn)
    test_knn.drop("details", inplace=True, axis=1)
    test_knn.columns = ['user_id', 'poi_id', 'actual', 'knn_predictions']
    print('KNN test: {}'.format(test_knn.head()))

    print("MSE: ", recmetrics.mse(test_knn.actual, test_knn.knn_predictions))
    print("RMSE: ", recmetrics.rmse(test_knn.actual, test_knn.knn_predictions))

    include_my_rf = False

    my_rf_test = test.copy()
    my_rf_test = my_rf_test.rename(columns={'actual': 'cut_jenks'})
    # my_rf_test = df_ratings
    M = my_rf_test.pivot_table(index=['user_id'], columns=['poi_id'], values='cut_jenks')

    if False:

        fill_na = False
        fill_na_by_pois_mean = False
        if fill_na:
            if fill_na_by_pois_mean:
                # Replacing NaN by POI average
                M = M.fillna(M.mean(axis=0))
            else:
                # Replacing NaN by user Average
                M = M.apply(lambda row: row.fillna(row.mean()), axis=1)

        my_predictions_bhatta = []
        my_predictions_pearson = []
        for user_id, poi_id in \
                zip(test.user_id.unique(),
                    test.poi_id.unique()):
            my_predictions_bhatta.append(
                sm.get_user_location_score(user_id, poi_id, my_rf_test, M, sm.get_bhatta_rec_user(user_id, M, len(M))))
            my_predictions_pearson.append(sm.get_user_location_score(user_id, poi_id, my_rf_test, M,
                                                                     sm.get_pearson_recs_by_user(user_id, M, len(M))))

        test['my_predictions_bhatta'] = my_predictions_bhatta
        my_predictions_bhatta_mse = recmetrics.mse(test.actual, test.my_predictions_bhatta)
        my_predictions_bhatta_rmse = recmetrics.rmse(test.actual, test.my_predictions_bhatta)
        print("my_predictions_bhatta MSE: ", recmetrics.mse(test.actual, test.my_predictions_bhatta))
        print("my_predictions_bhatta RMSE: ", recmetrics.rmse(test.actual, test.my_predictions_bhatta))

        test['my_predictions_pearson'] = my_predictions_pearson
        my_predictions_pearson_mse = recmetrics.mse(test.actual, test.my_predictions_pearson)
        my_predictions_pearson_rmse = recmetrics.rmse(test.actual, test.my_predictions_pearson)
        print("my_predictions_pearson MSE: ", recmetrics.mse(test.actual, test.my_predictions_pearson))
        print("my_predictions_pearson RMSE: ", recmetrics.rmse(test.actual, test.my_predictions_pearson))

    # create model (matrix of predicted values)
    cf_model = test.pivot_table(index='user_id', columns='poi_id', values='svd_predictions')  # .fillna(0)
    knn_model = test_knn.pivot_table(index='user_id', columns='poi_id', values='knn_predictions')  # .fillna(0)

    # get example prediction
    user_id_to_predict = my_rf_test.user_id[0]
    print('SVD Predictions for user {}: {}'.format(user_id_to_predict,
                                                   get_users_predictions(user_id_to_predict, 10, cf_model)))
    print('KNN Predictions for user {}: {}'.format(user_id_to_predict,
                                                   get_users_predictions(user_id_to_predict, 10, knn_model)))

    if include_my_rf:
        print('Bhatta Predictions for user {}: {}'.format(user_id_to_predict, get_users_predictions_bagging(
            sm.get_bhatta_rec_user(user_id_to_predict, M, 10), my_rf_test, 10)))
        print('Pearson Predictions for user {}: {}'.format(user_id_to_predict, get_users_predictions_bagging(
            sm.get_pearson_recs_by_user(user_id_to_predict, M, 10), my_rf_test, 10)))

    # Here we get all the POIs visited by each user and build a list in order to make the comparisons against the recommended list
    test = test.copy().groupby('user_id', as_index=False)['poi_id'].agg({'actual': (lambda x: list(set(x)))})
    test = test.set_index("user_id")

    # make recommendations for all members in the test data
    cf_recs = [] = []
    for user in test.index:
        cf_predictions = get_users_predictions(user, 10, cf_model)
        cf_recs.append(cf_predictions)

    test['cf_predictions'] = cf_recs
    print('cf_predictions: {}'.format(test.head()))

    knn_recs = [] = []
    for user in test.index:
        knn_predictions = get_users_predictions(user, 10, knn_model)
        knn_recs.append(knn_predictions)

    test['knn_predictions'] = knn_recs
    print('knn_predictions: {}'.format(test.head()))

    # make recommendations for all members in the test data
    # bhattacharyya_recs = get_bhatta_rec_user(user_id_to_recommend, M, 10)

    if include_my_rf:
        bhatta_recs = [] = []
        pearson_recs = [] = []

        for user in test.index:
            # print('get_bhatta_rec_user for user {}'.format(user))
            # bhattacharyya_predictions = get_bhatta_rec_user(user, M, 10) #bhattacharyya_recs #get_users_predictions(user, 10, cf_model)
            bhatta_recs.append(get_users_predictions_bagging(sm.get_bhatta_rec_user(user, M, 10), my_rf_test, 10))

            # print('get_pearson_recs_by_user for user {}'.format(user))
            # bhattacharyya_predictions = get_bhatta_rec_user(user, M, 10) #bhattacharyya_recs #get_users_predictions(user, 10, cf_model)
            pearson_recs.append(get_users_predictions_bagging(sm.get_pearson_recs_by_user(user, M, 10), my_rf_test, 10))

        test['bhattacharyya_predictions'] = bhatta_recs
        print('bhattacharyya_predictions: {}'.format(test.head()))
        test['pearson_predictions'] = pearson_recs
        print('pearson_predictions: {}'.format(test.head()))

    # make recommendations for all members in the test data
    popularity_recs = df_ratings.poi_id.value_counts().head(10).index.tolist()
    pop_recs = []
    for user in test.index:
        pop_predictions = popularity_recs
        pop_recs.append(pop_predictions)

    test['pop_predictions'] = pop_recs
    print('pop_predictions: {}'.format(test.head()))

    # make recommendations for all members in the test data
    ran_recs = []
    for user in test.index:
        random_predictions = df_ratings.poi_id.sample(10).values.tolist()
        ran_recs.append(random_predictions)

    test['random_predictions'] = ran_recs
    print('random_predictions: {}'.format(test.head()))

    actual = test.actual.values.tolist()
    cf_predictions = test.cf_predictions.values.tolist()
    knn_predictions = test.knn_predictions.values.tolist()
    pop_predictions = test.pop_predictions.values.tolist()
    random_predictions = test.random_predictions.values.tolist()
    if include_my_rf:
        bhattacharyya_predictions = test.bhattacharyya_predictions.values.tolist()
        pearson_predictions = test.pearson_predictions.values.tolist()

    pop_mark = []
    for K in np.arange(1, 11):
        pop_mark.extend([recmetrics.mark(actual, pop_predictions, k=K)])
    print('pop_mark: {}'.format(pop_mark))

    random_mark = []
    for K in np.arange(1, 11):
        random_mark.extend([recmetrics.mark(actual, random_predictions, k=K)])
    print('random_mark: {}'.format(random_mark))

    cf_mark = []
    for K in np.arange(1, 11):
        cf_mark.extend([recmetrics.mark(actual, cf_predictions, k=K)])
    print('cf_mark: {}'.format(cf_mark))

    knn_mark = []
    for K in np.arange(1, 11):
        knn_mark.extend([recmetrics.mark(actual, knn_predictions, k=K)])
    print('knn_mark: {}'.format(knn_mark))

    if include_my_rf:
        bhatta_mark = []
        for K in np.arange(1, 11):
            bhatta_mark.extend([recmetrics.mark(actual, bhattacharyya_predictions, k=K)])
        print('bhatta_mark: {}'.format(bhatta_mark))

        pearson_mark = []
        for K in np.arange(1, 11):
            pearson_mark.extend([recmetrics.mark(actual, pearson_predictions, k=K)])
        print('pearson_mark: {}'.format(pearson_mark))

    include_auto_algos = True

    if include_auto_algos:
        if include_my_rf:
            mark_scores = [random_mark, pop_mark, cf_mark, knn_mark, bhatta_mark, pearson_mark]
            names = ['Random Recommender', 'Popularity Recommender', 'Collaborative Filter', 'KNN Recommender',
                     'Bhattacharyya Collaborative Filter', 'Pearson Recommender']
        else:
            mark_scores = [random_mark, pop_mark, cf_mark, knn_mark]
            names = ['Random Recommender', 'Popularity Recommender', 'Collaborative Filter', 'KNN Recommender']
    else:
        mark_scores = [random_mark, pop_mark, bhatta_mark, pearson_mark]
        names = ['Random Recommender', 'Popularity Recommender', 'Bhattacharyya Collaborative Filter',
                 'Pearson Recommender']

    index = range(1, 10 + 1)

    fig = plt.figure(figsize=(15, 7))
    recmetrics.mark_plot(mark_scores, model_names=names, k_range=index)

    print('Done with the metrics')

    # test.to_pickle('C:\\MAPi\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\df_test_results.df')

    return mark_scores, names, test


def get_users_predictions_bagging(model, df_ratings, n):
    # Here we get the recommended pois for the current user and plot them to compare against the originals
    clean_user_id_list = [x[0] for x in set([r for r in model])]

    recommended_items = df_ratings[df_ratings.user_id.isin(clean_user_id_list)]
    recommended_items = recommended_items.sort_values('cut_jenks', ascending=False)
    recommended_items = recommended_items["poi_id"]
    recommended_items = recommended_items.head(n)
    return recommended_items.index.tolist()


def create_baseline_model(rating_matrix, k=20, metric="cosine"):
    """
    :param R : numpy array of item representations
    :param k : number of nearest neighbors to return
    :return model : our knn model
    """
    model = NearestNeighbors(metric=metric, n_neighbors=k + 1, algorithm='brute')
    model.fit(rating_matrix)
    return model


def get_svd_algo(trainset):
    from surprise import SVD

    algo = SVD()
    algo.fit(trainset)

    return algo


def get_knn_algo(trainset):
    from surprise import KNNBaseline

    algo = KNNBaseline()
    algo.fit(trainset)

    return algo


def cf_recommender_system(df_ids, df_stay_points, df_pois, process_build_user_ratings_matrix, dataset_enum):
    recommender_system_path = utils.get_folder_path(dataset_enum) + config.RECOMMENDER_SYSTEM_OUTPUT_FOLDER + '/'
    utils.get_or_create_path(recommender_system_path)

    maps_path = utils.get_folder_path(dataset_enum) + config.RECOMMENDER_SYSTEM_MAPS_OUTPUT_FOLDER + '/'
    utils.get_or_create_path(maps_path)

    if process_build_user_ratings_matrix:
        # filter out pois with no category
        df_pois = df_pois[df_pois['category'] != '']
        df_pois['poi_id'] = df_pois['cluster_id']
        # filter the pois that have at least X visits from different users

        # df_pois = df_pois[(df_pois.quantity_visits >= config.POI_MINIMUM_VISIT_COUNT)]

        df_pois = df_pois[(df_pois.quantity_visits > 0)]

        # df_pois['poi_id'] = np.arange(df_pois.shape[0])

        # df_stay_points = df_stay_points[(df_stay_points.cluster_id.isin(df_pois.cluster_id)) & (df_stay_points.stay_time > pd.to_timedelta(1, unit='s'))]
        df_stay_points = df_stay_points[(df_stay_points.cluster_id.isin(df_pois.cluster_id))]

        # (df_stay_points[(df_stay_points.cluster_id.isin(df_stay_points.groupby(by=['user_id'], as_index=False)['cluster_id'].size().values)]
        # .groupby(by=['user_id', 'cluster_id'], as_index=False).size() >= 3).values.reshape((-1, 1))

        df_stay_points['poi_id'] = df_stay_points['cluster_id']
        df_stay_points['rating'] = -1
        df_pois['rating'] = -1

        if config.ExtractMeaningfulPlaces.ANNOTATE_MPS:
            # df_pois = poi.annotate_pois(df_pois)
            df_pois = df_pois

        # row indices
        row_ind = np.array(
            df_ids[df_ids.directory.isin(df_stay_points.user_id)].directory.sort_values(ascending=True).values)
        # column indices
        col_ind = np.array(
            df_pois[df_pois.poi_id.isin(df_stay_points.poi_id)].poi_id.sort_values(ascending=True).values)

        # For the matrix
        mat_csr = sparse.lil_matrix((len(row_ind), len(col_ind)))

        # This matrix represents if an user has visited the poi, just to have an idea about the relations
        boolean_utility_matrix = sparse.lil_matrix((len(row_ind), len(col_ind)))

        user_profile = []

        df_ratings, df_stay_points, df_pois = \
            build_user_ratings_matrix(df_ids, df_stay_points, df_pois, recommender_system_path)

        #mat_csr.to_pickle(recommender_system_path + 'mat_csr_all.df')
        #mat_csr.to_csv(recommender_system_path + 'mat_csr_all.csv')
    else:
        # We read the processed data
        df_ratings = pd.read_pickle(recommender_system_path + 'df_ratings_all.df')
        df_pois = pd.read_pickle(recommender_system_path + 'df_pois_all_ratings.df')
        df_stay_points = pd.read_pickle(recommender_system_path + 'df_stay_points_all_ratings.df')

    if False:
        boolean_utility_matrix.tocsr()
        mat_csr.tocsr()

    # df_no_ratings = pd.DataFrame(df_ratings.groupby('poi_id')['cut_jenks'].count())
    # df_no_ratings.sort_values('cut_jenks', ascending=False).head()

    get_dataset_stats(dataset_enum, 'Trajectories', df_stay_points, df_pois, df_ratings)

    if False:
        df_ratings.cut_jenks.value_counts(sort=False).plot(kind='bar')

    # Calculating the recommendation
    # M = df_ratings.pivot_table(index=['user_id'], columns=['poi_id'], values='cut_jenks')
    M = df_ratings.pivot_table(index=['user_id'], columns=['poi_id'], values='rating')

    # Testing only the 10 first users
    # df_ratings = df_ratings[(df_ratings.user_id >= 0) & (df_ratings.user_id < 10)]



    if False:
        df_ratings = df_ratings.fillna(0)
        nb_stars_ratings = 3
        for user_id in df_ratings.user_id.unique():
            mask_user_df_stay_points = (df_ratings.user_id == user_id)
            df_ratings.loc[mask_user_df_stay_points] = jb.get_cut_jenks_transform(
                df_ratings.loc[mask_user_df_stay_points], nb_stars_ratings, by_rating=True)

    # df_ratings['cut_jenks'] = df_ratings['rating']

    check_metrics = False
    if check_metrics:
        get_metrics(df_ratings)

    user_id_to_recommend = 1

    check_pois = False
    if check_pois:
        for user_id_to_recommend in df_ids.directory:

            # Here we get the original pois for the current user and plot them to compare against the recommended
            df_pois_to_plot = df_pois[df_pois.poi_id.isin(df_ratings[df_ratings.user_id == user_id_to_recommend].poi_id)]
            if True:
                poi.plot_poi_points_df(df_pois_to_plot.to_dict(), maps_path,
                                       name=str(user_id_to_recommend).zfill(3) + '_original')

            get_bhatta_rec_recs = sm.get_bhatta_rec_user(user_id_to_recommend, M, len(df_pois))
            if len(get_bhatta_rec_recs):
                print('User {} x Bhattacharyya Correlation: {}'.format(user_id_to_recommend, get_bhatta_rec_recs))
                print('More correlated ')
                print(get_bhatta_rec_recs[:10])
                print('Less correlated ')
                print(get_bhatta_rec_recs[-10:])

                # clean_list = [x for x in set([r for r in get_bhatta_rec_recs]) if not x[0] in set(M.columns)]

                # Here we get the recommended pois for the current user and plot them to compare against the originals
                clean_list = [x[0] for x in set([r for r in get_bhatta_rec_recs[:10]])]

                # Here we plot each of the recommended user's pois to see if it makes sense
                for recommended_user in clean_list:
                    df_pois_to_plot = df_pois[df_pois.poi_id.isin(
                        df_ratings[df_ratings.user_id == recommended_user].poi_id)]  # df_pois[df_pois.index.isin(clean_list)]
                    # poi.plot_scatter_pois('longitude', 'latitude', df_pois, 'poi_id', show_legend=False, show_on_map=True)
                    poi.plot_poi_points_df(df_pois_to_plot.to_dict(), maps_path,
                                           name=str(user_id_to_recommend).zfill(3) + '_recommended_to_' + str(
                                               recommended_user).zfill(3))

                df_pois_to_plot = df_pois[df_pois.poi_id.isin(
                    df_ratings[df_ratings.user_id.isin(clean_list)].poi_id)]  # df_pois[df_pois.index.isin(clean_list)]
                # poi.plot_scatter_pois('longitude', 'latitude', df_pois, 'poi_id', show_legend=False, show_on_map=True)
                poi.plot_poi_points_df(df_pois_to_plot.to_dict(), maps_path,
                                       name=str(user_id_to_recommend).zfill(3) + '_recommended')

    get_recs_recs = sm.get_recs(df_ratings.poi_id.iloc[0], M, len(df_pois))
    print('Pearson Correlation: {}'.format(get_recs_recs))
    print('More correlated ')
    print(get_recs_recs[:10])
    print('Less correlated ')
    print(get_recs_recs[-10:])

    if False:
        get_recs_by_user_recs = sm.get_recs_by_user(1, M, len(df_pois))
        print('get_recs_by_user Correlation: {}'.format(get_recs_by_user_recs))
        print('More correlated ')
        print(get_recs_by_user_recs[:10])
        print('Less correlated ')
        print(get_recs_by_user_recs[-10:])

        get_recs_by_item_recs = sm.get_recs_by_item(250, M, len(df_pois))
        print('get_recs_by_item Correlation: {}'.format(get_recs_by_item_recs))
        print('More correlated ')
        print(get_recs_by_item_recs[:10])
        print('Less correlated ')
        print(get_recs_by_item_recs[-10:])

    if False:
        recs = sm.get_UUCF(df_ratings)
        print('get_UUCF: {}'.format(recs))
        print('More correlated ')
        print(recs[:10])
        print('Less correlated ')
        print(recs[-10:])

        recs = sm.get_LLCF(df_ratings, 1)
        print('get_LLCF: {}'.format(recs))
        print('More correlated ')
        print(recs[:10])
        print('Less correlated ')
        print(recs[-10:])

        recs = sm.get_CF(df_ratings, 1)
        print('get_CF: {}'.format(recs))
        print('More correlated ')
        print(recs[:10])
        print('Less correlated ')
        print(recs[-10:])

    get_cosine_sim_recs = sm.get_cosine_sim(df_ratings.user_id.iloc[0], df_ratings.poi_id.iloc[0], M, len(df_pois))
    print('Cosine Similarity: {}'.format(get_cosine_sim_recs))
    print('More correlated ')
    print(get_cosine_sim_recs[:10])
    print('Less correlated ')
    print(get_cosine_sim_recs[-10:])

    if False:
        get_joao_rec_recs = sm.get_joao_rec(df_ratings.user_id.iloc[0], df_ratings.poi_id.iloc[0], M, len(df_pois))
        print('João Gama Correlation: {}'.format(get_joao_rec_recs))
        print('More correlated ')
        print(get_joao_rec_recs[:10])
        print('Less correlated ')
        print(get_joao_rec_recs[-10:])

    print("Finished processing ratings")


def evaluate(x_test, y_test, neighbors, similarities, np_ratings, mean):
    print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
    preds = list(predict(u, i, neighbors, similarities, np_ratings, mean) for (u, i) in x_test)
    mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
    print('\nMAE :', mae)
    return mae


def get_model_similarities_neighbors(ratings, model_name='baseline'):
    def write_message(message):
        sys.stdout.write('\n')
        sys.stdout.write(message)
        sys.stdout.write('\n')
        sys.stdout.flush()

    save_to_pickle = False
    min_num_recommendations = 10
    max_num_recommendations = 100
    num_recommendations = 10

    DATASETS_PATH = config.GEOLIFE_RECOMMENDER_SYSTEM_PATH

    nb_users = ratings.user_id.nunique()
    mean, norm_ratings = preprocessing.normalize(ratings)
    np_ratings = norm_ratings.to_numpy()

    # metric : choose among ['cosine', 'adjusted_cosine', 'cosine_adj', 'bhattacharyya', 'SBCF_adj', 'correlation', 'jaccard', 'euclidean']
    metrics = ['bhattacharyya', 'cosine', 'SBCF_adj', 'correlation', 'jaccard']
    # metric = 'bhattacharyya'
    use_penalization = True

    ratings_items = pd.pivot_table(norm_ratings, values='norm_rating', index='user_id', columns='item_id')
    ratings_matrix = csr_matrix(
        ratings_items.values
    ).toarray()

    metric = model_name
    k = max_num_recommendations

    try:

        if metric == 'cosine':

            write_message('Calculating similarity for :{}'.format(metric))
            R = preprocessing.ratings_matrix(norm_ratings)
            model = create_baseline_model(R, k=k, metric=metric)
            similarities, neighbors = get_nearest_neighbors(R, model)

            if False:
                cosine_similarities_bkp = similarities
                cosine_neighbors_bkp = neighbors
                setattr(this_module, f'similatiries_{metric}', similarities)
                setattr(this_module, f'neighbors_{metric}', neighbors)

            vars()[f'similatiries_{metric}'] = similarities
            vars()[f'neighbors_{metric}'] = neighbors

            return similarities, neighbors

        elif metric == 'adjusted_cosine' and False:

            write_message('Calculating similarity for :{}'.format(metric))

            similarities, neighbors = adjusted_cosine(np_ratings, nb_users=nb_users,
                                                      dataset_name='geolife_pois', ratings=ratings)

            adjusted_cosine_similarities_bkp = similarities
            adjusted_cosine_neighbors_bkp = neighbors
            # similarities, neighbors = load_similarities('geolife_pois')

            vars()[f'similatiries_{metric}'] = similarities
            vars()[f'neighbors_{metric}'] = neighbors

            return similarities, neighbors

        elif metric == 'cosine_adj':

            write_message('Calculating similarity for :{}'.format(metric))

            cosine_adj_sim_matrix = calculate_similarity_matrix(ratings_matrix=ratings_matrix,
                                                                n_neighbors=nb_users, ratings=ratings,
                                                                metric='cosine_adj')
            np.fill_diagonal(cosine_adj_sim_matrix, 1)

            cosine_adj_similarity = pd.DataFrame(cosine_adj_sim_matrix, index=ratings_items.index)
            cosine_adj_similarity.columns = ratings_items.index

            if save_to_pickle: cosine_adj_similarity.to_pickle(
                DATASETS_PATH + 'user_matrix_cosine_adj_similarity_110422.df')

            # get neighbors by their neighbors in decreasing order of similarities
            neighbors = find_n_neighbours(cosine_adj_similarity, k).to_numpy()

            # sort similarities in decreasing order
            similarities = find_n_similarities(cosine_adj_similarity, k).to_numpy()

            cosine_adj_similarities_bkp = similarities
            cosine_adj_neighbors_bkp = neighbors

            vars()[f'similatiries_{metric}'] = similarities
            vars()[f'neighbors_{metric}'] = neighbors

            return similarities, neighbors

        elif metric == 'bhattacharyya':

            write_message('Calculating similarity for :{}'.format(metric))

            bhattacharyya_sim_matrix = calculate_similarity_matrix(ratings_matrix=ratings_matrix,
                                                                   n_neighbors=nb_users, ratings=ratings,
                                                                   metric='bhattacharyya',
                                                                   use_penalization=use_penalization)

            np.fill_diagonal(bhattacharyya_sim_matrix, 1)
            bhattacharyya_similarity = pd.DataFrame(bhattacharyya_sim_matrix, index=ratings_items.index)
            bhattacharyya_similarity.columns = ratings_items.index

            if use_penalization:
                if save_to_pickle:  utils.save_pickle_file(file=bhattacharyya_similarity, path_name=DATASETS_PATH,
                                                           file_name='user_matrix_bhatta_similarity_pen_score')
                # bhatta_similarity.to_pickle(DATASETS_PATH + 'user_matrix_bhatta_similarity_pen_score110422.df')
            else:
                if save_to_pickle:  utils.save_pickle_file(file=bhattacharyya_similarity, path_name=DATASETS_PATH,
                                                           file_name='user_matrix_bhatta_similarity')
                # bhatta_similarity.to_pickle(DATASETS_PATH + 'user_matrix_bhatta_similarity_110422.df')

            # get neighbors by their neighbors in decreasing order of similarities
            neighbors = find_n_neighbours(bhattacharyya_similarity, k).to_numpy()

            # sort similarities in decreasing order
            similarities = find_n_similarities(bhattacharyya_similarity, k).to_numpy()

            # save similarities to disk
            save_similarities(similarities, neighbors, dataset_name='bhattacharyya_geolife_original')

            if use_penalization:
                bhattacharyya_pen_similarities_bkp = similarities
                bhattacharyya_pen_neighbors_bkp = neighbors
            else:
                bhattacharyya_similarities_bkp = similarities
                bhattacharyya_neighbors_bkp = neighbors

            vars()[f'similatiries_{metric}'] = similarities
            vars()[f'neighbors_{metric}'] = neighbors

            return similarities, neighbors

        elif metric == 'SBCF_adj':

            write_message('Calculating similarity for :{}'.format(metric))

            SBCF_adj_sim_matrix = calculate_similarity_matrix(ratings_matrix=ratings_matrix, n_neighbors=nb_users,
                                                              ratings=ratings, metric='SBCF_adj')

            np.fill_diagonal(SBCF_adj_sim_matrix, 1)
            SBCF_adj_similarity = pd.DataFrame(SBCF_adj_sim_matrix, index=ratings_items.index)
            SBCF_adj_similarity.columns = ratings_items.index

            if save_to_pickle: SBCF_adj_similarity.to_pickle(
                DATASETS_PATH + 'user_matrix_SBCF_adj_similarity_110422.df')

            # get neighbors by their neighbors in decreasing order of similarities
            neighbors = find_n_neighbours(SBCF_adj_similarity, k)

            # sort similarities in decreasing order
            similarities = find_n_similarities(SBCF_adj_similarity, k)

            neighbors = neighbors.to_numpy()
            similarities = similarities.to_numpy()

            SBCF_adj_similarities_bkp = similarities
            SBCF_adj_neighbors_bkp = neighbors

            # save similarities to disk
            if save_to_pickle: save_similarities(similarities, neighbors, dataset_name='bhattacharyya_geolife_SBCF_adj')

            vars()[f'similatiries_{metric}'] = similarities
            vars()[f'neighbors_{metric}'] = neighbors

            return similarities, neighbors

        elif metric == 'correlation':

            write_message('Calculating similarity for :{}'.format(metric))

            pearson_correlation_sim_matrix = calculate_similarity_matrix(ratings_matrix=ratings_matrix,
                                                                         n_neighbors=nb_users, ratings=ratings,
                                                                         metric='correlation')
            np.fill_diagonal(pearson_correlation_sim_matrix, 1)

            correlation_similarity = pd.DataFrame(pearson_correlation_sim_matrix, index=ratings_items.index)
            correlation_similarity.columns = ratings_items.index

            if save_to_pickle: correlation_similarity.to_pickle(
                DATASETS_PATH + 'user_matrix_pearson_similarity_110422.df')

            # get neighbors by their neighbors in decreasing order of similarities
            neighbors = find_n_neighbours(correlation_similarity, k).to_numpy()

            # sort similarities in decreasing order
            similarities = find_n_similarities(correlation_similarity, k).to_numpy()

            correlation_similarities_bkp = similarities
            correlation_neighbors_bkp = neighbors

            vars()[f'similatiries_{metric}'] = similarities
            vars()[f'neighbors_{metric}'] = neighbors

            return similarities, neighbors

        elif metric == 'jaccard':

            write_message('Calculating similarity for :{}'.format(metric))

            jaccard_sim_matrix = calculate_similarity_matrix(ratings_matrix=ratings_matrix, n_neighbors=nb_users,
                                                             ratings=ratings, metric='jaccard')
            np.fill_diagonal(jaccard_sim_matrix, 1)

            jaccard_similarity = pd.DataFrame(jaccard_sim_matrix, index=ratings_items.index)
            jaccard_similarity.columns = ratings_items.index

            if save_to_pickle: jaccard_similarity.to_pickle(DATASETS_PATH + 'user_matrix_jaccard_similarity_110422.df')

            # get neighbors by their neighbors in decreasing order of similarities
            neighbors = find_n_neighbours(jaccard_similarity, k).to_numpy()

            # sort similarities in decreasing order
            similarities = find_n_similarities(jaccard_similarity, k).to_numpy()

            jaccard_similarities_bkp = similarities
            jaccard_neighbors_bkp = neighbors

            vars()[f'similatiries_{metric}'] = similarities
            vars()[f'neighbors_{metric}'] = neighbors

            return similarities, neighbors

    except BaseException as e:
        write_message('Failed to do something: ' + str(e))

    return None, None


def evaluate_bench(x_test, y_test, df_test, min_num_recommendations, max_num_recommendations, k_neighbours_step,
                   ratings):

    print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))

    benchmark = dict()
    benchmark_model_names = [] = []
    mae_scores_benchmark = [] = []
    rmse_scores_benchmark = [] = []
    mape_scores_benchmark = [] = []

    # metrics = ['cosine', 'adjusted_cosine', 'cosine_adj', 'bhattacharyya', 'SBCF_adj', 'correlation', 'jaccard', 'euclidean']
    metrics = ['cosine', 'bhattacharyya', 'correlation', 'jaccard']


    for metric in metrics:
        vars()[f'similatiries_{metric}'], vars()[f'neighbors_{metric}'] = \
            get_model_similarities_neighbors(ratings, model_name=metric)


    k_neighbours_step = 10

    df_test['actual'] = y_test

    test = pd.DataFrame(y_test, columns=['actual'])
    df_benchmark = pd.DataFrame(columns=['model', 'k', 'mae', 'rmse', 'mape'])

    # mean ratings for each user
    mean = ratings.groupby(by='user_id', as_index=False)['rating'].mean()
    mean_ratings = pd.merge(ratings, mean, suffixes=('', '_mean'), on='user_id')

    # normalized ratings for each item
    mean_ratings['norm_rating'] = mean_ratings['rating'] - mean_ratings['rating_mean']

    mean = mean.to_numpy()[:, 1]

    _, norm_ratings = preprocessing.normalize(ratings)
    np_ratings = norm_ratings.to_numpy()

    for n_k in range(min_num_recommendations, max_num_recommendations, k_neighbours_step):

        print('************ Working on K: {} ************'.format(n_k))

        mae_scores = []
        rmse_scores = []
        mape_scores = []

        for metric in metrics:

            # vars()[f'similatiries_{metric}'], vars()[f'neighbors_{metric}'] = \
            #     get_model_similarities_neighbors(ratings, model_name=metric)

            if vars()[f'similatiries_{metric}'] is not None:
                similarities = vars()[f'similatiries_{metric}'][:, 0:n_k]
                neighbors = vars()[f'neighbors_{metric}'][:, 0:n_k]

                preds = list(predict(u, i, neighbors, similarities, np_ratings, mean) for (u, i) in x_test)
                model_name = f'{metric}-{n_k}'
                test[model_name] = preds
                df_test[model_name] = preds

                mae = mean_absolute_error(y_true=test['actual'], y_pred=test[model_name])
                mae_scores.append(mae)

                rmse = sqrt(mean_squared_error(y_true=test['actual'], y_pred=test[model_name]))
                rmse_scores.append(rmse)

                mape = evaluation_metrics.calculate_mape(y_true=test['actual'], y_pred=test[model_name])
                mape_scores.append(mape)

                benchmark_model_names.append(model_name)
                df_benchmark = \
                    df_benchmark.append(pd.DataFrame([[model_name.split('-')[0], n_k, mae, rmse, mape]],
                                                     columns=['model', 'k', 'mae', 'rmse', 'mape']), ignore_index=True,
                                        sort=False)

                benchmark.update({'model_name': model_name, 'mae': mae, 'rmse': rmse, 'mape': mape})
                print('{} --- MAE = {} --- RMSE = {} --- MAPE = {}'.format(model_name, mae, rmse, mape))

            mae_scores_benchmark.append(mae_scores)
            rmse_scores_benchmark.append(rmse_scores)
            mape_scores_benchmark.append(mape_scores)

    return test, df_benchmark


def get_users_predictions_model(user_id, num_recommendations, model):
    if num_recommendations > len(model.loc[user_id]):
        print('get_users_predictions for user {} is {} while the k is {}'.format(user_id, num_recommendations))
        num_recommendations = len(model.loc[user_id])

    recommended_items = pd.DataFrame(model.loc[user_id])
    recommended_items.columns = ["predicted_rating"]
    recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)
    recommended_items = recommended_items.head(num_recommendations)

    return recommended_items.index.tolist()


def get_model_recommendations_list(recommendations_list_data, df_test, model_name, num_recommendations):
    if 'poi_id' in df_test.columns.to_list():
        df_test.rename(columns={'poi_id': 'item_id'}, inplace=True)
    model_recommendations_list = df_test.copy().pivot_table(index='user_id', columns='item_id', values=model_name)

    model_recs = [] = []
    for user_id in model_recommendations_list.index:
        # if False or model_name == 'bhattacharyya':
        #    model_predictions = get_users_predictions_bhattacharya(user, num_recommendations)
        # else:
        model_predictions = get_users_predictions(user_id, num_recommendations, model_recommendations_list)

        model_recs.append(model_predictions)

        # print('Model: {} -- k:{} -- recommendations : {}'.format(model_name, num_recommendations, model_predictions))

    # recommendations_list_data[model_name] = model_recs
    # print('{} recommendations : {}'.format(model_name, recommendations_list_data.head()))

    # print('k:{} -- Size of data: {} -- recommendations : {}'.format(num_recommendations, len(model_recs), model_recs))
    return model_recs


def get_popularity_recs(recommendations_list_data, num_recommendations):

    if len(recommendations_list_data) <= num_recommendations:
        num_recommendations = len(recommendations_list_data)

    # make recommendations for all members in the test data
    popularity_recs = recommendations_list_data.item_id.value_counts().head(num_recommendations).index.tolist()
    model_recommendations_list = recommendations_list_data.copy().pivot_table(index='user_id', columns='item_id',
                                                                              values='actual')

    pop_recs = [] = []
    for user in model_recommendations_list.index:
        pop_predictions = popularity_recs
        pop_recs.append(pop_predictions)

    # test['pop_predictions'] = pop_recs
    # print('pop_predictions: {}'.format(recommendations_list_data.head()))

    return pop_recs


def get_random_recs(recommendations_list_data, num_recommendations):
    # make recommendations for all members in the test data
    model_recommendations_list = recommendations_list_data.copy().pivot_table(index='user_id', columns='item_id',
                                                                              values='actual')
    ran_recs = [] = []
    for user in model_recommendations_list.index:
        random_predictions = recommendations_list_data.item_id.sample(num_recommendations).values.tolist()
        ran_recs.append(random_predictions)

    # recommendations_list_data['random_predictions'] = ran_recs
    # print('random_predictions: {}'.format(recommendations_list_data.head()))

    return ran_recs


def fill_models_recommendations_list(original_recommendations_list, df_test, num_recommendations):
    # We start from the 4th colum because the 2 first ones are user and item IDs and the 3rd is the actual value
    models_list = df_test.iloc[:, 3:].columns.to_list()

    has_num_recs = True
    try:
        models_at_k = [m.split('-')[0] for m in models_list if int(m.split('-')[1]) == num_recommendations]
    except:
        has_num_recs = False
        models_at_k = [m.split('-')[0] for m in models_list]

    for model_name in models_at_k:
        if has_num_recs:
            print('fill_models_recommendations_list for model {} and k {}'.format(model_name, num_recommendations))
            model_name = str(model_name) + '-' + str(num_recommendations)
        else:
            print('fill_models_recommendations_list for model {}'.format(model_name))
            model_name = str(model_name)
        original_recommendations_list[model_name] = get_model_recommendations_list(original_recommendations_list,
                                                                                   df_test, model_name,
                                                                                   num_recommendations)

    original_recommendations_list['popularity'] = get_popularity_recs(df_test, num_recommendations)
    original_recommendations_list['random'] = get_random_recs(df_test, num_recommendations)

    return original_recommendations_list


def get_original_recommendations_list(test, num_recommendations):
    # Here we get all the POIs visited by each user and build a list in order to make the comparisons against the recommended list
    if 'poi_id' in test.columns.to_list():
        test.rename(columns={'poi_id': 'item_id'}, inplace=True)

    test_original_recommendations_list = test.copy().groupby('user_id', as_index=False)['item_id'].agg(
        {'actual': (lambda x: list(set(x))[:num_recommendations])})
    test_original_recommendations_list = test_original_recommendations_list.set_index("user_id")

    return test_original_recommendations_list


def get_users_predictions(user_id, num_recommendations, model):
    if num_recommendations > len(model.loc[user_id]):
        print('get_users_predictions for user {} is {} while the k is {}'.format(user_id,
                                                                                 len(model.loc[user_id]),
                                                                                 num_recommendations))
        num_recommendations = len(model.loc[user_id])

    recommended_items = pd.DataFrame(model.loc[user_id])
    recommended_items.columns = ["predicted_rating"]
    recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)
    recommended_items = recommended_items.head(num_recommendations)

    return recommended_items.index.tolist()


def get_users_predictions_bhattacharya(user_id, num_recommendations, bhatta_user_item_scores_full):
    # print('getting pedictions for user {}'.format(user_id))
    _, recommended_items = bhatta_user_item_scores_full(user_id, num_recommendations, num_recommendations)
    # recommended_items.columns = ["predicted_rating"]
    # recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)
    # recommended_items = recommended_items.head(num_recommendations)
    return recommended_items.index.tolist()


def get_model_recommendations_list_2(recommendations_list_data, model_name, num_recommendations):
    model_recommendations_list = recommendations_list_data.copy().pivot_table(index='user_id', columns='item_id',
                                                                              values=model_name)

    model_recs = [] = []
    for user in model_recommendations_list.index:
        # if False or model_name == 'bhattacharyya':
        #    model_predictions = get_users_predictions_bhattacharya(user, num_recommendations)
        # else:
        model_predictions = get_users_predictions(user, num_recommendations, model_recommendations_list)

        model_recs.append(model_predictions)

        # print('Model: {} -- k:{} -- recommendations : {}'.format(model_name, num_recommendations, model_predictions))

    # recommendations_list_data[model_name] = model_recs
    # print('{} recommendations : {}'.format(model_name, recommendations_list_data.head()))

    # print('k:{} -- Size of data: {} -- recommendations : {}'.format(num_recommendations, len(model_recs), model_recs))
    return model_recs


def get_popularity_recs(recommendations_list_data, num_recommendations):
    # make recommendations for all members in the test data
    popularity_recs = recommendations_list_data.item_id.value_counts().head(num_recommendations).index.tolist()
    model_recommendations_list = recommendations_list_data.copy().pivot_table(index='user_id', columns='item_id',
                                                                              values='actual')

    pop_recs = [] = []
    for user in model_recommendations_list.index:
        pop_predictions = popularity_recs
        pop_recs.append(pop_predictions)

    # test['pop_predictions'] = pop_recs
    # print('pop_predictions: {}'.format(recommendations_list_data.head()))

    return pop_recs


def get_random_recs_2(recommendations_list_data, num_recommendations):
    # make recommendations for all members in the test data
    model_recommendations_list = recommendations_list_data.copy().pivot_table(index='user_id', columns='item_id',
                                                                              values='actual')
    ran_recs = [] = []
    for user in model_recommendations_list.index:
        random_predictions = recommendations_list_data.item_id.sample(num_recommendations).values.tolist()
        ran_recs.append(random_predictions)

    # recommendations_list_data['random_predictions'] = ran_recs
    # print('random_predictions: {}'.format(recommendations_list_data.head()))

    return ran_recs


def fill_models_recommendations_list_2(test, original_recommendations_list, num_recommendations, k):
    # We start from the 4th colum because the 2 first ones are user and item IDs and the 3rd is the actual value
    for model_name in test.iloc[:, 3:].columns:
        print('fill_models_recommendations_list for model {} and k {}'.format(model_name, k))
        original_recommendations_list[model_name] = get_model_recommendations_list(test, model_name,
                                                                                   num_recommendations)

    original_recommendations_list['popularity'] = get_popularity_recs(test, num_recommendations)
    original_recommendations_list['random'] = get_random_recs(test, num_recommendations)

    return original_recommendations_list


def reorder_recommendations_list(test, original_recommendations_list):
    # We start from the 4th colum because the 2 first ones are user and item IDs and the 3rd is the actual value
    for model_name in test.iloc[:, 3:].columns:
        # print(model_name)
        original_recommendations_list[model_name] = original_recommendations_list[model_name].map(np.sort).map(list)
        # test_description['actual'] = test_description['actual'].map(np.sort).map(list)


# reorder_recommendations_list()


def get_top_k_locations_ratio(user_id, top_k_recommendations, test, check):
    total_visits_user = check.columns[check[check.index == user_id].notna().any()].tolist()

    test_top = test.pivot_table(index='user_id', columns='poi_id', values='bhattacharyya')

    top_k_locations_visits_user = get_users_predictions(user_id, top_k_recommendations, test_top)

    return len(top_k_locations_visits_user) / len(total_visits_user)


# get_top_k_locations_ratio(1, 10, test)


def get_user_similar_pois(df_ratings_avg, df_pois, user1, user2, k=None):
    common_pois = df_ratings_avg[df_ratings_avg.user_id == user1].merge(
        df_ratings_avg[df_ratings_avg.user_id == user2],
        on="poi_id",
        how="inner")

    merged_common_pois = common_pois.merge(df_pois, on='poi_id')

    if k:
        return merged_common_pois.sort_values(by=['poi_id'], ascending=False).head(k)
    else:
        return merged_common_pois.sort_values(by=['poi_id'], ascending=False)


def check_similar_pois_values(df_ratings_avg, df_pois, poi_a, poi_b):
    a = get_user_similar_pois(df_ratings_avg, df_pois, poi_a, poi_b)
    a = a.loc[:, ['rating_x_x', 'rating_x_y', 'poi_id']]
    a = a.drop_duplicates()
    a.head()


def cosine_user_item_score(df_ratings_mean, cosine_sim_user_p, cosine_similarity_with_poi, final_poi, user_id, item_id,
                           k):
    # actual_k = cosine_sim_user_p[cosine_sim_user_p.index==user].shape[1]

    avg_user = df_ratings_mean.loc[df_ratings_mean['user_id'] == user_id, 'rating'].values[0]
    a = cosine_sim_user_p[cosine_sim_user_p.index == user_id].values

    """
    # If we changed the K value, need to get the neighbors again    
    if k != actual_k:
        #print('K has changed from {} to {}'.format(actual_k, k))  
        n_neighbours = find_n_neighbours(cosine_similarity_with_poi, k)      
        a = n_neighbours[n_neighbours.index==user].values
        #print('I Have new A :{}'.format(a))
        #cosine_sim_user_p = n_neighbours
    """

    b = a.squeeze().tolist()
    # c = final_poi.loc[:,item]
    if item_id in final_poi.columns:
        c = final_poi.loc[:, item_id]
    else:
        return avg_user
    d = c[c.index.isin(b)]
    f = d[d.notnull()]
    index = f.index.values.squeeze().tolist()
    corr = cosine_similarity_with_poi.loc[user_id, index]
    fin = pd.concat([f, corr], axis=1)
    fin.columns = ['adg_score', 'correlation']
    fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)

    nume = fin['score'].sum()
    deno = fin['correlation'].sum()
    final_score = avg_user + (nume / deno)

    return final_score


def bhatta_user_item_score(df_ratings_mean, bhatta_sim_user_p, bhatta_final_poi, bhatta_similarity_with_poi,
                           user_id, item_id, k):
    # actual_k = bhatta_sim_user_p[bhatta_sim_user_p.index==user].shape[1]
    avg_user = df_ratings_mean.loc[df_ratings_mean['user_id'] == user_id, 'rating'].values[0]
    a = bhatta_sim_user_p[bhatta_sim_user_p.index == user_id].values

    """
    # If we changed the K value, need to get the neighbors again    
    if k != actual_k:
        #print('K has changed')  
        n_neighbours = find_n_neighbours(bhatta_similarity_with_poi, k)      
        a = n_neighbours[n_neighbours.index==user].values
        #print('I Have new A :{}'.format(a))
    """

    b = a.squeeze().tolist()
    # c = bhatta_final_poi.loc[:, item_id]
    c = bhatta_final_poi.loc[item_id, :]
    if c.empty:
        return avg_user
    if False:
        if item_id in bhatta_final_poi.columns:
            c = bhatta_final_poi.loc[:, item_id]
        else:
            return avg_user

    d = c[c.index.isin(b)]
    if d.empty:
        return avg_user

    f = pd.Series(d[d.notnull()], name='adg_score')
    if f.empty:
        return avg_user

    index = f.index.values.squeeze().tolist()
    corr = bhatta_similarity_with_poi.loc[user_id, index]
    if type(corr) is np.float64:
        corr = pd.Series([corr], name='correlation')
    if corr.empty:
        return avg_user

    # fin = pd.concat([f, corr], axis=1)
    fin = pd.concat([f.reset_index(drop=True), corr.reset_index(drop=True)], axis=1)
    fin.columns = ['adg_score', 'correlation']
    fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
    nume = fin['score'].sum()
    deno = fin['correlation'].sum()
    final_score = (nume / deno)
    if not math.isnan(final_score):
        return avg_user + final_score

    return avg_user


def bhatta_adj_user_item_score(df_ratings_mean, bhatta_sim_user_p_adj, bhatta_final_poi, user_id, item_id, k):
    # actual_k = bhatta_sim_user_p[bhatta_sim_user_p.index==user].shape[1]
    avg_user = df_ratings_mean.loc[df_ratings_mean['user_id'] == user_id, 'rating'].values[0]
    a = bhatta_sim_user_p_adj[bhatta_sim_user_p_adj.index == user_id].values

    """
    # If we changed the K value, need to get the neighbors again    
    if k != actual_k:
        #print('K has changed')  
        n_neighbours = find_n_neighbours(bhatta_similarity_with_poi, k)      
        a = n_neighbours[n_neighbours.index==user].values
        #print('I Have new A :{}'.format(a))
    """

    b = a.squeeze().tolist()
    # c = bhatta_final_poi_adj.loc[:, item_id]
    c = bhatta_final_poi.loc[item_id, :]
    if c.empty:
        return avg_user
    if False:
        if item_id in bhatta_final_poi.columns:
            c = bhatta_final_poi.loc[:, item_id]
        else:
            return avg_user

    d = c[c.index.isin(b)]
    if d.empty:
        return avg_user

    f = d[d.notnull()]
    if f.empty:
        return avg_user

    index = f.index.values.squeeze().tolist()
    corr = bhatta_similarity_with_poi_adj.loc[user_id, index]
    if type(corr) is np.float64: corr = pd.Series([corr])
    if corr.empty:
        return avg_user

    fin = pd.concat([f, corr], axis=1)
    fin.columns = ['adg_score', 'correlation']
    fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
    nume = fin['score'].sum()
    deno = fin['correlation'].sum()
    final_score = avg_user + (nume / deno)

    return final_score


def pearson_user_item_score(df_ratings_mean, user_id, pearson_sim_user_p, pearson_final_poi,
                            pearson_similarity_with_poi, item_id, k):
    # actual_k = pearson_sim_user_p[pearson_sim_user_p.index==user].shape[1]
    avg_user = df_ratings_mean.loc[df_ratings_mean['user_id'] == user_id, 'rating'].values[0]
    a = pearson_sim_user_p[pearson_sim_user_p.index == user_id].values

    """
    # If we changed the K value, need to get the neighbors again    
    if k != actual_k:
        #print('K has changed')  
        n_neighbours = find_n_neighbours(pearson_similarity_with_poi, k)      
        a = n_neighbours[n_neighbours.index==user].values
        #print('I Have new A :{}'.format(a))
    """

    b = a.squeeze().tolist()
    if item_id in pearson_final_poi.columns:
        c = pearson_final_poi.loc[:, item_id]
    else:
        return avg_user
    d = c[c.index.isin(b)]
    f = d[d.notnull()]
    index = f.index.values.squeeze().tolist()
    corr = pearson_similarity_with_poi.loc[user_id, index]
    if type(corr) is np.float64: corr = pd.Series([corr])
    fin = pd.concat([f, corr], axis=1)
    fin.columns = ['adg_score', 'correlation']
    fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
    nume = fin['score'].sum()
    deno = fin['correlation'].sum()
    final_score = avg_user + (nume / deno)
    return final_score


def fill_poi_user(df_ratings_avg):

    df_ratings_avg = df_ratings_avg.loc[:, ~df_ratings_avg.columns.duplicated()]

    if 'poi_id' not in df_ratings_avg.columns.to_list():
        df_ratings_avg['poi_id'] = df_ratings_avg['item_id']
        # df_ratings_avg = df_ratings_avg.rename(columns={'poi_id': 'item_id'}, inplace=True)

    df_ratings_avg = df_ratings_avg.astype({'poi_id': str})
    poi_user = df_ratings_avg.groupby(by='user_id')['poi_id'].apply(lambda x: ','.join(x))
    # poi_user['user_id'] = poi_user['user_id'].astype('int64')
    # poi_user['poi_id'] = poi_user['poi_id'].astype('int64')
    df_ratings_avg['poi_id'] = df_ratings_avg['poi_id'].astype('int64')

    return poi_user, df_ratings_avg


def pearson_user_item_score_full(df_ratings_mean, cosine_sim_user_p, cosine_similarity_with_poi, poi_user, final_poi,
                                 df_pois, check, user, k_recommendations, k_neighbors):
    item_rated_by_user = check.columns[check[check.index == user].notna().any()].tolist()
    a = cosine_sim_user_p[cosine_sim_user_p.index == user].values

    # If we changed the K value, need to get the neighbors again
    # if k_neighbors and len(cosine_sim_user_p) != k_neighbors:
    #    cosine_sim_user_p = find_n_neighbours(cosine_similarity_with_poi, num_similar_locations)

    b = a.squeeze().tolist()
    d = poi_user[poi_user.index.isin(b)]
    l = ','.join(d.values)
    item_rated_by_similar_users = l.split(',')
    items_under_consideration = list(set(item_rated_by_similar_users) - set(list(map(str, item_rated_by_user))))
    items_under_consideration = list(map(int, items_under_consideration))
    score = []

    for item in items_under_consideration:
        c = final_poi.loc[:, item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        avg_user = df_ratings_mean.loc[df_ratings_mean['user_id'] == user, 'rating'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = cosine_similarity_with_poi.loc[user, index]
        if type(corr) is np.float64: corr = pd.Series([corr])
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_score', 'correlation']
        fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume / deno)
        score.append(final_score)

    data = pd.DataFrame({'poi_id': items_under_consideration, 'score': score})
    top_k_recommendations = data.sort_values(by='score', ascending=False).head(k_recommendations)
    item_name = top_k_recommendations.merge(df_pois, how='inner', on='poi_id')
    item_names = item_name.poi_id.values.tolist()

    return item_names


# This function finds k similar users given the user_id and ratings matrix M
# Note that the similarities are same as obtained via using pairwise_distances
def find_k_similar_users(user_id, rating_matrix, metric='cosine', k=100, print_stats=False):
    from sklearn.neighbors import NearestNeighbors

    similarities = []
    indices = []
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(rating_matrix)

    distances, indices = model_knn.kneighbors(rating_matrix.iloc[user_id - 1, :].values.reshape(1, -1),
                                              n_neighbors=k + 1)
    similarities = 1 - distances.flatten()

    if print_stats:
        print('{0} most similar users for User {1}:\n'.format(k, user_id))
        for i in range(0, len(indices.flatten())):

            if indices.flatten()[i] + 1 == user_id:
                continue;
            else:
                print('{0}: User {1}, with similarity of {2}'.format(i, indices.flatten()[i] + 1,
                                                                     similarities.flatten()[i]))

    return similarities, indices


def bhatta_user_item_scores_full(df_ratings_mean, bhatta_sim_user_p, bhatta_final_user, bhatta_final_poi,
                                 bhatta_similarity_with_poi, poi_user, df_pois,
                                 check, user, k_recommendations, k_neighbors):
    item_rated_by_user = check.columns[check[check.index == user].notna().any()].tolist()
    avg_user = df_ratings_mean.loc[df_ratings_mean['user_id'] == user, 'rating'].values[0]
    a = bhatta_sim_user_p[bhatta_sim_user_p.index == user].values

    # If we changed the K value, need to get the neighbors again
    # if k_neighbors and len(bhatta_sim_user_p) != k_neighbors:
    #    bhatta_sim_user_p = find_n_neighbours(bhatta_similarity_with_poi, num_similar_locations)

    b = a.squeeze().tolist()
    d = poi_user[poi_user.index.isin(b)]
    if d.empty:
        while d.empty:
            # similarities, indices = find_k_similar_users(35, find_k_similar_users, 'cosine', num_recommendations, True)
            similarities, bhatta_sim_user_p_temp = find_k_similar_users(user, bhatta_final_user, 'cosine',
                                                                        k_recommendations, False)

            b = [x + 1 for x in bhatta_sim_user_p_temp]
            d = poi_user[poi_user.index.isin(b)]

    l = ','.join(d.values)
    item_rated_by_similar_users = l.split(',')
    items_under_consideration = list(set(item_rated_by_similar_users) - set(list(map(str, item_rated_by_user))))
    items_under_consideration = list(map(int, items_under_consideration))
    scores = []

    for item in items_under_consideration:

        # c = bhatta_final_poi.loc[:, item]
        c = bhatta_final_poi.loc[item, :]
        if c.empty:
            print('c is empty, returnig user average')
            scores.append(avg_user)
            continue

        d = c[c.index.isin(b)]
        if d.empty:
            print('d is empty, returnig user average')
            scores.append(avg_user)
            continue

        f = d[d.notnull()]
        if f.empty:
            print('f is empty, returnig user average')
            scores.append(avg_user)
            continue

        index = f.index.values.squeeze().tolist()
        corr = bhatta_similarity_with_poi.loc[user, index]
        if type(corr) is np.float64: corr = pd.Series([corr])
        if corr.empty:
            print('corr is empty, returnig user average')
            scores.append(avg_user)
            continue

        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_scores', 'correlation']
        fin['scores'] = fin.apply(lambda x: x['adg_scores'] * x['correlation'], axis=1)

        nume = fin['scores'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume / deno)

        scores.append(final_score)

    data = pd.DataFrame({'poi_id': items_under_consideration, 'score': scores})
    top_k_recommendations = data.sort_values(by='score', ascending=False).head(k_recommendations)
    item_name_df_pois = top_k_recommendations.merge(df_pois, how='inner', on='poi_id')
    item_names = item_name_df_pois.poi_id.values.tolist()

    return item_names, top_k_recommendations


def bhatta_adj_user_item_scores_full(df_ratings_mean, bhatta_sim_user_p_adj, bhatta_final_user_adj,
                                     bhatta_final_poi_adj, bhatta_similarity_with_poi_adj, poi_user,
                                     df_pois, check, user, k_recommendations, k_neighbors):
    item_rated_by_user = check.columns[check[check.index == user].notna().any()].tolist()
    avg_user = df_ratings_mean.loc[df_ratings_mean['user_id'] == user, 'rating'].values[0]
    a = bhatta_sim_user_p_adj[bhatta_sim_user_p_adj.index == user].values

    # If we changed the K value, need to get the neighbors again
    # if k_neighbors and len(bhatta_sim_user_p) != k_neighbors:
    #    bhatta_sim_user_p = find_n_neighbours(bhatta_similarity_with_poi, num_similar_locations)

    b = a.squeeze().tolist()
    d = poi_user[poi_user.index.isin(b)]
    if d.empty:
        while d.empty:
            # similarities, indices = find_k_similar_users(35, find_k_similar_users, 'cosine', num_recommendations, True)
            similarities, bhatta_sim_user_p_adj_temp = find_k_similar_users(user, bhatta_final_user_adj, 'cosine',
                                                                            k_recommendations, False)

            b = [x + 1 for x in bhatta_sim_user_p_adj_temp]
            d = poi_user[poi_user.index.isin(b)]

    l = ','.join(d.values)
    item_rated_by_similar_users = l.split(',')
    items_under_consideration = list(set(item_rated_by_similar_users) - set(list(map(str, item_rated_by_user))))
    items_under_consideration = list(map(int, items_under_consideration))
    scores = []

    for item in items_under_consideration:

        # c = bhatta_final_poi.loc[:, item]
        c = bhatta_final_poi_adj.loc[item, :]
        if c.empty:
            print('c is empty, returnig user average')
            scores.append(avg_user)
            continue

        d = c[c.index.isin(b)]
        if d.empty:
            print('d is empty, returnig user average')
            scores.append(avg_user)
            continue

        f = d[d.notnull()]
        if f.empty:
            print('f is empty, returnig user average')
            scores.append(avg_user)
            continue

        index = f.index.values.squeeze().tolist()
        corr = bhatta_similarity_with_poi_adj.loc[user, index]
        if type(corr) is np.float64: corr = pd.Series([corr])
        if corr.empty:
            print('corr is empty, returnig user average')
            scores.append(avg_user)
            continue

        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_scores', 'correlation']
        fin['scores'] = fin.apply(lambda x: x['adg_scores'] * x['correlation'], axis=1)

        nume = fin['scores'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume / deno)

        scores.append(final_score)

    data = pd.DataFrame({'poi_id': items_under_consideration, 'score': scores})
    top_k_recommendations = data.sort_values(by='score', ascending=False).head(k_recommendations)
    item_name_df_pois = top_k_recommendations.merge(df_pois, how='inner', on='poi_id')
    item_names = item_name_df_pois.poi_id.values.tolist()

    return item_names, top_k_recommendations


def pearson_user_item_score_full(df_ratings_mean, pearson_sim_user_p, pearson_final_poi, pearson_similarity_with_poi,
                                 poi_user, df_pois, check, user, k_recommendations, k_neighbors):
    item_rated_by_user = check.columns[check[check.index == user].notna().any()].tolist()
    a = pearson_sim_user_p[pearson_sim_user_p.index == user].values

    # If we changed the K value, need to get the neighbors again
    # if k_neighbors and len(pearson_sim_user_p) != k_neighbors:
    #    pearson_sim_user_p = find_n_neighbours(pearson_similarity_with_poi, num_similar_locations)

    b = a.squeeze().tolist()
    d = poi_user[poi_user.index.isin(b)]
    l = ','.join(d.values)
    item_rated_by_similar_users = l.split(',')
    items_under_consideration = list(set(item_rated_by_similar_users) - set(list(map(str, item_rated_by_user))))
    items_under_consideration = list(map(int, items_under_consideration))
    score = []

    for item in items_under_consideration:
        c = pearson_final_poi.loc[:, item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        avg_user = df_ratings_mean.loc[df_ratings_mean['user_id'] == user, 'rating'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = pearson_similarity_with_poi.loc[user, index]
        if type(corr) is np.float64: corr = pd.Series([corr])
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_score', 'correlation']
        fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume / deno)
        score.append(final_score)

    data = pd.DataFrame({'poi_id': items_under_consideration, 'score': score})
    top_k_recommendations = data.sort_values(by='score', ascending=False).head(k_recommendations)
    item_name = top_k_recommendations.merge(df_pois, how='inner', on='poi_id')
    item_names = item_name.poi_id.values.tolist()

    return item_names, top_k_recommendations


def fill_cosine_scores(df_ratings_mean, cosine_sim_user_p, cosine_similarity_with_poi, final_poi, cosine_testset, k,
                       print_MAE_RMSE_calc=False):
    cosine_testset['cosine'] = np.nan
    for user_id in cosine_testset.user_id.unique():
        for poi_id in cosine_testset[cosine_testset.user_id == user_id].poi_id.sort_values().unique():
            # print('user_item_score_bhatta --- user_id: {} x poi_id {}'.format(user_id, poi_id))
            score = cosine_user_item_score(df_ratings_mean, cosine_sim_user_p, cosine_similarity_with_poi,
                                           final_poi, user_id, poi_id, k)
            cosine_testset.loc[((cosine_testset.user_id == user_id) &
                                (cosine_testset.poi_id == poi_id)), 'cosine'] = score

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(cosine_testset.actual, cosine_testset.cosine))
        print("RMSE: ", recmetrics.rmse(cosine_testset.actual, cosine_testset.cosine))

    return cosine_testset


def fill_bhattacharyya_scores(df_ratings_mean, bhatta_sim_user_p, bhatta_similarity_with_poi, bhatta_final_poi, testset,
                              k,
                              print_MAE_RMSE_calc=False):
    # df_ratings_mean, bhatta_sim_user_p, bhatta_final_poi, bhatta_similarity_with_poi, user_id, item_id, k
    # bhattacharyya_predictions = []
    bhattacharyya_testset = testset
    bhattacharyya_testset['bhattacharyya'] = np.nan
    for user_id in testset.user_id.unique():
        for poi_id in testset[testset.user_id == user_id].poi_id.sort_values().unique():
            if testset.loc[((testset.user_id == user_id) & (testset.poi_id == poi_id))].shape[0] > 0:
                # print('user_item_score_bhatta --- user_id: {} x poi_id {}'.format(user_id, poi_id))
                # print('values testset {}'.format(testset.loc[((testset.user_id == user_id) & (testset.poi_id == poi_id))]))

                score = bhatta_user_item_score(df_ratings_mean, bhatta_sim_user_p, bhatta_final_poi,
                                               bhatta_similarity_with_poi, user_id, poi_id, k)
                bhattacharyya_testset.loc[
                    ((testset.user_id == user_id) & (testset.poi_id == poi_id)), 'bhattacharyya'] = score
                # bhattacharyya_predictions.append((user_id, poi_id, score))

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(bhattacharyya_testset.actual, bhattacharyya_testset.bhattacharyya))
        print("RMSE: ", recmetrics.rmse(bhattacharyya_testset.actual, bhattacharyya_testset.bhattacharyya))

    return bhattacharyya_testset


def fill_pearson_scores(testset, k, print_MAE_RMSE_calc=False):
    # pearson_predictions = []
    pearson_testset = testset
    pearson_testset['pearson'] = np.nan
    for user_id in testset.user_id.unique():
        for poi_id in testset[testset.user_id == user_id].poi_id.sort_values().unique():
            # print('user_item_score_pearson --- user_id: {} x poi_id {}'.format(user_id, poi_id))
            score = pearson_user_item_score(user_id, poi_id, k)
            pearson_testset.loc[((testset.user_id == user_id) & (testset.poi_id == poi_id)), 'pearson'] = score
            # pearson_predictions.append((user_id, poi_id, score))

    if print_MAE_RMSE_calc:

        import recmetrics
        if False:
            pearson_testset_metrics = pearson_testset.copy()

            # Missing data are nan's
            # mask = pearson_testset_metrics.isnan(pearson_testset)
            # pearson_testset_metrics = (pearson_testset_metrics[~mask]-pearson_testset_metrics[~mask])

            pearson_testset_metrics = pearson_testset_metrics.dropna()

            # pearson_testset_metrics.fillna(0,inplace=True)
            print("MSE: ", recmetrics.mse(pearson_testset_metrics.actual, pearson_testset_metrics.pearson))
            print("RMSE: ", recmetrics.rmse(pearson_testset_metrics.actual, pearson_testset_metrics.pearson))
        else:
            print("MSE: ", recmetrics.mse(pearson_testset.actual, pearson_testset.pearson))
            print("RMSE: ", recmetrics.rmse(pearson_testset.actual, pearson_testset.pearson))

    return pearson_testset


def fill_KNNBasic_pearson_baseine_scores(trainset, testset, k, print_MAE_RMSE_calc=False):
    from surprise import KNNBasic

    sim_options = {'name': 'pearson_baseline',
                   'shrinkage': 0  # no shrinkage
                   }
    algo = KNNBasic(sim_options=sim_options)
    # algo = KNNBasic(sim_options=sim_options, k=k)
    algo.fit(trainset)

    test = algo.test(testset)

    test = pd.DataFrame(test)
    test.drop("details", inplace=True, axis=1)
    test.columns = ['user_id', 'poi_id', 'actual', 'KNNBasic_pearson_baseline']

    test['user_id'] = test['user_id'].astype('int64')
    test['poi_id'] = test['poi_id'].astype('int64')

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNBasic_pearson_baseline))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNBasic_pearson_baseline))

    return test


def fill_KNNBasic_pearson_scores(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import KNNBasic

    sim_options = {'name': 'pearson',
                   'shrinkage': 0  # no shrinkage
                   }
    algo = KNNBasic(sim_options=sim_options)
    # algo = KNNBasic(sim_options=sim_options, k=k)
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', 'KNNBasic_pearson']

    test['KNNBasic_pearson'] = predictions['KNNBasic_pearson']

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNBasic_pearson))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNBasic_pearson))

    return test


def fill_KNNWithMeans_pearson_scores(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import KNNWithMeans

    # To use item-based cosine similarity
    sim_options = {
        "name": "pearson",
        "user_based": False,  # Compute  similarities between items
    }
    algo = KNNWithMeans(sim_options=sim_options)
    # algo = KNNWithMeans(sim_options=sim_options, k=k)
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', 'KNNWithMeans_pearson']

    test['KNNWithMeans_pearson'] = predictions['KNNWithMeans_pearson']

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNWithMeans_pearson))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNWithMeans_pearson))

    return test


def fill_KNNBasic_scores(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import KNNBasic

    algo = KNNBasic()
    # algo = KNNBasic(k=k)
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', 'KNNBasic']

    test['KNNBasic'] = predictions['KNNBasic']

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNBasic))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNBasic))

    return test


def fill_KNNBasic_msd_scores(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import KNNBasic

    sim_options = {
        "name": "msd"
    }

    algo = KNNBasic(sim_options)
    # algo = KNNBasic(sim_options, k)
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', 'KNNBasic_msd']

    test['KNNBasic_msd'] = predictions['KNNBasic_msd']

    # prediction = algo.predict(4, 1645)
    # prediction.est

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNBasic_msd))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNBasic_msd))

    return test


def fill_NormalPredictor_scores(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import NormalPredictor

    algo = NormalPredictor()
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', 'NormalPredictor']

    test['NormalPredictor'] = predictions['NormalPredictor']

    # prediction = algo.predict(4, 1645)

    # prediction.est
    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.NormalPredictor))
        print("RMSE: ", recmetrics.rmse(test.actual, test.NormalPredictor))

    return test


def fill_KNNBaseline_scores(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import KNNBaseline

    algo = KNNBaseline()
    # algo = KNNBaseline(k=k)
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', 'KNNBaseline']

    test['KNNBaseline'] = predictions['KNNBaseline']

    # prediction = algo.predict(4, 1645)
    # prediction.est

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNBaseline))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNBaseline))

    return test


def fill_BaselineOnly_scores(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import BaselineOnly

    algo = BaselineOnly()
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', 'BaselineOnly']

    test['BaselineOnly'] = predictions['BaselineOnly']

    # prediction = algo.predict(4, 1645)
    # prediction.est

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.BaselineOnly))
        print("RMSE: ", recmetrics.rmse(test.actual, test.BaselineOnly))

    return test


def clean_recommendations_list(my_predictions_bhatta):
    clean_user_id_list = [x[0] for x in set([r for r in my_predictions_bhatta])]
    clean_poi_id_list = [x[1] for x in set([r for r in my_predictions_bhatta])]
    clean_score_id_list = [x[2] for x in set([r for r in my_predictions_bhatta])]

    return clean_user_id_list, clean_poi_id_list, clean_score_id_list


def make_train(ratings, pct_test=0.2):
    test_set = ratings.copy()  # Make a copy of the original set to be the test set.
    test_set[test_set != 0] = 1  # Store the test set as a binary preference matrix

    training_set = ratings.copy()  # Make a copy of the original data we can alter as our training set.

    nonzero_inds = training_set.nonzero()  # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))  # Zip these pairs together of item,user index into list

    random.seed(0)  # Set the random seed to zero for reproducibility

    num_samples = int(
        np.ceil(pct_test * len(nonzero_pairs)))  # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples)  # Sample a random number of item-user pairs without replacement

    content_inds = [index[0] for index in samples]  # Get the item row indices

    person_inds = [index[1] for index in samples]  # Get the user column indices

    training_set[content_inds, person_inds] = 0  # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros()  # Get rid of zeros in sparse array storage after update to save space

    return training_set, test_set, list(set(person_inds))


def find_n_neighbours(df, n):
    if df.shape[0] <= n:
        n = df.shape[0]

    order = np.argsort(df.values, axis=1)[:, :n+1]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
                                      .iloc[:n+1].index,
                                      index=['top{}'.format(i) for i in range(0, n + 1)]), axis=1)
    return df.iloc[:, 1:]


def set_cosine_matrices(final_user, final_poi):
    # user similarity on replacing NAN by user avg
    cosine_user = cosine_similarity(final_user)
    np.fill_diagonal(cosine_user, 1)
    cosine_similarity_with_user = pd.DataFrame(cosine_user, index=final_user.index)
    cosine_similarity_with_user.columns = final_user.index

    # user similarity on replacing NAN by item(poi) avg
    cosine_poi = cosine_similarity(final_poi)
    np.fill_diagonal(cosine_poi, 1)
    cosine_similarity_with_poi = pd.DataFrame(cosine_poi, index=final_poi.index)
    cosine_similarity_with_poi.columns = final_poi.index

    return cosine_user, cosine_similarity_with_user, cosine_poi, cosine_similarity_with_poi


def set_bhattacharyya_matrices(bhatta_final_user, bhatta_final_poi):
    # user similarity on replacing NAN by user avg
    bhatta_user = sm.bhattacharyya_similarity(bhatta_final_user, with_adjustment_factor=False)
    np.fill_diagonal(bhatta_user, 1)
    bhatta_similarity_with_user = pd.DataFrame(bhatta_user, index=bhatta_final_user.index)
    bhatta_similarity_with_user.columns = bhatta_final_user.index

    # user similarity on replacing NAN by item(poi) avg
    bhatta_poi = sm.bhattacharyya_similarity(bhatta_final_poi, with_adjustment_factor=False)
    np.fill_diagonal(bhatta_poi, 1)
    bhatta_similarity_with_poi = pd.DataFrame(bhatta_poi, index=bhatta_final_poi.index)
    bhatta_similarity_with_poi.columns = bhatta_final_poi.index

    return bhatta_user, bhatta_similarity_with_user, bhatta_poi, bhatta_similarity_with_poi


def set_bhattacharyya_adj_matrices(bhatta_final_user_adj, bhatta_final_poi_adj):
    # user similarity on replacing NAN by user avg
    bhatta_user_adj = sm.bhattacharyya_similarity(bhatta_final_user_adj, with_adjustment_factor=True)
    np.fill_diagonal(bhatta_user_adj, 1)
    bhatta_similarity_with_user_adj = pd.DataFrame(bhatta_user_adj, index=bhatta_final_user_adj.index)
    bhatta_similarity_with_user_adj.columns = bhatta_final_user_adj.index

    # user similarity on replacing NAN by item(poi) avg
    bhatta_poi_adj = sm.bhattacharyya_similarity(bhatta_final_poi_adj, with_adjustment_factor=True)
    np.fill_diagonal(bhatta_poi_adj, 1)
    bhatta_similarity_with_poi_adj = pd.DataFrame(bhatta_poi_adj, index=bhatta_final_poi_adj.index)
    bhatta_similarity_with_poi_adj.columns = bhatta_final_poi_adj.index

    return bhatta_user_adj, bhatta_similarity_with_user_adj, bhatta_poi_adj, bhatta_similarity_with_poi_adj


def set_pearson_matrices(pearson_final_user, pearson_final_poi):
    # user similarity on replacing NAN by user avg
    pearson_user = sm.pearson_similarity(pearson_final_user)
    np.fill_diagonal(pearson_user, 1)
    pearson_similarity_with_user = pd.DataFrame(pearson_user, index=pearson_final_user.index)
    pearson_similarity_with_user.columns = pearson_final_user.index

    # user similarity on replacing NAN by user avg
    pearson_poi = sm.pearson_similarity(pearson_final_poi)
    np.fill_diagonal(pearson_poi, 1)
    pearson_similarity_with_poi = pd.DataFrame(pearson_poi, index=pearson_final_poi.index)
    pearson_similarity_with_poi.columns = pearson_final_poi.index

    return pearson_user, pearson_similarity_with_user, pearson_poi, pearson_similarity_with_poi


def evaluate_bench_2(x_test, y_test, df_test, min_num_recommendations, max_num_recommendations, k_neighbours_step):
    from recommender_system.evaluation_metrics import calculate_mape
    print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))

    df_test['actual'] = y_test

    test = pd.DataFrame(y_test, columns=['actual'])
    df_benchmark = pd.DataFrame(columns=['model', 'k', 'mae', 'rmse', 'mape'])

    metrics = ['bhattacharyya', 'cosine', 'euclidean', 'adjusted_cosine', 'correlation', 'jaccard', 'cosine_adj',
               'SBCF_adj']

    for n_k in range(min_num_recommendations, max_num_recommendations, k_neighbours_step):

        print('************ Working on K: {} ************'.format(n_k))

        mae_scores = []
        rmse_scores = []
        mape_scores = []

        # ['bhattacharyya', 'cosine', 'euclidean', 'adjusted_cosine', 'correlation', 'jaccard', 'cosine_adj', 'SBCF_adj']

        for metric in metrics:

            model_name = '{}-{}'.format(metric, n_k)

            try:

                if metric == 'cosine':
                    similarities = cosine_similarities_bkp[:, 1:n_k]
                    neighbors = cosine_neighbors_bkp[:, 1:n_k]
                elif metric == 'adjusted_cosine' and False:
                    model_name = 'CSA_adj-{}'.format(n_k)
                    similarities = adjusted_cosine_similarities_bkp[:, 1:n_k]
                    neighbors = adjusted_cosine_neighbors_bkp[:, 1:n_k]
                elif metric == 'cosine_adj':
                    similarities = cosine_adj_similarities_bkp[:, 1:n_k]
                    neighbors = cosine_adj_neighbors_bkp[:, 1:n_k]
                if metric == 'bhattacharyya' and False:
                    model_name = 'SBCF_pen-{}'.format(n_k)
                    similarities = bhattacharyya_similarities_bkp[:, 1:n_k]
                    neighbors = bhattacharyya_neighbors_bkp[:, 1:n_k]
                elif metric == 'SBCF_adj':
                    model_name = 'SBCF-{}'.format(n_k)
                    similarities = SBCF_adj_similarities_bkp[:, 1:n_k]
                    neighbors = SBCF_adj_neighbors_bkp[:, 1:n_k]
                elif metric == 'correlation':
                    model_name = 'PCC-{}'.format(n_k)
                    similarities = correlation_similarities_bkp[:, 1:n_k]
                    neighbors = correlation_neighbors_bkp[:, 1:n_k]
                elif metric == 'jaccard':
                    similarities = jaccard_similarities_bkp[:, 1:n_k]
                    neighbors = jaccard_neighbors_bkp[:, 1:n_k]

            except BaseException as e:
                write_message('Failed to do something: ' + str(e))

            model_name_preds = list(predict(u, i, neighbors, similarities, np_ratings, mean) for (u, i) in x_test)
            test[model_name] = model_name_preds
            mae = mean_absolute_error(y_true=test['actual'], y_pred=test[model_name])
            rmse = sqrt(mean_squared_error(y_true=test['actual'], y_pred=test[model_name]))
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            benchmark_model_names.append(model_name)
            df_benchmark = \
                df_benchmark.append(pd.DataFrame([[model_name.split('-')[0], n_k, mae, rmse]],
                                                 columns=['model', 'k', 'mae', 'rmse']), ignore_index=True, sort=False)
            benchmark.update({'model_name': model_name, 'mae': mae, 'rmse': rmse})
            print('{} --- MAE = {} --- RMSE = {}'.format(model_name, mae, rmse))

            mae_scores_benchmark.append(mae_scores)
            rmse_scores_benchmark.append(rmse_scores)
            mape_scores_benchmark.append(mape_scores)

        # original_recommendations_list = get_original_recommendations_list(test)

        # print('Test Dataset with K : {}'.format(k))
        # print(test)

        # test_original_recommendations_list = fill_models_recommendations_list(test_original_recommendations_list, df_test, num_recommendations=n_k)

        if False:
            for user_id in test_original_recommendations_list.index.unique():
                topn, topn_predict = topn_prediction(user_id=user_id, n=n_k)

                # original_recommendations_list = fill_models_recommendations_list(original_recommendations_list, k)
                # original_recommendations_list_benchmark.append(original_recommendations_list)

        if False:
            mae_scores = []
            rmse_scores = []
            for model_name in test.iloc[:, 1:].columns:
                mae = mean_absolute_error(y_true=test['actual'], y_pred=test[model_name])
                rmse = sqrt(mean_squared_error(y_true=test['actual'], y_pred=test[model_name]))

                print('{} --- MAE = {} --- RMSE = {}'.format(model_name, mae, rmse))

                mae_scores.append(mae)
                rmse_scores.append(rmse)
                benchmark_model_names.append(model_name)

                mae_scores_benchmark.append(mae_scores)
                rmse_scores_benchmark.append(rmse_scores)

        if False:
            mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
            print('\nMAE :', mae)
            return mae

    return test, df_benchmark


def test_cosine(testset, df_ratings_mean, cosine_sim_user_p, cosine_similarity_with_poi, final_poi, k,
                print_MAE_RMSE_calc=False):
    # my_predictions_cosine = []
    cosine_testset = testset
    cosine_testset[f'cosine-{k}'] = np.nan
    for user_id in testset.user_id.unique():
        for poi_id in testset[testset.user_id == user_id].poi_id.sort_values().unique():
            # print('user_item_score_bhatta --- user_id: {} x poi_id {}'.format(user_id, poi_id))
            score = cosine_user_item_score(df_ratings_mean, cosine_sim_user_p, cosine_similarity_with_poi,
                                           final_poi, user_id, poi_id, k)
            cosine_testset.loc[((testset.user_id == user_id) & (testset.poi_id == poi_id)), f'cosine-{k}'] = score
            # my_predictions_cosine.append((user_id, poi_id, score))

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(cosine_testset.actual, cosine_testset.cosine))
        print("RMSE: ", recmetrics.rmse(cosine_testset.actual, cosine_testset.cosine))

    return cosine_testset


def test_bhattacharyya(testset, df_ratings_mean, bhatta_sim_user_p, bhatta_similarity_with_poi, bhatta_final_poi, k,
                       print_MAE_RMSE_calc=False):
    # bhattacharyya_predictions = []
    bhattacharyya_testset = testset
    bhattacharyya_testset[f'bhattacharyya-{k}'] = np.nan
    for user_id in testset.user_id.unique():
        for poi_id in testset[testset.user_id == user_id].poi_id.sort_values().unique():
            if testset.loc[((testset.user_id == user_id) & (testset.poi_id == poi_id))].shape[0] > 0:
                # print('user_item_score_bhatta --- user_id: {} x poi_id {}'.format(user_id, poi_id))
                # print('values testset {}'.format(testset.loc[((testset.user_id == user_id) & (testset.poi_id == poi_id))]))

                score = bhatta_user_item_score(df_ratings_mean, bhatta_sim_user_p, bhatta_final_poi,
                                               bhatta_similarity_with_poi, user_id, poi_id, k)
                bhattacharyya_testset.loc[
                    ((testset.user_id == user_id) & (testset.poi_id == poi_id)), f'bhattacharyya-{k}'] = score
                # bhattacharyya_predictions.append((user_id, poi_id, score))

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(bhattacharyya_testset.actual, bhattacharyya_testset.bhattacharyya))
        print("RMSE: ", recmetrics.rmse(bhattacharyya_testset.actual, bhattacharyya_testset.bhattacharyya))

    return bhattacharyya_testset


def test_pearson(testset, k, print_MAE_RMSE_calc=False):
    # pearson_predictions = []
    pearson_testset = testset
    pearson_testset['pearson'] = np.nan
    for user_id in testset.user_id.unique():
        for poi_id in testset[testset.user_id == user_id].poi_id.sort_values().unique():
            # print('user_item_score_pearson --- user_id: {} x poi_id {}'.format(user_id, poi_id))
            score = pearson_user_item_score(user_id, poi_id, k)
            pearson_testset.loc[((testset.user_id == user_id) & (testset.poi_id == poi_id)), 'pearson'] = score
            # pearson_predictions.append((user_id, poi_id, score))

    if print_MAE_RMSE_calc:

        import recmetrics
        if False:
            pearson_testset_metrics = pearson_testset.copy()

            # Missing data are nan's
            # mask = pearson_testset_metrics.isnan(pearson_testset)
            # pearson_testset_metrics = (pearson_testset_metrics[~mask]-pearson_testset_metrics[~mask])

            pearson_testset_metrics = pearson_testset_metrics.dropna()

            # pearson_testset_metrics.fillna(0,inplace=True)
            print("MSE: ", recmetrics.mse(pearson_testset_metrics.actual, pearson_testset_metrics.pearson))
            print("RMSE: ", recmetrics.rmse(pearson_testset_metrics.actual, pearson_testset_metrics.pearson))
        else:
            print("MSE: ", recmetrics.mse(pearson_testset.actual, pearson_testset.pearson))
            print("RMSE: ", recmetrics.rmse(pearson_testset.actual, pearson_testset.pearson))

    return pearson_testset


def test_KNNBasic_pearson_baseine(trainset, testset, k, print_MAE_RMSE_calc=False):
    from surprise import KNNBasic

    sim_options = {'name': 'pearson_baseline',
                   'shrinkage': 0  # no shrinkage
                   }
    algo = KNNBasic(sim_options=sim_options)
    # algo = KNNBasic(sim_options=sim_options, k=k)
    algo.fit(trainset)

    test = algo.test(testset)

    test = pd.DataFrame(test)
    test.drop("details", inplace=True, axis=1)
    test.columns = ['user_id', 'poi_id', 'actual', f'KNNBasic_pearson_baseline-{k}']

    test['user_id'] = test['user_id'].astype('int64')
    test['poi_id'] = test['poi_id'].astype('int64')

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNBasic_pearson_baseline))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNBasic_pearson_baseline))

    return test


def test_KNNBasic_pearson(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import KNNBasic

    sim_options = {'name': 'pearson',
                   'shrinkage': 0  # no shrinkage
                   }
    algo = KNNBasic(sim_options=sim_options)
    # algo = KNNBasic(sim_options=sim_options, k=k)
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', f'KNNBasic_pearson-{k}']

    test['KNNBasic_pearson'] = predictions['KNNBasic_pearson']

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNBasic_pearson))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNBasic_pearson))

    return test


def test_KNNWithMeans_pearson(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import KNNWithMeans

    # To use item-based cosine similarity
    sim_options = {
        "name": "pearson",
        "user_based": False,  # Compute  similarities between items
    }
    algo = KNNWithMeans(sim_options=sim_options)
    # algo = KNNWithMeans(sim_options=sim_options, k=k)
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', f'KNNWithMeans_pearson-{k}']

    test['KNNWithMeans_pearson'] = predictions['KNNWithMeans_pearson']

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNWithMeans_pearson))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNWithMeans_pearson))

    return test


def test_KNNBasic(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import KNNBasic

    algo = KNNBasic()
    # algo = KNNBasic(k=k)
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', f'KNNBasic-{k}']

    test['KNNBasic'] = predictions['KNNBasic']

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNBasic))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNBasic))

    return test


def test_KNNBasic_msd(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import KNNBasic

    sim_options = {
        "name": "msd"
    }

    algo = KNNBasic(sim_options)
    # algo = KNNBasic(sim_options, k)
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', f'KNNBasic_msd-{k}']

    test['KNNBasic_msd'] = predictions['KNNBasic_msd']

    # prediction = algo.predict(4, 1645)
    # prediction.est

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNBasic_msd))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNBasic_msd))

    return test


def test_NormalPredictor(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import NormalPredictor

    algo = NormalPredictor()
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', f'NormalPredictor-{k}']

    test['NormalPredictor'] = predictions['NormalPredictor']

    # prediction = algo.predict(4, 1645)

    # prediction.est
    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.NormalPredictor))
        print("RMSE: ", recmetrics.rmse(test.actual, test.NormalPredictor))

    return test


def test_KNNBaseline(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import KNNBaseline

    algo = KNNBaseline()
    # algo = KNNBaseline(k=k)
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', f'KNNBaseline-{k}']

    test['KNNBaseline'] = predictions['KNNBaseline']

    # prediction = algo.predict(4, 1645)
    # prediction.est

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.KNNBaseline))
        print("RMSE: ", recmetrics.rmse(test.actual, test.KNNBaseline))

    return test


def test_BaselineOnly(trainset, testset, test, k, print_MAE_RMSE_calc=False):
    from surprise import BaselineOnly

    algo = BaselineOnly()
    algo.fit(trainset)

    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)
    predictions.drop("details", inplace=True, axis=1)
    predictions.columns = ['user_id', 'poi_id', 'actual', f'BaselineOnly-{k}']

    test['BaselineOnly'] = predictions['BaselineOnly']

    # prediction = algo.predict(4, 1645)
    # prediction.est

    if print_MAE_RMSE_calc:
        import recmetrics
        print("MSE: ", recmetrics.mse(test.actual, test.BaselineOnly))
        print("RMSE: ", recmetrics.rmse(test.actual, test.BaselineOnly))

    return test


def predict_fast_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = np.squeeze(np.asarray(similarity[i, :][top_k_users])).dot(
                    np.squeeze(np.asarray(ratings[:, j][top_k_users])))
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = np.squeeze(np.asarray(similarity[j, :][top_k_items])).dot(
                    np.squeeze(np.asarray(ratings[i, :][top_k_items])).T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))

    return pred


def predict_nobias(ratings, similarity, kind='user'):
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        pred += user_bias[:, np.newaxis]
    elif kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred += item_bias[np.newaxis, :]

    return pred


def predict_topk_nobias(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        user_bias = ratings.mean(axis=1)
        ratings = (ratings - user_bias[:, np.newaxis]).copy()
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = np.squeeze(np.asarray(similarity[i, :][top_k_users])).dot(
                    np.squeeze(np.asarray(ratings[:, j][top_k_users])))
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        pred += user_bias[:, np.newaxis]
    if kind == 'item':
        item_bias = ratings.mean(axis=0)
        ratings = (ratings - item_bias[np.newaxis, :]).copy()
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = np.squeeze(np.asarray(similarity[j, :][top_k_items])).dot(
                    np.squeeze(np.asarray(ratings[i, :][top_k_items])).T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
        pred += item_bias[np.newaxis, :]

    return pred
