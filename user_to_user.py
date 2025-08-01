#!/usr/bin/env python
# coding: utf-8

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from datetime import datetime
from glob import glob

from .preprocessing import ids_encoder

import pandas as pd
import numpy as np
import os


class UserToUser:

    def __init__(self, ratings, items, k=20, predictions_dir='predictions/user2user', metric='cosine'):

        if metric not in ['cosine', 'euclidean', 'correlation']:
            raise Exception('UnknownSimilarityMetric : The similarity metric must be selected among '
                            'the followings : cosine, euclidean. You choosed {}'.format(metric))

        self.ratings, self.user_encoder, self.item_encoder = ids_encoder(ratings)
        self.means, self.ratings = self.prepare_ratings()
        self.ratings_matrix = self.create_ratings_matrix()
        self.k = k
        self.metric = metric
        self.model = self.init_similarity_model()
        self.predictions_dir = predictions_dir
        self.similarities, self.neighbors = self.compute_nearest_neighbors()
        self.items = items

        self.np_ratings = self.ratings.to_numpy()

        os.makedirs(self.predictions_dir, exist_ok=True)
        print('User to user recommendation model created with success ...')

    def create_ratings_matrix(self):
        return csr_matrix(
            pd.crosstab(self.ratings.user_id, self.ratings.item_id, self.ratings.rating, aggfunc=sum).fillna(0).values
        )

    def init_similarity_model(self):
        print('Initialize the similarity model ...')
        model = NearestNeighbors(metric=self.metric, n_neighbors=self.k+1, algorithm='brute')
        model.fit(self.ratings_matrix)
        return model

    def prepare_ratings(self):
        """
        Add to the rating dataframe :
        - mean_ratings : mean rating for all users
        - norm_ratings : normalized ratings for each (user,item) pair
        """
        print('Normalize users ratings ...')
        means = self.ratings.groupby(by='user_id', as_index=False)['rating'].mean()
        means_ratings = pd.merge(self.ratings, means, suffixes=('', '_mean'), on='user_id')
        means_ratings['norm_rating'] = means_ratings['rating'] - means_ratings['rating_mean']

        return means.to_numpy()[:, 1], means_ratings

    def get_user_nearest_neighbors(self, user_id):
        return self.similarities[user_id], self.neighbors[user_id]

    def compute_nearest_neighbors(self):
        print('Compute nearest neighbors ...')
        similarities, neighbors = self.model.kneighbors(self.ratings_matrix)
        return similarities[:, 1:], neighbors[:, 1:]

    def user_rated_items(self, user_id):
        activities = self.np_ratings[self.np_ratings[:, 0] == user_id]
        items = activities[:, 1]
        return items

    def find_user_candidate_items(self, user_id, n=50):
        user_neighbors = self.neighbors[user_id]
        user_rated_items = self.user_rated_items(user_id)

        neighbors_rated_items = self.ratings.loc[self.ratings.user_id.isin(user_neighbors)]

        # sort items in decreasing order of frequency
        items_frequencies = neighbors_rated_items.groupby('item_id')['rating']\
            .count()\
            .reset_index(name='count')\
            .sort_values(['count'], ascending=False)

        neighbors_rated_items_sorted_by_frequency = items_frequencies.item_id
        candidates_items = np.setdiff1d(neighbors_rated_items_sorted_by_frequency, user_rated_items, assume_unique=True)

        return candidates_items[:n]

    def similar_users_who_rated_this_item(self, user_id, item_id):
        """
        :param user_id: target user
        :param item_id: target item
        :return:
        """
        users_who_rated_this_item = self.np_ratings[self.np_ratings[:, 1] == item_id][:, 0]
        sim_users_who_rated_this_item = \
            users_who_rated_this_item[np.isin(users_who_rated_this_item, self.neighbors[user_id])]
        return users_who_rated_this_item, sim_users_who_rated_this_item

    def predict(self, user_id, item_id):
        """
        predict what score user_id would have given to item_id.
        :param user_id:
        :param item_id:
        :return: r_hat : predicted rating of user user_id on item item_id
        """
        user_mean = self.means[user_id]

        user_similarities = self.similarities[user_id]
        user_neighbors = self.neighbors[user_id]

        # find users who rated item 'item_id'
        iratings = self.np_ratings[self.np_ratings[:, 1].astype('int') == item_id]

        # find similar users to 'user_id' who rated item 'item_id'
        suri = iratings[np.isin(iratings[:, 0], user_neighbors)]

        normalized_ratings = suri[:, 4]
        indexes = [np.where(user_neighbors == uid)[0][0] for uid in suri[:, 0].astype('int')]
        sims = user_similarities[indexes]

        num = np.dot(normalized_ratings, sims)
        den = np.sum(np.abs(sims))

        if num == 0 or den == 0:
            return user_mean

        r_hat = user_mean + np.dot(normalized_ratings, sims) / np.sum(np.abs(sims))

        return r_hat

    def evaluate(self, x_test, y_test):
        print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
        preds = list(self.predict(u, i) for (u, i) in x_test)
        mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
        print('\nMAE :', mae)
        return mae

    def user_predictions(self, user_id, predictions_file):
        """
        Make rating prediction for the active user on each candidate item and save in file prediction.csv
        :param user_id : id of the active user
        :param predictions_file : where to save predictions
        """
        # find candidate items for the active user
        candidates = self.find_user_candidate_items(user_id, n=30)

        # loop over candidates items to make predictions
        for item_id in candidates:

            # prediction for user_id on item_id
            r_hat = self.predict(user_id, item_id)

            # save predictions
            with open(predictions_file, 'a+') as file:
                line = f'{user_id},{item_id},{r_hat}\n'
                file.write(line)

    def all_predictions(self):
        """
        Make predictions for each user in the database.
        """
        # get list of users in the database
        users = self.ratings.user_id.unique()

        now = str(datetime.now()).replace(' ', '-').split('.')[0]
        file_name = f'prediction.{now}.csv'
        predictions_file = os.path.join(self.predictions_dir, file_name)

        for user_id in users:
            # make rating predictions for the current user
            self.user_predictions(user_id, predictions_file)

    def make_recommendations(self, user_id):

        uid = self.user_encoder.transform([user_id])[0]
        predictions_files = glob(f'{self.predictions_dir}/*.csv')
        last_predictions = sorted(
            predictions_files, 
            key=lambda file: datetime.fromtimestamp(os.path.getmtime(file)),
            reverse=True
        )[0]

        predictions = pd.read_csv(
            last_predictions, sep=',', 
            names=['user_id', 'item_id', 'predicted_rating']
        )
        predictions = predictions[predictions.user_id == uid]
        recommendation_list = predictions.sort_values(
            by=['predicted_rating'], 
            ascending=False
        )

        recommendation_list.user_id = self.user_encoder.inverse_transform(recommendation_list.user_id.tolist())
        recommendation_list.item_id = self.item_encoder.inverse_transform(recommendation_list.item_id.tolist())

        recommendation_list = pd.merge(
            recommendation_list, 
            self.items,
            on='item_id', 
            how='inner'
        )

        return recommendation_list