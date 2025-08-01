from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd

from core import config

DATASETS_PATH = config.GEOLIFE_RECOMMENDER_SYSTEM_PATH

DF_RATINGS_PATH = DATASETS_PATH + 'df_ratings_all0104.df'
DF_POIS_PATH = DATASETS_PATH + 'df_pois_all_ratings0104.df'
DF_STAY_POINTS_PATH = DATASETS_PATH + 'df_stay_points_all_ratings0104.df'


# Computing Similarity Between Users on Location-Based Data for Collaborative Filtering
def get_all_ratings_df_2():

    df_ratings = pd.read_pickle(DF_RATINGS_PATH)
    df_ratings['user_id'] = df_ratings['user_id'].astype('int64')
    df_ratings['poi_id'] = df_ratings['poi_id'].astype('int64')
    if 'poi_id' in df_ratings.columns.to_list() and 'item_id' not in df_ratings.columns.to_list():
        df_ratings.rename(columns={'poi_id': 'item_id'}, inplace=True)
    df_ratings = df_ratings.drop('rating', axis=1)
    #df_ratings = df_ratings.rename(columns={'cut_jenks': 'rating', 'poi_id': 'item_id'}, inplace=True)
    df_ratings.rename(columns={'cut_jenks': 'rating', 'poi_id': 'item_id'}, inplace=True)

    df_ratings = df_ratings[df_ratings.rating != -1]
    df_ratings['rating'] = df_ratings.apply(lambda z: z.rating + 1, axis=1)

    df_pois = pd.read_pickle(DF_POIS_PATH)
    df_pois.drop(['cluster_id', 'datetime_start', 'datetime_end', 'time_spent', 'quantity_visits',
                  'category', 'name', 'rating', 'cut_jenks'], axis=1, inplace=True)
    #df_pois = df_pois.rename(columns={'poi_id': 'item_id'}, inplace=True)
    df_pois.rename(columns={'poi_id': 'item_id'}, inplace=True)

    return df_ratings, df_pois


def ratings_matrix(ratings):
    return csr_matrix(
        pd.crosstab(ratings.user_id, ratings.item_id, ratings.norm_rating, aggfunc=sum).fillna(0).values
    )


def item_representation(ratings):
    return csr_matrix(
        pd.crosstab(ratings.item_id, ratings.user_id, ratings.norm_rating, aggfunc=sum).fillna(0).values
    )


def normalize(dataframe):

    if 'userid' in dataframe.columns.to_list():
        dataframe.rename(columns={'userid': 'user_id', 'itemid': 'item_id'}, inplace=True)

    # compute mean rating for each user
    mean = dataframe.groupby(by='user_id', as_index=False)['rating'].mean()
    norm_ratings = pd.merge(dataframe, mean, suffixes=('', '_mean'), on='user_id')

    # normalize each rating by subtracting the mean rating of the corresponding user
    norm_ratings['norm_rating'] = norm_ratings['rating'] - norm_ratings['rating_mean']
    return mean.to_numpy()[:, 1], norm_ratings


def get_examples(dataframe, labels_column="rating"):

    if 'poi_id' in dataframe.columns.to_list() and 'item_id' not in dataframe.columns.to_list():
        dataframe.rename(columns={'poi_id': 'item_id'}, inplace=True)

    examples = dataframe[['user_id', 'item_id']].values
    labels = dataframe[f'{labels_column}'].values
    return examples, labels


def train_test_split(examples, labels, test_size=0.1, verbose=0):
    if verbose:
        print("Train/Test split ")
        print(100-test_size*100, "% of training data")
        print(test_size*100, "% of testing data")

    # split data into train and test sets
    train_examples, test_examples, train_labels, test_labels = sklearn_train_test_split(
        examples,
        labels,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    # transform train and test examples to their corresponding one-hot representations
    train_users = train_examples[:, 0]
    test_users = test_examples[:, 0]

    train_items = train_examples[:, 1]
    test_items = test_examples[:, 1]

    # Final training and test set
    x_train = np.array(list(zip(train_users, train_items)))
    x_test = np.array(list(zip(test_users, test_items)))

    y_train = train_labels
    y_test = test_labels

    if verbose:
        print()
        print('number of training examples : ', x_train.shape)
        print('number of training labels : ', y_train.shape)
        print('number of test examples : ', x_test.shape)
        print('number of test labels : ', y_test.shape)

    return (x_train, x_test), (y_train, y_test)


def mean_ratings(dataframe, labels_column="rating"):
    means = dataframe.groupby(by='user_id', as_index=False)[f'{labels_column}'].mean()
    return means


def normalized_ratings(dataframe, norm_column="norm_rating"):
    """
    Subscribe users mean ratings from each rating
    """
    mean = mean_ratings(dataframe=dataframe)
    norm = pd.merge(dataframe, mean, suffixes=('', '_mean'), on='userid')
    norm[f'{norm_column}'] = norm['rating'] - norm['rating_mean']

    return norm


def rating_matrix(dataframe, column):
    crosstab = pd.crosstab(dataframe.userid, dataframe.itemid, dataframe[f'{column}'], aggfunc=sum).fillna(0).values
    matrix = csr_matrix(crosstab)
    return matrix


def scale_ratings(dataframe, scaled_column="scaled_rating"):
    dataframe[f"{scaled_column}"] = dataframe.rating / 5.0
    return dataframe


def ids_encoder(ratings):

    if 'itemid' in ratings.columns.to_list():
        ratings.rename(columns={'itemid': 'item_id'}, inplace=True)
        ratings.rename(columns={'userid': 'user_id'}, inplace=True)

    users = sorted(ratings['user_id'].unique())
    items = sorted(ratings['item_id'].unique())

    # create users and items encoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    # fit users and items ids to the corresponding encoder
    user_encoder.fit(users)
    item_encoder.fit(items)

    # encode user_ids and item_ids
    ratings.user_id = user_encoder.transform(ratings.user_id.tolist())
    ratings.item_id = item_encoder.transform(ratings.item_id.tolist())

    return ratings, user_encoder, item_encoder


def get_train_test_set_split(my_data, my_columns, my_rating_value, test_size=0.25):
    from surprise import Dataset, Reader
    from surprise.model_selection import train_test_split

    reader = Reader(rating_scale=(my_data[my_rating_value].min(), my_data[my_rating_value].max()))
    data = Dataset.load_from_df(my_data[my_columns], reader)
    train_set, test_set = train_test_split(data, test_size=test_size)

    return train_set, test_set


def get_train_test_set(df_ratings_avg):

    if 'poi_id' in df_ratings_avg.columns.to_list() and 'item_id' not in df_ratings_avg.columns.to_list():
        df_ratings_avg.rename(columns={'poi_id': 'item_id'}, inplace=True)

    df_ratings_avg = df_ratings_avg.loc[:, ~df_ratings_avg.columns.duplicated()]

    my_data = df_ratings_avg
    my_columns = ['user_id', 'item_id', 'adg_rating']
    my_rating_value = 'adg_rating'
    test_size = 0.25

    my_data['user_id'] = my_data['user_id'].astype('int64')
    my_data['item_id'] = my_data['item_id'].astype('int64')

    train_set, test_set = get_train_test_set_split(my_data, my_columns, my_rating_value, test_size)

    return train_set, test_set


def get_examples(dataframe, labels_column="adg_rating", users_column="user_id", items_column="item_id"):

    if 'poi_id' in dataframe.columns.to_list() and 'item_id' not in dataframe.columns.to_list():
        dataframe.rename(columns={'poi_id': 'item_id'}, inplace=True)

    examples = dataframe[[users_column, items_column]].values
    labels = dataframe[f'{labels_column}'].values
    return examples, labels


def train_test_split(examples, labels, test_size=0.25, verbose=1):
    from sklearn.model_selection import train_test_split as sklearn_train_test_split

    if verbose:
        print("Train/Test split ")
        print(100 - test_size * 100, "% of training data")
        print(test_size * 100, "% of testing data")

        # split data into train and test sets
    train_examples, test_examples, train_labels, test_labels = sklearn_train_test_split(
        examples,
        labels,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    # transform train and test examples to their corresponding one-hot representations
    train_users = train_examples[:, 0]
    test_users = test_examples[:, 0]

    train_items = train_examples[:, 1]
    test_items = test_examples[:, 1]

    # Final training and test set
    x_train = np.array(list(zip(train_users, train_items)))
    x_test = np.array(list(zip(test_users, test_items)))

    y_train = train_labels
    y_test = test_labels

    if verbose:
        print()
        print('number of training examples : ', x_train.shape)
        print('number of training labels : ', y_train.shape)
        print('number of test examples : ', x_test.shape)
        print('number of test labels : ', y_test.shape)

    return (x_train, x_test), (y_train, y_test)


def save_all_dfs_predictions(df_ratings_mean, df_ratings_avg,
                             bhatta_sim_user_p, bhatta_sim_user_u,
                             bhatta_similarity_with_poi, bhatta_similarity_with_user,
                             bhatta_final_poi, bhatta_final_user,
                             bhatta_sim_user_p_adj, bhatta_sim_user_u_adj,
                             bhatta_similarity_with_poi_adj, bhatta_similarity_with_user_adj,
                             bhatta_final_poi_adj, bhatta_final_user_adj,
                             calculate_pearson_similarities,
                             pearson_sim_user_p, pearson_sim_user_u,
                             pearson_similarity_with_poi, pearson_similarity_with_user,
                             pearson_final_poi, pearson_final_user


                             ):
    df_ratings_mean.to_pickle(DATASETS_PATH + 'df_ratings_mean.df')
    df_ratings_avg.to_pickle(DATASETS_PATH + 'df_ratings_avg.df')

    bhatta_sim_user_p.to_pickle(DATASETS_PATH + 'bhatta_sim_user_p.df')
    bhatta_sim_user_u.to_pickle(DATASETS_PATH + 'bhatta_sim_user_u.df')
    bhatta_similarity_with_poi.to_pickle(
        DATASETS_PATH + 'bhatta_similarity_with_poi.df')
    bhatta_similarity_with_user.to_pickle(
        DATASETS_PATH + 'bhatta_similarity_with_user.df')
    bhatta_final_poi.to_pickle(DATASETS_PATH + 'bhatta_final_poi.df')
    bhatta_final_user.to_pickle(DATASETS_PATH + 'bhatta_final_user.df')

    bhatta_sim_user_p_adj.to_pickle(
        DATASETS_PATH + 'bhatta_sim_user_p_adj.df')
    bhatta_sim_user_u_adj.to_pickle(
        DATASETS_PATH + 'bhatta_sim_user_u_adj.df')
    bhatta_similarity_with_poi_adj.to_pickle(
        DATASETS_PATH + 'bhatta_similarity_with_poi_adj.df')
    bhatta_similarity_with_user_adj.to_pickle(
        DATASETS_PATH + 'bhatta_similarity_with_user_adj.df')
    bhatta_final_poi_adj.to_pickle(
        DATASETS_PATH + 'bhatta_final_poi_adj.df')
    bhatta_final_user_adj.to_pickle(
        DATASETS_PATH + 'bhatta_final_user_adj.df')

    if calculate_pearson_similarities:
        pearson_sim_user_p.to_pickle(
            DATASETS_PATH + 'pearson_sim_user_p.df')
        pearson_sim_user_u.to_pickle(
            DATASETS_PATH + 'pearson_sim_user_u.df')
        pearson_similarity_with_poi.to_pickle(
            DATASETS_PATH + 'pearson_similarity_with_poi.df')
        pearson_similarity_with_user.to_pickle(
            DATASETS_PATH + 'pearson_similarity_with_user.df')
        pearson_final_poi.to_pickle(
            DATASETS_PATH + 'pearson_final_poi.df')
        pearson_final_user.to_pickle(
            DATASETS_PATH + 'pearson_final_user.df')


def get_pois_and_users_matrices(final, final_pois):
    # Replacing NaN by POI Average
    final_poi = final_pois.fillna(final.mean(axis=0))
    # Replacing NaN by user Average
    final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)

    bhatta_final_poi = final_pois  # .fillna(final.mean(axis=0))
    bhatta_final_user = final  # .apply(lambda row: row.fillna(row.mean()), axis=1)

    bhatta_final_poi_adj = final_pois  # .fillna(final.mean(axis=0))
    bhatta_final_user_adj = final  # .apply(lambda row: row.fillna(row.mean()), axis=1)

    pearson_final_poi = final.fillna(final.mean(axis=0))
    pearson_final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)

    return final_poi, final_user, bhatta_final_poi, bhatta_final_user, pearson_final_poi, pearson_final_user, bhatta_final_poi_adj, bhatta_final_user_adj


# Computing Similarity Between Users on Location-Based Data for Collaborative Filtering
def get_all_ratings_df():

    df_ratings = pd.read_pickle(
        DATASETS_PATH + 'df_ratings_all0104.df')
    df_pois = pd.read_pickle(
        DATASETS_PATH + 'df_pois_all_ratings0104.df')
    df_stay_points = pd.read_pickle(
        DATASETS_PATH + 'df_stay_points_all_ratings0104.df')

    df_ratings['user_id'] = df_ratings['user_id'].astype('int64')
    df_ratings['poi_id'] = df_ratings['poi_id'].astype('int64')
    if 'poi_id' in df_ratings.columns.to_list() and 'item_id' not in df_ratings.columns.to_list():
        df_ratings.rename(columns={'poi_id': 'item_id'}, inplace=True)
    df_ratings = df_ratings.drop('rating', axis=1)
    df_ratings.rename(columns={'cut_jenks': 'rating'}, inplace=True)

    df_ratings = df_ratings[df_ratings.rating != -1]
    df_ratings['rating'] = df_ratings.apply(lambda z: z.rating + 1, axis=1)

    df_ratings_mean = df_ratings.groupby(by="user_id", as_index=False)['rating'].mean()
    df_ratings_mean['user_id'] = df_ratings_mean['user_id'].astype('int64')

    df_ratings_avg = pd.merge(df_ratings, df_ratings_mean, on='user_id')
    df_ratings_avg['user_id'] = df_ratings_avg['user_id'].astype('int64')
    if 'poi_id' in df_ratings_avg.columns.to_list() and 'item_id' not in df_ratings_avg.columns.to_list():
        df_ratings_avg.rename(columns={'poi_id': 'item_id'}, inplace=True)
    df_ratings_avg['item_id'] = df_ratings_avg['item_id'].astype('int64')

    # df_ratings_avg = pd.merge(df_ratings, df_ratings_mean, on='user_id')
    df_ratings_avg['adg_rating'] = (
                df_ratings_avg['rating_x'] - df_ratings_avg['rating_y'])  # + 1 #to prevent negative values

    if False:
        min_max_scaler = preprocessing.MinMaxScaler()
        df_ratings_avg['rating_x'] = min_max_scaler.fit_transform(df_ratings_avg['rating_x'].values.reshape(-1, 1))
        df_ratings_avg['rating_y'] = min_max_scaler.fit_transform(df_ratings_avg['rating_y'].values.reshape(-1, 1))
        df_ratings_avg['adg_rating'] = min_max_scaler.fit_transform(df_ratings_avg['adg_rating'].values.reshape(-1, 1))

        df_ratings_mean['rating'] = min_max_scaler.fit_transform(df_ratings_mean['rating'].values.reshape(-1, 1))

    # df_ratings_avg.head()

    #check = pd.pivot_table(df_ratings_avg, values='rating_x', index='user_id', columns='poi_id')
    check = pd.pivot_table(df_ratings_avg, values='rating_x', index='user_id', columns='item_id')
    # check.head()

    #final = pd.pivot_table(df_ratings_avg, values='adg_rating', index='user_id', columns='poi_id')
    #final_pois = pd.pivot_table(df_ratings_avg, values='adg_rating', index='poi_id', columns='user_id')

    final = pd.pivot_table(df_ratings_avg, values='adg_rating', index='user_id', columns='item_id')
    final_pois = pd.pivot_table(df_ratings_avg, values='adg_rating', index='item_id', columns='user_id')

    # final.head()

    return df_ratings, df_ratings_avg, df_ratings_mean, df_pois, check, final, final_pois


def read_all_dfs_predictions():
    df_ratings_mean = pd.read_pickle(
        DATASETS_PATH + 'df_ratings_mean.df')
    df_ratings_avg = pd.read_pickle(
        DATASETS_PATH + 'df_ratings_avg.df')
    if 'poi_id' in df_ratings_avg.columns.to_list() and 'item_id' not in df_ratings_avg.columns.to_list():
        df_ratings_avg.rename(columns={'poi_id': 'item_id'}, inplace=True)

    bhatta_sim_user_p = pd.read_pickle(
        DATASETS_PATH + 'bhatta_sim_user_p.df')
    bhatta_sim_user_u = pd.read_pickle(
        DATASETS_PATH + 'bhatta_sim_user_u.df')
    bhatta_similarity_with_poi = pd.read_pickle(
        DATASETS_PATH + 'bhatta_similarity_with_poi.df')
    bhatta_similarity_with_user = pd.read_pickle(
        DATASETS_PATH + 'bhatta_similarity_with_user.df')
    bhatta_final_poi = pd.read_pickle(
        DATASETS_PATH + 'bhatta_final_poi.df')
    bhatta_final_user = pd.read_pickle(
        DATASETS_PATH + 'bhatta_final_user.df')

    pearson_sim_user_p = pd.read_pickle(
        DATASETS_PATH + 'pearson_sim_user_p.df')
    pearson_sim_user_u = pd.read_pickle(
        DATASETS_PATH + 'pearson_sim_user_u.df')
    pearson_similarity_with_poi = pd.read_pickle(
        DATASETS_PATH + 'pearson_similarity_with_poi.df')
    pearson_similarity_with_user = pd.read_pickle(
        DATASETS_PATH + 'pearson_similarity_with_user.df')
    pearson_final_poi = pd.read_pickle(
        DATASETS_PATH + 'pearson_final_poi.df')
    pearson_final_user = pd.read_pickle(
        DATASETS_PATH + 'pearson_final_user.df')

    return df_ratings_mean, df_ratings_avg, bhatta_sim_user_p, bhatta_sim_user_u, \
        bhatta_similarity_with_poi, bhatta_similarity_with_user, bhatta_final_poi, bhatta_final_user, \
        pearson_sim_user_p, pearson_sim_user_u, pearson_similarity_with_poi, pearson_similarity_with_user, \
        pearson_final_poi, pearson_final_user


def make_train(ratings, pct_test=0.2):
    # https://nbviewer.org/github/jmsteinw/Notebooks/blob/master/RecEngine_NB.ipynb
    import random

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


def train_test_split_simple(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        try:
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=10,
                                        replace=True)
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]
        except:
            test_ratings = np.random.choice(ratings[user + 1, :].nonzero()[0],
                                        size=10,
                                        replace=True)
            train[user + 1, test_ratings] = 0.
            test[user + 1, test_ratings] = ratings[user + 1, test_ratings]

    # Test and training are truly disjoint
    if not (np.isnan(train).any() or np.isnan(test).any()):
        assert(np.all((train * test) == 0))
    return train, test


#Computing Similarity Between Users on Location-Based Data for Collaborative Filtering
def get_all_ratings_df_nb():


    #df_stay_points = pd.read_pickle('C:\\MAPi\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\df_stay_points_all_ratings0104.df')

    df_ratings = pd.read_pickle(DF_RATINGS_PATH)
    df_ratings['user_id'] = df_ratings['user_id'].astype('int64')
    df_ratings['poi_id'] = df_ratings['poi_id'].astype('int64')
    df_ratings = df_ratings.drop('rating', axis=1)
    df_ratings = df_ratings.rename(columns={'cut_jenks': 'rating', 'poi_id': 'item_id'})

    df_ratings = df_ratings[df_ratings.rating != -1]
    df_ratings['rating'] = df_ratings.apply(lambda z: z.rating + 1, axis=1)

    df_pois = pd.read_pickle(DF_POIS_PATH)
    df_pois.drop(['cluster_id', 'datetime_start', 'datetime_end', 'time_spent', 'quantity_visits', 'category', 'name', 'rating', 'cut_jenks'], axis=1, inplace=True)
    df_pois.drop(['cluster_min_latitude', 'cluster_min_longitude', 'cluster_max_latitude', 'cluster_max_longitude'], axis=1, inplace=True)

    df_pois = df_pois.rename(columns={'poi_id': 'itemid'})
    #df_ratings = df_ratings.rename(columns={'user_id': 'userid', 'item_id': 'itemid'})

    return df_ratings, df_pois

