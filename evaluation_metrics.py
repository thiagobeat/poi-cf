# -*- coding: utf-8 -*-
#
# Author: Thiago Andrade
#
# Recommender system ranking metrics derived from Spark source for use with
# Python-based recommender libraries (i.e., implicit,
# http://github.com/benfred/implicit/).
# These metrics are derived from the
# original Spark Scala source code for recommender metrics.
# https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/evaluation/RankingMetrics.scala

#from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
from typing import List

import warnings

__all__ = [
    'mean_average_precision',
    'ndcg_at',
    'precision_at',

]
if False:
    try:
        xrange
    except NameError:  # python 3 does not have an 'xrange'
        xrange = range


def auc_score(predictions, test):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_mean_auc(training_set, altered_persons, predictions, test_set):
    store_auc = []  # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = []  # To store popular AUC scores
    pop_contents = np.array(test_set.sum(axis=1)).reshape(-1)  # Get sum of item iteractions to find most popular
    content_vecs = predictions[1]
    for person in altered_persons:  # Iterate through each user that had an item altered
        training_column = training_set[:, person].toarray().reshape(-1)  # Get the training set column
        zero_inds = np.where(training_column == 0)  # Find where the interaction had not yet occurred

        # Get the predicted values based on our user/item vectors
        person_vec = predictions[0][person, :]
        pred = person_vec.dot(content_vecs).toarray()[0, zero_inds].reshape(-1)

        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[:, person].toarray()[zero_inds, 0].reshape(-1)

        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training
        pop = pop_contents[zero_inds]  # Get the item popularity for our chosen items

        store_auc.append(auc_score(pred, actual))  # Calculate AUC for the given user and store

        popularity_auc.append(auc_score(pop, actual))  # Calculate AUC using most popular and score
    # End users iteration

    return float('%.3f' % np.mean(store_auc)), float('%.3f' % np.mean(popularity_auc))


def prediction_coverage(predicted: List[list], catalog: list) -> float:
    """
    Computes the prediction coverage for a list of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', Z]
    Returns
    ----------
    prediction_coverage:
        The prediction coverage of the recommendations as a percent
        rounded to 2 decimal places
    ----------
    Metric Defintion:
    Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
    Beyond accuracy: evaluating recommender systems by coverage and serendipity.
    In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
    """
    predicted_flattened = [p for sublist in predicted for p in sublist]
    unique_predictions = len(set(predicted_flattened))
    prediction_coverage = round(unique_predictions/(len(catalog)* 1.0)*100,2)
    return prediction_coverage


def catalog_coverage(predicted: List[list], catalog: list, k: int) -> float:
    """
    Computes the catalog coverage for k lists of recommendations
    Parameters
    ----------
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', Z]
    k: integer
        The number of observed recommendation lists
        which randomly choosed in our offline setup
    Returns
    ----------
    catalog_coverage:
        The catalog coverage of the recommendations as a percent
        rounded to 2 decimal places
    ----------
    Metric Defintion:
    Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
    Beyond accuracy: evaluating recommender systems by coverage and serendipity.
    In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
    """
    sampling = random.choices(predicted, k=k)
    predicted_flattened = [p for sublist in sampling for p in sublist]
    L_predictions = len(set(predicted_flattened))
    catalog_coverage = round(L_predictions/(len(catalog)*1.0)*100,2)
    return catalog_coverage



def _require_positive_k(k):
    """Helper function to avoid copy/pasted code for validating K"""
    if k <= 0:
        raise ValueError("ranking position k should be positive")


def _mean_ranking_metric(predictions, labels, metric):
    """Helper function for precision_at_k and mean_average_precision"""
    # do not zip, as this will require an extra pass of O(N). Just assert
    # equal length and index (compute in ONE pass of O(N)).
    # if len(predictions) != len(labels):
    #     raise ValueError("dim mismatch in predictions and labels!")
    # return np.mean([
    #     metric(np.asarray(predictions[i]), np.asarray(labels[i]))
    #     for i in xrange(len(predictions))
    # ])

    # Actually probably want lazy evaluation in case preds is a
    # generator, since preds can be very dense and could blow up
    # memory... but how to assert lengths equal? FIXME
    return np.mean([
        metric(np.asarray(prd), np.asarray(labels[i]))
        for i, prd in enumerate(predictions)  # lazy eval if generator
    ])


def _warn_for_empty_labels():
    """Helper for missing ground truth sets"""
    warnings.warn("Empty ground truth set! Check input data")
    return 0.


def precision_at(predictions, labels, k=10, assume_unique=True):
    """Compute the precision at K.
    Compute the average precision of all the queries, truncated at
    ranking position k. If for a query, the ranking algorithm returns
    n (n is less than k) results, the precision value will be computed
    as #(relevant items retrieved) / k. This formula also applies when
    the size of the ground truth set is less than k.
    If a query has an empty ground truth set, zero will be used as
    precision together with a warning.
    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    k : int, optional (default=10)
        The rank at which to measure the precision.
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.
    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> precision_at(preds, labels, 1)
    0.33333333333333331
    >>> precision_at(preds, labels, 5)
    0.26666666666666666
    >>> precision_at(preds, labels, 15)
    0.17777777777777778
    """
    # validate K
    _require_positive_k(k)

    def _inner_pk(pred, lab):
        # need to compute the count of the number of values in the predictions
        # that are present in the labels. We'll use numpy in1d for this (set
        # intersection in O(1))
        if lab.shape[0] > 0:
            n = min(pred.shape[0], k)
            cnt = np.in1d(pred[:n], lab, assume_unique=assume_unique).sum()
            return float(cnt) / k
        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, _inner_pk)


def mean_average_precision(predictions, labels, assume_unique=True):
    """Compute the mean average precision on predictions and labels.
    Returns the mean average precision (MAP) of all the queries. If a query
    has an empty ground truth set, the average precision will be zero and a
    warning is generated.
    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.
    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> mean_average_precision(preds, labels)
    0.35502645502645497
    """

    def _inner_map(pred, lab):
        if lab.shape[0]:
            # compute the number of elements within the predictions that are
            # present in the actual labels, and get the cumulative sum weighted
            # by the index of the ranking
            n = pred.shape[0]

            # Scala code from Spark source:
            # var i = 0
            # var cnt = 0
            # var precSum = 0.0
            # val n = pred.length
            # while (i < n) {
            #     if (labSet.contains(pred(i))) {
            #         cnt += 1
            #         precSum += cnt.toDouble / (i + 1)
            #     }
            #     i += 1
            # }
            # precSum / labSet.size

            arange = np.arange(n, dtype=np.float32) + 1.  # this is the denom
            present = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            prec_sum = np.ones(present.sum()).cumsum()
            denom = arange[present]
            return (prec_sum / denom).sum() / lab.shape[0]

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, _inner_map)


def ndcg_at(predictions, labels, k=10, assume_unique=True):
    """Compute the normalized discounted cumulative gain at K.
    Compute the average NDCG value of all the queries, truncated at ranking
    position k. The discounted cumulative gain at position k is computed as:
        sum,,i=1,,^k^ (2^{relevance of ''i''th item}^ - 1) / log(i + 1)
    and the NDCG is obtained by dividing the DCG value on the ground truth set.
    In the current implementation, the relevance value is binary.
    If a query has an empty ground truth set, zero will be used as
    NDCG together with a warning.
    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    k : int, optional (default=10)
        The rank at which to measure the NDCG.
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.
    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> ndcg_at(preds, labels, 3)
    0.3333333432674408
    >>> ndcg_at(preds, labels, 10)
    0.48791273434956867
    References
    ----------
    .. [1] K. Jarvelin and J. Kekalainen, "IR evaluation methods for
           retrieving highly relevant documents."
    """
    # validate K
    _require_positive_k(k)

    def _inner_ndcg(pred, lab):
        if lab.shape[0]:
            # if we do NOT assume uniqueness, the set is a bit different here
            if not assume_unique:
                lab = np.unique(lab)

            n_lab = lab.shape[0]
            n_pred = pred.shape[0]
            n = min(max(n_pred, n_lab), k)  # min(min(p, l), k)?

            # similar to mean_avg_prcsn, we need an arange, but this time +2
            # since python is zero-indexed, and the denom typically needs +1.
            # Also need the log base2...
            arange = np.arange(n, dtype=np.float32)  # length n

            # since we are only interested in the arange up to n_pred, truncate
            # if necessary
            arange = arange[:n_pred]
            denom = np.log2(arange + 2.)  # length n
            gains = 1. / denom  # length n

            # compute the gains where the prediction is present in the labels
            dcg_mask = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            dcg = gains[dcg_mask].sum()

            # the max DCG is sum of gains where the index < the label set size
            max_dcg = gains[arange < n_lab].sum()
            return dcg / max_dcg

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, _inner_ndcg)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

#mapk(original_recommendations_list.actual.values.tolist(), original_recommendations_list.bhattacharyya.values.tolist())


def _ark(actual, predicted, k=10):
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : int
        The average recall at k.
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / len(actual)


def mark(actual, predicted, k=10):
    """
    Computes the mean average recall at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: int
            The mean average recall at k (mar@k)
    """
    return np.mean([_ark(a,p,k) for a,p in zip(actual, predicted)])


def mape(y_test, pred):
    #https://stackoverflow.com/questions/72382501/how-to-interpret-mape-in-python-sklearn
    #https://datagy.io/mape-python/
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape


def calculate_mape(y_true, y_pred):
    #y_true, y_pred = check_array(y_true, y_pred)
    return mean_absolute_percentage_error(y_true, y_pred)
    #return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    if False:
        for model_name in test.iloc[:,3:].columns:
            score = calculate_mape(np.asarray(test['actual']).reshape(-1, 1), np.asarray(test[model_name]).reshape(-1, 1))
            print('{} -- MAPE = {}'.format(model_name, score))


def plot_MAE_2(bench):
    df = bench

    # define index column
    # df.set_index('k', inplace=True)

    # group data by product and display sales as line chart
    df.groupby('model')['mae'].plot(legend=True, marker='.', markersize=10, linestyle='-')
    plt.legend(loc='best')
    plt.xlabel("K")
    plt.ylabel("MAE")
    # Use this to show the plot in a new window
    plt.show()


def plot_RMSE_2(bench):
    df = bench

    # define index column
    # df.set_index('k', inplace=True)

    # group data by product and display sales as line chart
    df.groupby('model')['rmse'].plot(legend=True, marker='.', markersize=10, linestyle='-')
    plt.legend(loc='upper right')
    plt.xlabel("K")
    plt.ylabel("RMSE")
    # Use this to show the plot in a new window
    plt.show()


def plot_MAPE_2(bench):
    df = bench

    # define index column
    # df.set_index('k', inplace=True)

    # group data by product and display sales as line chart
    df.groupby('model')['mape'].plot(legend=True, marker='.', markersize=10, linestyle='-')
    plt.legend(loc='upper right')
    plt.xlabel("K")
    plt.ylabel("MAPE")
    # Use this to show the plot in a new window
    plt.show()


def plot_precision_at_results_2(num_recommendations, step, original_recommendations_list):

    def precision_at_plot(mapk_scores, model_names, k_range):

        from matplotlib.lines import Line2D
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        # from sklearn.utils.fixes import signature
        from funcsigs import signature
        """
        Plots the average precision at k for a set of models to compare.
        ----------
        mapk_scores: list of lists
            list of list of ap@k scores over k. This lis is in same order as model_names
            example: [[0.17, 0.25, 0.76],[0.2, 0.5, 0.74]]
        model_names: list
            list of model names in same order as coverage_scores
            example: ['Model A', 'Model B']
        k_range: list
            list or array indeitifying all k values in order
            example: [1,2,3,4,5,6,7,8,9,10]
        Returns:
        -------
            A ap@k plot
        """
        # create palette
        recommender_palette = ["#FF0000", "#008000", "#000080", "#800000", "#FFD700", "#00FF00", "#800080"]
        sns.set_palette(recommender_palette)

        # ['*', '.', '#', '+', 'x', 'o', '^']

        # lineplot
        mapk_df = pd.DataFrame(np.column_stack(mapk_scores), k_range, columns=model_names)
        ax = sns.lineplot(data=mapk_df, dashes=False, markers=True)
        plt.xticks(k_range)
        plt.setp(ax.lines, linewidth=5)

        # set labels
        ax.set_title('Average Precision of Queries at K Comparison')
        ax.set_ylabel('Precision')

        ax.set_xlabel('K')
        plt.show()


    step = step

    actual = original_recommendations_list['actual'].values.tolist()
    popularity_predictions = original_recommendations_list['popularity'].values.tolist()
    random_predictions = original_recommendations_list['random'].values.tolist()

    SBCF_predictions = original_recommendations_list['SBCF' + '-' + str(num_recommendations)].values.tolist()
    SBCF_pen_predictions = original_recommendations_list['SBCF_pen' + '-' + str(num_recommendations)].values.tolist()
    jaccard_predictions = original_recommendations_list['jaccard' + '-' + str(num_recommendations)].values.tolist()
    PCC_predictions = original_recommendations_list['PCC' + '-' + str(num_recommendations)].values.tolist()
    cosine_predictions = original_recommendations_list['cosine' + '-' + str(num_recommendations)].values.tolist()

    popularity_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        popularity_predictions_mark.extend([precision_at(labels=actual, predictions=popularity_predictions, k=K)])
    print('popularity_predictions_mark: {}'.format(popularity_predictions_mark))

    random_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        random_predictions_mark.extend([precision_at(labels=actual, predictions=random_predictions, k=K)])
    print('random_predictions_mark: {}'.format(random_predictions_mark))

    SBCF_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        SBCF_predictions_mark.extend([precision_at(labels=actual, predictions=SBCF_predictions, k=K)])
    print('SBCF_predictions_mark: {}'.format(SBCF_predictions_mark))

    SBCF_pen_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        SBCF_pen_predictions_mark.extend([precision_at(labels=actual, predictions=SBCF_pen_predictions, k=K)])
    print('SBCF_pen_predictions_mark: {}'.format(SBCF_pen_predictions_mark))

    PCC_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        PCC_predictions_mark.extend([precision_at(labels=actual, predictions=PCC_predictions, k=K)])
    print('PCC_predictions_mark: {}'.format(PCC_predictions_mark))

    jaccard_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        jaccard_predictions_mark.extend([precision_at(labels=actual, predictions=jaccard_predictions, k=K)])
    print('jaccard_predictions_mark: {}'.format(jaccard_predictions_mark))

    cosine_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        cosine_predictions_mark.extend([precision_at(labels=actual, predictions=cosine_predictions, k=K)])
    print('cosine_predictions_mark: {}'.format(cosine_predictions_mark))

    mark_scores = [random_predictions_mark,
                   popularity_predictions_mark,
                   # SBCF_predictions_mark,
                   SBCF_pen_predictions_mark,
                   # PCC_predictions_mark,
                   # jaccard_predictions_mark,
                   # cosine_predictions_mark
                   ]

    names = ['Random Recommender',
             'Popularity Recommender',
             # 'SBCF',
             'SBCF_pen',
             # 'PCC',
             # 'jaccard',
             # 'cosine'
             ]

    index = np.arange(0, num_recommendations, step)
    fig = plt.figure(figsize=(15, 7))
    # fig = plt.figure()
    precision_at_plot(mark_scores, model_names=names, k_range=index)

    print('Done with the metrics')


def plot_mapk_results_2(num_recommendations, step, original_recommendations_list):

    def mapk_plot(mapk_scores, model_names, k_range):

        from matplotlib.lines import Line2D
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        # from sklearn.utils.fixes import signature
        from funcsigs import signature
        """
        Plots the mean average precision at k for a set of models to compare.
        ----------
        mapk_scores: list of lists
            list of list of map@k scores over k. This lis is in same order as model_names
            example: [[0.17, 0.25, 0.76],[0.2, 0.5, 0.74]]
        model_names: list
            list of model names in same order as coverage_scores
            example: ['Model A', 'Model B']
        k_range: list
            list or array indeitifying all k values in order
            example: [1,2,3,4,5,6,7,8,9,10]
        Returns:
        -------
            A map@k plot
        """
        # create palette
        recommender_palette = ["#FF0000", "#008000", "#000080", "#800000", "#FFD700", "#00FF00", "#800080"]
        sns.set_palette(recommender_palette)

        # lineplot
        mapk_df = pd.DataFrame(np.column_stack(mapk_scores), k_range, columns=model_names)
        ax = sns.lineplot(data=mapk_df, dashes=False, markers=True)
        plt.xticks(k_range)
        plt.setp(ax.lines, linewidth=5)

        # set labels
        ax.set_title('Mean Average Precision at K (MAP@K) Comparison')
        ax.set_ylabel('MAP@K')
        ax.set_xlabel('K')
        plt.show()

    step = step

    actual = original_recommendations_list['actual'].values.tolist()
    popularity_predictions = original_recommendations_list['popularity'].values.tolist()
    random_predictions = original_recommendations_list['random'].values.tolist()

    SBCF_predictions = original_recommendations_list['SBCF' + '-' + str(num_recommendations)].values.tolist()
    SBCF_pen_predictions = original_recommendations_list['SBCF_pen' + '-' + str(num_recommendations)].values.tolist()
    jaccard_predictions = original_recommendations_list['jaccard' + '-' + str(num_recommendations)].values.tolist()
    PCC_predictions = original_recommendations_list['PCC' + '-' + str(num_recommendations)].values.tolist()
    cosine_predictions = original_recommendations_list['cosine' + '-' + str(num_recommendations)].values.tolist()

    popularity_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        popularity_predictions_mark.extend([mapk(actual=actual, predicted=popularity_predictions, k=K)])
    print('popularity_predictions_mark: {}'.format(popularity_predictions_mark))

    random_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        random_predictions_mark.extend([mapk(actual=actual, predicted=random_predictions, k=K)])
    print('random_predictions_mark: {}'.format(random_predictions_mark))

    SBCF_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        SBCF_predictions_mark.extend([mapk(actual=actual, predicted=SBCF_predictions, k=K)])
    print('SBCF_predictions_mark: {}'.format(SBCF_predictions_mark))

    SBCF_pen_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        SBCF_pen_predictions_mark.extend([mapk(actual=actual, predicted=SBCF_pen_predictions, k=K)])
    print('SBCF_pen_predictions_mark: {}'.format(SBCF_pen_predictions_mark))

    PCC_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        PCC_predictions_mark.extend([mapk(actual=actual, predicted=PCC_predictions, k=K)])
    print('PCC_predictions_mark: {}'.format(PCC_predictions_mark))

    jaccard_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        jaccard_predictions_mark.extend([mapk(actual=actual, predicted=jaccard_predictions, k=K)])
    print('jaccard_predictions_mark: {}'.format(jaccard_predictions_mark))

    cosine_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        cosine_predictions_mark.extend([mapk(actual=actual, predicted=cosine_predictions, k=K)])
    print('cosine_predictions_mark: {}'.format(cosine_predictions_mark))

    scores = [random_predictions_mark,
              popularity_predictions_mark,
              # SBCF_predictions_mark,
              SBCF_pen_predictions_mark,
              # PCC_predictions_mark,
              # jaccard_predictions_mark,
              # cosine_predictions_mark
              ]

    names = ['Random Recommender',
             'Popularity Recommender',
             # 'SBCF',
             'SBCF_pen',
             # 'PCC',
             # 'jaccard',
             # 'cosine'
             ]

    index = np.arange(0, num_recommendations, step)
    fig = plt.figure(figsize=(15, 7))
    # fig = plt.figure()
    mapk_plot(scores, model_names=names, k_range=index)

    print('Done with the metrics')


def plot_mark_results_2(num_recommendations, step, original_recommendations_list):

    def mark_plot(mapk_scores, model_names, k_range):

        from matplotlib.lines import Line2D
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        # from sklearn.utils.fixes import signature
        from funcsigs import signature
        """
        Plots the mean average recall at k for a set of models to compare.
        ----------
        mark_scores: list of lists
            list of list of mar@k scores over k. This lis is in same order as model_names
            example: [[0.17, 0.25, 0.76],[0.2, 0.5, 0.74]]
        model_names: list
            list of model names in same order as coverage_scores
            example: ['Model A', 'Model B']
        k_range: list
            list or array indeitifying all k values in order
            example: [1,2,3,4,5,6,7,8,9,10]
        Returns:
        -------
            A mar@k plot
        """
        # create palette
        recommender_palette = ["#FF0000", "#008000", "#000080", "#800000", "#FFD700", "#00FF00", "#800080"]
        sns.set_palette(recommender_palette)

        # lineplot
        mapk_df = pd.DataFrame(np.column_stack(mapk_scores), k_range, columns=model_names)
        ax = sns.lineplot(data=mapk_df, dashes=False, markers=True)
        plt.xticks(k_range)
        plt.setp(ax.lines, linewidth=5)

        # set labels
        ax.set_title('Mean Average Recall at K (MAR@K) Comparison')
        ax.set_ylabel('MAR@K')
        ax.set_xlabel('K')
        plt.show()

    step = step

    actual = original_recommendations_list['actual'].values.tolist()
    popularity_predictions = original_recommendations_list['popularity'].values.tolist()
    random_predictions = original_recommendations_list['random'].values.tolist()

    SBCF_predictions = original_recommendations_list['SBCF' + '-' + str(num_recommendations)].values.tolist()
    SBCF_pen_predictions = original_recommendations_list['SBCF_pen' + '-' + str(num_recommendations)].values.tolist()
    jaccard_predictions = original_recommendations_list['jaccard' + '-' + str(num_recommendations)].values.tolist()
    PCC_predictions = original_recommendations_list['PCC' + '-' + str(num_recommendations)].values.tolist()
    cosine_predictions = original_recommendations_list['cosine' + '-' + str(num_recommendations)].values.tolist()

    popularity_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        popularity_predictions_mark.extend([mark(actual=actual, predicted=popularity_predictions, k=K)])
    print('popularity_predictions_mark: {}'.format(popularity_predictions_mark))

    random_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        random_predictions_mark.extend([mark(actual=actual, predicted=random_predictions, k=K)])
    print('random_predictions_mark: {}'.format(random_predictions_mark))

    SBCF_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        SBCF_predictions_mark.extend([mark(actual=actual, predicted=SBCF_predictions, k=K)])
    print('SBCF_predictions_mark: {}'.format(SBCF_predictions_mark))

    SBCF_pen_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        SBCF_pen_predictions_mark.extend([mark(actual=actual, predicted=SBCF_pen_predictions, k=K)])
    print('SBCF_pen_predictions_mark: {}'.format(SBCF_pen_predictions_mark))

    PCC_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        PCC_predictions_mark.extend([mark(actual=actual, predicted=PCC_predictions, k=K)])
    print('PCC_predictions_mark: {}'.format(PCC_predictions_mark))

    jaccard_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        jaccard_predictions_mark.extend([mark(actual=actual, predicted=jaccard_predictions, k=K)])
    print('jaccard_predictions_mark: {}'.format(jaccard_predictions_mark))

    cosine_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        cosine_predictions_mark.extend([mark(actual=actual, predicted=cosine_predictions, k=K)])
    print('cosine_predictions_mark: {}'.format(cosine_predictions_mark))

    scores = [random_predictions_mark,
              popularity_predictions_mark,
              # SBCF_predictions_mark,
              SBCF_pen_predictions_mark,
              # PCC_predictions_mark,
              # jaccard_predictions_mark,
              # cosine_predictions_mark
              ]

    names = ['Random Recommender',
             'Popularity Recommender',
             # 'SBCF',
             'SBCF_pen',
             # 'PCC',
             # 'jaccard',
             # 'cosine'
             ]

    index = np.arange(0, num_recommendations, step)
    fig = plt.figure(figsize=(15, 7))
    # fig = plt.figure()
    mark_plot(scores, model_names=names, k_range=index)

    print('Done with the metrics')


def plot_ndcg_at_results_2(num_recommendations, step, original_recommendations_list):

    def ndcg_at_plot(mapk_scores, model_names, k_range):

        from matplotlib.lines import Line2D
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        # from sklearn.utils.fixes import signature
        from funcsigs import signature
        """
        Plots the mean average recall at k for a set of models to compare.
        ----------
        mark_scores: list of lists
            list of list of mar@k scores over k. This lis is in same order as model_names
            example: [[0.17, 0.25, 0.76],[0.2, 0.5, 0.74]]
        model_names: list
            list of model names in same order as coverage_scores
            example: ['Model A', 'Model B']
        k_range: list
            list or array indeitifying all k values in order
            example: [1,2,3,4,5,6,7,8,9,10]
        Returns:
        -------
            A mar@k plot
        """
        # create palette
        recommender_palette = ["#FF0000", "#008000", "#000080", "#800000", "#FFD700", "#00FF00", "#800080"]
        sns.set_palette(recommender_palette)

        # lineplot
        mapk_df = pd.DataFrame(np.column_stack(mapk_scores), k_range, columns=model_names)
        ax = sns.lineplot(data=mapk_df, dashes=False, markers=True)
        plt.xticks(k_range)
        plt.setp(ax.lines, linewidth=5)

        # set labels
        ax.set_title('nDCG at K (nDCG@K) Comparison')
        ax.set_ylabel('nDCG@K')
        ax.set_xlabel('K')
        plt.show()

    step = step

    actual = original_recommendations_list['actual'].values.tolist()
    popularity_predictions = original_recommendations_list['popularity'].values.tolist()
    random_predictions = original_recommendations_list['random'].values.tolist()

    SBCF_predictions = original_recommendations_list['SBCF' + '-' + str(num_recommendations)].values.tolist()
    SBCF_pen_predictions = original_recommendations_list['SBCF_pen' + '-' + str(num_recommendations)].values.tolist()
    jaccard_predictions = original_recommendations_list['jaccard' + '-' + str(num_recommendations)].values.tolist()
    PCC_predictions = original_recommendations_list['PCC' + '-' + str(num_recommendations)].values.tolist()
    cosine_predictions = original_recommendations_list['cosine' + '-' + str(num_recommendations)].values.tolist()

    popularity_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        popularity_predictions_mark.extend([ndcg_at(labels=actual[:K], predictions=popularity_predictions[:K], k=K, assume_unique=True)])
    print('popularity_predictions_mark: {}'.format(popularity_predictions_mark))

    random_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        random_predictions_mark.extend([ndcg_at(labels=actual[:K], predictions=random_predictions[:K], k=K, assume_unique=True)])
    print('random_predictions_mark: {}'.format(random_predictions_mark))

    SBCF_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        SBCF_predictions_mark.extend([ndcg_at(labels=actual[:K], predictions=SBCF_predictions[:K], k=K, assume_unique=True)])
    print('SBCF_predictions_mark: {}'.format(SBCF_predictions_mark))

    SBCF_pen_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        SBCF_pen_predictions_mark.extend([ndcg_at(labels=actual[:K], predictions=SBCF_pen_predictions[:K], k=K, assume_unique=True)])
    print('SBCF_pen_predictions_mark: {}'.format(SBCF_pen_predictions_mark))

    PCC_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        PCC_predictions_mark.extend([ndcg_at(labels=actual[:K], predictions=PCC_predictions[:K], k=K, assume_unique=True)])
    print('PCC_predictions_mark: {}'.format(PCC_predictions_mark))

    jaccard_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        jaccard_predictions_mark.extend([ndcg_at(labels=actual[:K], predictions=jaccard_predictions[:K], k=K, assume_unique=True)])
    print('jaccard_predictions_mark: {}'.format(jaccard_predictions_mark))

    cosine_predictions_mark = []
    for K in np.arange(1, num_recommendations, step):
        cosine_predictions_mark.extend([ndcg_at(labels=actual[:K], predictions=cosine_predictions[:K], k=K, assume_unique=True)])
    print('cosine_predictions_mark: {}'.format(cosine_predictions_mark))

    scores = [random_predictions_mark,
              popularity_predictions_mark,
              # SBCF_predictions_mark,
              SBCF_pen_predictions_mark,
              # PCC_predictions_mark,
              # jaccard_predictions_mark,
              # cosine_predictions_mark
              ]

    names = ['Random Recommender',
             'Popularity Recommender',
             # 'SBCF',
             'SBCF_pen',
             # 'PCC',
             # 'jaccard',
             # 'cosine'
             ]

    index = np.arange(0, num_recommendations, step)
    fig = plt.figure(figsize=(15, 7))
    # fig = plt.figure()
    ndcg_at_plot(scores, model_names=names, k_range=index)

    print('Done with the metrics')



def get_coverage_at_k(original_recommendations_list, step=1):

    bhattacharyya_predictions_coverage = []

    for k in np.arange(1, 30, step):
        actual = original_recommendations_list.actual.values.tolist()
        # actual_flattened = [p for sublist in actual for p in sublist]

        bhattacharyya = original_recommendations_list.bhattacharyya.values.tolist()

        coverage = coverage(predicted=bhattacharyya, catalog=actual)
        bhattacharyya_predictions_coverage.extend([coverage])

        print('Coverrage at K: {} = {}'.format(k, coverage))


def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered.

    parameters:

    training_set - The training set resulting from make_train, where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model

    predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero and item vectors as item one.

    altered_users - The indices of the users where at least one user/item pair was altered from make_train function

    test_set - The test set constucted earlier from make_train function



    returns:

    The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions
    there were originally zero to test ranking ability in addition to the most popular items as a benchmark.
    '''

    store_auc = []  # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = []  # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis=0)).reshape(-1)  # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users:  # Iterate through each user that had an item altered
        training_row = training_set[user, :].toarray().reshape(-1)  # Get the training set row
        zero_inds = np.where(training_row == 0)  # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user, :]
        pred = user_vec.dot(item_vecs).toarray()[0, zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user, :].toarray()[0, zero_inds].reshape(-1)
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training
        pop = pop_items[zero_inds]  # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual))  # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual))  # Calculate AUC using most popular and score
    # End users iteration

    return float('%.3f' % np.mean(store_auc)), float('%.3f' % np.mean(popularity_auc))
    # Return the mean AUC rounded to three decimal places for both test and popularity benchmark


def hit_rate(topNPredicted, leftOutPredictions):
    hits = 0
    total = 0

# For each left-out rating
    for leftOut in leftOutPredictions:
        userID = leftOut[0]
        leftOutMovieID = leftOut[1]
        # Is it in the predicted top 10 for this user?
        hit = False
        for movieID, predictedRating in topNPredicted[int(userID)]:
            if (int(leftOutMovieID) == int(movieID)):
                hit = True
                break
        if (hit) :
            hits += 1

        total += 1

    # Compute overall precision
    return hits/total

    #print("\nHit Rate: ", HitRate(topNPredicted, leftOutPredictions))