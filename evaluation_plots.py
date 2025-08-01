# -*- coding: utf-8 -*-
#
# Author: Thiago Andrade
#
# Recommender system ranking metrics

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import recmetrics

from recommender_system import evaluation_metrics


def plot_mark(num_recommendations, step, original_recommendations_list):
    def mark_plot(mark_scores, model_names, k_range):
        """
        Plots the mean average recall at k for a set of models to compare.
        ----------
        mark_scores: list of lists
            list of list of mar@k scores over k. This list is in same order as model_names
            example: [[0.17, 0.25, 0.76],[0.2, 0.5, 0.74]]
        model_names: list
            list of model names in same order as coverage_scores
            example: ['Model A', 'Model B']
        k_range: list
            list or array identifying all k values in order
            example: [1,2,3,4,5,6,7,8,9,10]
        Returns:
        -------
            A mar@k plot
        """
        # create palette
        #recommender_palette = ["#ED2BFF", "#14E2C0", "#FF9F1C", "#5E2BFF", "#FC5FA3"]
        #sns.set_palette(recommender_palette)
        sns.color_palette(palette=None, n_colors=len(model_names), desat=.5)

        # lineplot
        df = pd.DataFrame(np.column_stack(mark_scores), k_range, columns=model_names)
        ax = sns.lineplot(data=df)
        plt.xticks(k_range)
        plt.setp(ax.lines, linewidth=5)

        # set labels
        ax.set_title('Mean Average Recall at K (MAR@K) Comparison')
        ax.set_ylabel('MAR@K')
        ax.set_xlabel('K')

        plt.show()

    dict_model_names = {}
    scores = []
    names = []

    for model_name in original_recommendations_list.columns[1:]:
        dict_model_names[model_name] = original_recommendations_list[model_name].values.tolist()
        names.append(model_name)
        model_scores = []
        for k in np.arange(1, (num_recommendations + step) - 1, step):
            model_scores.extend([evaluation_metrics.mark(
                original_recommendations_list.actual.values.tolist(),
                original_recommendations_list[model_name].values.tolist(), k=k)])
        scores.append(model_scores)

    index = np.arange(1, (num_recommendations + step) - 1, step)
    fig = plt.figure(figsize=(15, 7))
    # fig = plt.figure()
    mark_plot(mark_scores=scores, model_names=names, k_range=index)

    print('Done with the metrics for - plot_mark_results')
    #test.to_pickle('D:\\MAPi\Datasets\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\df_test_mark_results0104.df')


def plot_mapk(num_recommendations, step, original_recommendations_list):
    def mapk_plot(mapk_scores, model_names, k_range):
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
            list or array identifying all k values in order
            example: [1,2,3,4,5,6,7,8,9,10]
        Returns:
        -------
            A map@k plot
        """
        # create palette
        #recommender_palette = ["#ED2BFF", "#14E2C0", "#FF9F1C", "#5E2BFF", "#FC5FA3"]
        #sns.set_palette(recommender_palette)
        sns.color_palette(palette=None, n_colors=len(model_names), desat=.5)

        # lineplot
        df = pd.DataFrame(np.column_stack(mapk_scores), k_range, columns=model_names)
        ax = sns.lineplot(data=df)
        plt.xticks(k_range)
        plt.setp(ax.lines, linewidth=5)

        # set labels
        ax.set_title('Mean Average Precision at K (MAP@K) Comparison')
        ax.set_ylabel('MAP@K')
        ax.set_xlabel('K')
        plt.show()

    dict_model_names = {}
    scores = []
    names = []

    for model_name in original_recommendations_list.columns[1:]:
        dict_model_names[model_name] = original_recommendations_list[model_name].values.tolist()
        names.append(model_name)
        model_scores = []
        for k in np.arange(1, (num_recommendations + step) - 1, step):
            model_scores.extend([evaluation_metrics.mapk(
                original_recommendations_list.actual.values.tolist(),
                original_recommendations_list[model_name].values.tolist(), k=k)])
        scores.append(model_scores)

    index = np.arange(1, (num_recommendations + step) - 1, step)
    fig = plt.figure(figsize=(15, 7))
    # fig = plt.figure()
    mapk_plot(mapk_scores=scores, model_names=names, k_range=index)

    print('Done with the metrics for - plot_mapk_results')
    #test.to_pickle('D:\\MAPi\Datasets\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\df_test_mapk_results0104.df')


def plot_class_separation(test, model_name='bhattacharyya'):
    """pred_df: pandas dataframe
        a dataframe containing a column of predicted interaction values or classification probabilites,
        and a column of true class 1 and class 0 states.
        This dataframe must contain columns named "predicted" and "truth"
        example:
            predicted | truth
            5.345345	|  5
            2.072020	|  2
    """

    class_separation_plot_df = test[['actual', model_name]].copy()
    class_separation_plot_df = class_separation_plot_df.rename(columns={'actual': 'truth', model_name: 'predicted'})

    recmetrics.class_separation_plot(class_separation_plot_df)


def plot_precision_at(num_recommendations, step, original_recommendations_list):
    def precision_at_plot(precision_at_scores, model_names, k_range):

        #import numpy as np
        #import pandas as pd
        #import seaborn as sns
        #import matplotlib.pyplot as plt
        #from matplotlib.lines import Line2D
        #from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        #from sklearn.utils.fixes import signature
        #from funcsigs import signature
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
        #recommender_palette = ["#ED2BFF", "#14E2C0", "#FF9F1C", "#5E2BFF", "#FC5FA3"]
        #sns.set_palette(recommender_palette)
        sns.color_palette(palette=None, n_colors=len(model_names), desat=.5)

        # lineplot
        df = pd.DataFrame(np.column_stack(precision_at_scores), k_range, columns=model_names)
        ax = sns.lineplot(data=df)
        plt.xticks(k_range)
        plt.setp(ax.lines, linewidth=5)

        # set labels
        ax.set_title('Average Precision of Queries at K Comparison')
        ax.set_ylabel('Precision')

        ax.set_xlabel('K')
        plt.show()

    dict_model_names = {}
    scores = []
    names = []

    for model_name in original_recommendations_list.columns[1:]:
        dict_model_names[model_name] = original_recommendations_list[model_name].values.tolist()
        names.append(model_name)
        model_scores = []
        for k in np.arange(1, (num_recommendations + step) - 1, step):
            model_scores.extend([evaluation_metrics.precision_at(
                original_recommendations_list.actual.values.tolist(),
                original_recommendations_list[model_name].values.tolist(), k=k)])
        scores.append(model_scores)

    index = np.arange(1, (num_recommendations + step) - 1, step)
    fig = plt.figure(figsize=(15, 7))
    # fig = plt.figure()
    precision_at_plot(precision_at_scores=scores, model_names=names, k_range=index)

    print('Done with the metrics for - plot_precision_at_results')
    #test.to_pickle('D:\\MAPi\Datasets\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\df_test_precision_at_results0104.df')


def plot_mape(bench):
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

def plot_mae(bench):
    df = bench

    #define index column
    #df.set_index('k', inplace=True)

    #group data by product and display sales as line chart
    df.groupby('model')['mae'].plot(legend=True,  marker='.', markersize=10, linestyle='-')
    plt.legend(loc='upper right')
    plt.xlabel("K")
    plt.ylabel("MAE")
    # Use this to show the plot in a new window
    plt.show()


def plot_mae_results(test, num_recommendations, step):
    def mae_plot(mae_scores, model_names, k_range):
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        from sklearn.metrics import precision_score, mean_absolute_error, mean_squared_error
        # from sklearn.utils.fixes import signature
        from funcsigs import signature
        """
        Plots the mean average error for a set of models to compare.
        ----------
        mae_scores: list of lists
            list of list of mae scores over k. This lis is in same order as model_names
            example: [[0.17, 0.25, 0.76],[0.2, 0.5, 0.74]]
        model_names: list
            list of model names in same order as coverage_scores
            example: ['Model A', 'Model B']
        k_range: list
            list or array indeitifying all k values in order
            example: [1,2,3,4,5,6,7,8,9,10]
        Returns:
        -------
            A MAE plot
        """
        # create palette
        recommender_palette = ["#ED2BFF", "#14E2C0", "#FF9F1C", "#5E2BFF", "#FC5FA3"]
        sns.set_palette(recommender_palette)

        # lineplot
        df = pd.DataFrame(np.column_stack(mae_scores), k_range, columns=model_names)
        ax = sns.lineplot(data=df)
        plt.xticks(k_range)
        plt.setp(ax.lines, linewidth=5)

        # set labels
        ax.set_title('Mean Average Error Comparison')
        ax.set_ylabel('MAE')

        ax.set_xlabel('K')
        plt.show()

    scores = []
    names = []

    for model_name in test.iloc[:, 3:].columns:
        # mae = mean_absolute_error(y_true=test['actual'], y_pred=test[model_name])
        # mae = recmetrics.mse(test['actual'], test[model_name])
        mae = np.mean(np.abs(test['actual'] - test[model_name]), axis=0)
        scores.append(mae)
        names.append(model_name)

    index = np.arange(1, 2, step)
    fig = plt.figure(figsize=(15, 7))
    # fig = plt.figure()
    mae_plot(scores, model_names=names, k_range=index)

    print('Done with the metrics')

    # test.to_pickle(DATASETS_PATH + 'df_test_mae_results0104.df')


def plot_rmse(bench):
    df = bench

    #define index column
    #df.set_index('k', inplace=True)

    #group data by product and display sales as line chart
    df.groupby('model')['rmse'].plot(legend=True,  marker='.', markersize=10, linestyle='-')
    plt.legend(loc='upper right')
    plt.xlabel("K")
    plt.ylabel("RMSE")
    # Use this to show the plot in a new window
    plt.show()

def plot_rmse_2(num_recommendations, step, original_recommendations_list):
    def rmse_plot(rmse_scores, model_names, k_range):
        #import numpy as np
        #import pandas as pd
        #import seaborn as sns
        #import matplotlib.pyplot as plt
        #from matplotlib.lines import Line2D
        #from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        #from sklearn.metrics import precision_score, mean_absolute_error, mean_squared_error
        #from sklearn.utils.fixes import signature
        #from funcsigs import signature
        """
        Plots the root mean square error for a set of models to compare.
        ----------
        rmse_scores: list of lists
            list of list of RMSE scores over k. This lis is in same order as model_names
            example: [[0.17, 0.25, 0.76],[0.2, 0.5, 0.74]]
        model_names: list
            list of model names in same order as coverage_scores
            example: ['Model A', 'Model B']
        k_range: list
            list or array indeitifying all k values in order
            example: [1,2,3,4,5,6,7,8,9,10]
        Returns:
        -------
            A RMSE plot
        """
        # create palette
        #recommender_palette = ["#ED2BFF", "#14E2C0", "#FF9F1C", "#5E2BFF", "#FC5FA3"]
        #sns.set_palette(recommender_palette)
        sns.color_palette(palette=None, n_colors=len(model_names), desat=.5)

        # lineplot
        df = pd.DataFrame(np.column_stack(rmse_scores), k_range, columns=model_names)
        ax = sns.lineplot(data=df)
        plt.xticks(k_range)
        plt.setp(ax.lines, linewidth=5)

        # set labels
        ax.set_title('Root Mean Square Error Comparison')
        ax.set_ylabel('RMSE')

        ax.set_xlabel('K')
        plt.show()

    scores = []
    names = []
    y_true = original_recommendations_list['actual']

    for model_name in original_recommendations_list.columns[1:]:
        rmse = sqrt(mean_squared_error(y_true=y_true,
                                       y_pred=original_recommendations_list[model_name]))
        scores.append(rmse)
        names.append(model_name)

    index = np.arange(1, num_recommendations, step)
    fig = plt.figure(figsize=(15, 7))
    # fig = plt.figure()
    rmse_plot(rmse_scores=scores, model_names=names, k_range=index)

    print('Done with the metrics for - plot_rmse_results')
    #test.to_pickle('D:\\MAPi\Datasets\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\df_test_rmse_results0104.df')


def plot_ndcg_at(num_recommendations, step, original_recommendations_list):

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
        df = pd.DataFrame(np.column_stack(mapk_scores), k_range, columns=model_names)
        ax = sns.lineplot(data=df, dashes=False, markers=True)
        plt.xticks(k_range)
        plt.setp(ax.lines, linewidth=5)

        # set labels
        ax.set_title('nDCG at K (nDCG@K) Comparison')
        ax.set_ylabel('nDCG@K')
        ax.set_xlabel('K')
        plt.show()

    dict_model_names = {}
    scores = []
    names = []

    for model_name in original_recommendations_list.columns[1:]:
        dict_model_names[model_name] = original_recommendations_list[model_name].values.tolist()
        names.append(model_name)
        model_scores = []
        for k in np.arange(1, (num_recommendations + step) - 1, step):
            model_scores.extend([evaluation_metrics.ndcg_at(
                labels=original_recommendations_list.actual.values.tolist()[:k],
                predictions=original_recommendations_list[model_name].values.tolist()[:k],
                k=k, assume_unique=True)])
        scores.append(model_scores)

    if False:
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

    index = np.arange(1, (num_recommendations + step) - 1, step)
    fig = plt.figure(figsize=(15, 7))
    # fig = plt.figure()
    ndcg_at_plot(scores, model_names=names, k_range=index)

    print('Done with the metrics')
