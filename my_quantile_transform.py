
import numpy as np
import pandas as pd
import jenkspy
from sklearn.preprocessing import quantile_transform

# https://cmdlinetips.com/2020/06/computing-quantile-normalization-in-python/
# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py


def get_quantile_transform(df_stay_points):
    df_stay_points['rating_quantile'] = \
        quantile_transform(X=np.asarray(df_stay_points.rating).reshape(-1, 1),
                           n_quantiles=5, random_state=0, output_distribution="normal", copy=True)
    df_stay_points['rating_sqrt_quantile'] = \
        quantile_transform(X=np.asarray(df_stay_points.rating_sqrt).reshape(-1, 1),
                           n_quantiles=5, random_state=0, output_distribution="normal", copy=True)
    df_stay_points['rating_brais_quantile'] = \
        quantile_transform(X=np.asarray(df_stay_points.rating_brais).reshape(-1, 1),
                           n_quantiles=5, random_state=0, output_distribution="normal", copy=True)
    # df_stay_points['rating_brais_sig_quantile'] = \
    # quantile_transform(X=np.asarray(df_stay_points.rating_brais_sig).reshape(-1, 1),
    #                    n_quantiles=5, random_state=0, output_distribution="normal", copy=True)

    if False:
        df_stay_points['cut_jenks_sqrt_quantile'] = -1.

        nb_stars = len(set(df_stay_points.loc[df_stay_points.user_id == user_id, 'rating_sqrt_quantile'].to_list())) - 1
        if nb_stars >= 2:
            if nb_stars > 5:
                nb_stars = 5

            nb_bins = jenkspy.jenks_breaks(df_stay_points.loc[df_stay_points.user_id == user_id].rating_sqrt_quantile.values, nb_class=nb_stars)
            nb_bins = list(set(nb_bins))
            nb_bins.sort()

            nb_labels = list(np.arange(0, len(list(set(nb_bins)))))
            while len(nb_labels) >= len(nb_bins):
                nb_labels = list(np.arange(0, len(nb_bins) - 1))

            df_stay_points.loc[df_stay_points.user_id == user_id, 'cut_jenks_sqrt_quantile'] = pd.cut(df_stay_points.loc[df_stay_points.user_id == user_id, 'rating_sqrt_quantile'],
                                                                                                      bins=nb_bins,
                                                                                                      duplicates='drop',
                                                                                                      labels=nb_labels,
                                                                                                      include_lowest=True)
            df_stay_points.loc[df_stay_points.user_id == user_id, 'cut_jenks_sqrt_quantile'] = pd.to_numeric(df_stay_points.loc[df_stay_points.user_id == user_id, 'cut_jenks_sqrt_quantile'])

    return df_stay_points

