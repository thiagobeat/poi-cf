
import numpy as np
import pandas as pd
import jenkspy
import matplotlib.pyplot as plt

# Jenks Breaks Handlers
# https://stackoverflow.com/questions/28416408/scikit-learn-how-to-run-kmeans-on-a-one-dimensional-array


def get_my_jenks_breaks(data_list, number_class):
    data_list.sort()
    mat1 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, number_class + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(data_list) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(data_list) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data_list[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, number_class + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(data_list)
    kclass = []
    for i in range(number_class + 1):
        kclass.append(min(data_list))
    kclass[number_class] = float(data_list[len(data_list) - 1])
    count_num = number_class
    while count_num >= 2:  # print "rank = " + str(mat1[k][count_num])
        idx = int((mat1[k][count_num]) - 2)
        # print "val = " + str(data_list[idx])
        kclass[count_num - 1] = data_list[idx]
        k = int((mat1[k][count_num] - 1))
        count_num -= 1

    # return kclass

    #data_list['rating_my_cut_jenks'] = -1.

    data_list['rating_my_cut_jenks'] = pd.cut(data_list['rating'],
                                              bins=kclass,
                                              duplicates='drop',
                                              labels=kclass,
                                              include_lowest=True)
    #data_list['rating_my_cut_jenks'] = pd.to_numeric(data_list['rating_my_cut_jenks'])

    return data_list


def plot_jenks_breaks(data, nb_stars):
    """_summary_
        Call it as data = df_stay_points.loc[df_stay_points.user_id == user_id].rating.values
    Args:
        data (_type_): _description_
        nb_stars (_type_): _description_
    """

    # data = df_stay_points.loc[df_stay_points.user_id == user_id].rating.values

    data.sort()
    breaks = jenkspy.jenks_breaks(data, nb_class=nb_stars)
    # breaks = get_jenks_breaks(data, nb_class=nb_stars)
    # breaks.sort()

    for line in breaks:
        plt.plot([line for _ in range(len(data))], 'k--')

    plt.plot(data)
    plt.grid(True)
    plt.show()


def get_cut_jenks_transform(df_stay_points, nb_stars_ratings, by_rating=False):
    df_stay_points['cut_jenks'] = -1.

    #Can't calculate with nan values
    if not np.isnan(df_stay_points.rating.values).any():
        if by_rating:
            # if nb_class >= len(values) or nb_class < 2:
            if len(np.asarray(df_stay_points.rating.values).reshape(-1, 1)) > 2 and len(np.asarray(df_stay_points.rating.values).reshape(-1, 1)) > nb_stars_ratings:

                nb_stars = nb_stars_ratings
                # nb_stars = len(set(df_stay_points.loc[df_stay_points.user_id == user_id, 'rating_sqrt'].to_list()))-1

                if nb_stars >= 2:
                    if nb_stars > 4:
                        nb_stars = 4

                    nb_bins = jenkspy.jenks_breaks(df_stay_points.rating.values, nb_class=nb_stars)
                    nb_bins = list(set(nb_bins))
                    nb_bins.sort()

                    nb_labels = list(np.arange(0, len(list(set(nb_bins)))))
                    while len(nb_labels) >= len(nb_bins):
                        nb_labels = list(np.arange(0, len(nb_bins) - 1))

                    df_stay_points['cut_jenks'] = pd.cut(df_stay_points['rating'],
                                                                bins=nb_bins,
                                                                duplicates='drop',
                                                                labels=nb_labels,
                                                                include_lowest=True)
                    # df_stay_points['rating_cut_jenks'] = pd.to_numeric(df_stay_points['rating_cut_jenks'])
            return df_stay_points

        # if nb_class >= len(values) or nb_class < 2:
        if len(np.asarray(df_stay_points.rating_norm.values).reshape(-1, 1)) > 2 and len(np.asarray(df_stay_points.rating_norm.values).reshape(-1, 1)) > nb_stars_ratings:

            nb_stars = nb_stars_ratings
            # nb_stars = len(set(df_stay_points.loc[df_stay_points.user_id == user_id, 'rating_sqrt'].to_list()))-1

            if nb_stars >= 2:
                if nb_stars > 4:
                    nb_stars = 4

                nb_bins = jenkspy.jenks_breaks(df_stay_points.rating_norm.values, nb_class=nb_stars)
                nb_bins = list(set(nb_bins))
                nb_bins.sort()

                nb_labels = list(np.arange(0, len(list(set(nb_bins)))))
                while len(nb_labels) >= len(nb_bins):
                    nb_labels = list(np.arange(0, len(nb_bins) - 1))

                df_stay_points['rating_cut_jenks'] = pd.cut(df_stay_points['rating_norm'],
                                                            bins=nb_bins,
                                                            duplicates='drop',
                                                            labels=nb_labels,
                                                            include_lowest=True)
                #df_stay_points['rating_cut_jenks'] = pd.to_numeric(df_stay_points['rating_cut_jenks'])

        return df_stay_points

    else:
        #df_stay_points['rating'] = -1.
        return df_stay_points