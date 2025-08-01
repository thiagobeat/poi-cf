from scipy import spatial

from sklearn.metrics import pairwise

import numpy as np
import math
from math import sqrt
from scipy.stats import gaussian_kde

import mplleaflet
import pandas as pd
import matplotlib.pyplot as plt

MAIN_FOLDER = 'C:/MAPi/Dataset Algar/Data'
POIS_OUTPUT_FOLDER = '/processed_data/pois/'

MAIN_FOLDER = 'C:\\MAPi\\Yu Zheng\\Datasets\\Geolife Trajectories 1.3\\battacharrya'

import jenkspy


def get_bhattacharyya_similarity(df_stay_points):

    headers_similarity_pois = ['userA', 'userB', 'places_A_B', 'similarity', 'percentage']
    df_similarity_pois_bhatthacharrya = pd.DataFrame(columns=headers_similarity_pois)
    list_similarity_pois = []

    listVisitedUsers = []
    print_map = False

    for userA in set(df_stay_points['user_id'].values):
        print(userA)
        listVisitedUsers.append(userA)

        # Gets all users but the those the external loop to compare by jaccard similarity
        for userB in set(df_stay_points.loc[(~df_stay_points['user_id'].isin(listVisitedUsers)), 'user_id'].values):
            print(userB)

            poisA = df_stay_points.iloc[userA, :].values.reshape(1, -1).ravel()
            poisB = df_stay_points.iloc[userB, :].values.reshape(1, -1).ravel()

            percentage = bhatta_dist(X1=poisA, X2=poisB, method='continuous')
            similarity = percentage #str(len(set(poisA).intersection(set(poisB)))) + '/' + str(len(set(poisA).union(set(poisB))))
            places_A_B = str(df_stay_points.iloc[userA, :].astype(bool).sum(axis=1)) + '/' + str(df_stay_points.iloc[userB, :].astype(bool).sum(axis=1))

            print('Similarity User A x User B = {}'.format(similarity))

            list_similarity_pois.append(pd.DataFrame([[userA, userB, places_A_B, similarity, percentage]],
                                                     columns=headers_similarity_pois))


    df_similarity_pois_bhatthacharrya = pd.concat(list_similarity_pois)
    df_similarity_pois_bhatthacharrya.index = pd.RangeIndex(len(df_similarity_pois_bhatthacharrya.index))
    df_similarity_pois_bhatthacharrya = df_similarity_pois_bhatthacharrya.sort_values(by=['percentage'], ascending=False)
    print(df_similarity_pois_bhatthacharrya)

    df_similarity_pois_file = MAIN_FOLDER + POIS_OUTPUT_FOLDER + '/similarity_pois_bhatthacharrya.csv'
    df_similarity_pois_bhatthacharrya.to_csv(df_similarity_pois_file, index=False)

    return df_similarity_pois_bhatthacharrya


def get_bhat_similarity_old(df_stay_points):

    # bhattacharyya test


    h1 = list(df_stay_points[((df_stay_points.user_id == 4))].cluster_id);
    h2 = list(df_stay_points[((df_stay_points.user_id == 5))].cluster_id);

    h = [h1, h2];

    def bhattacharyya_distance(dist1, dist2, lower, upper):
        distance = 0
        d = 0.1
        for dx in np.arange(lower, upper, d):
            distance += np.sqrt(dist1(dx) * dist2(dx))
        return -np.log(distance * d)

    def normalize(h):
        return h / np.sum(h)

    def mean(hist):
        mean = 0.0;
        for i in hist:
            mean += i;
        mean /= len(hist);
        return mean;

    def bhattacharyya(hist1, hist2):
        # calculate mean of hist1
        h1_ = mean(hist1);

        # calculate mean of hist2
        h2_ = mean(hist2);

        # calculate score
        score = 0;
        for i in range(len(h1)):
            score += math.sqrt(hist1[i] * hist2[i]);
        # print h1_,h2_,score;
        score = math.sqrt(1 - (1 / math.sqrt(h1_ * h2_ *  len(h1) *  len(h2))) * score);
        return score;

    # generate and output scores
    scores = [];
    for i in range(len(h)):
        score = [];
        for j in range(len(h)):
            score.append(bhattacharyya(h[i], h[j]));
        scores.append(score);

    for i in scores:
        print(i)


    if False:

        dataSetI = [1, 0, 2, 0, 1, 0, 2, 0, 3, 0]
        dataSetII = [0, 1, 0, 2, 0, 1, 0, 2, 0, 3]

        result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
        print ("cosine sim: ", result)

        result = 1 - spatial.distance.jaccard(dataSetI, dataSetII)
        print ("jaccard sim: ", result)

        result = jaccard_similarity(dataSetI, dataSetII)
        print ("jaccard_similarity: ", result)

        result = 1 - spatial.distance.euclidean(dataSetI, dataSetII)
        print ("euclidean sim: ", result)

        result = 1 - pairwise.euclidean_distances([dataSetI, dataSetII])
        print ("euclidean distances: ", result)

        result = 1 - pairwise.manhattan_distances([dataSetI, dataSetII])
        print ("manhattan distances: ", result)


"""
The function bhatta_dist() calculates the Bhattacharyya distance between two classes on a single feature.
    The distance is positively correlated to the class separation of this feature. Four different methods are
    provided for calculating the Bhattacharyya coefficient.
Created on 4/14/2018
Author: Eric Williamson (ericpaulwill@gmail.com)

https://github.com/EricPWilliamson/bhattacharyya-distance
"""
import numpy as np
from math import sqrt
from scipy.stats import gaussian_kde

def bhatta_coef(X1, X2, method='continuous'):
    #Calculate the Bhattacharyya coefficient between X1 and X2. X1 and X2 should be 1D numpy arrays representing the same
    # feature in two separate classes.
    
    X1 = X1[~np.isnan(X1)]
    X2 = X2[~np.isnan(X2)]

    def get_density(x, cov_factor=0.1):
        #Produces a continuous density function for the data in 'x'. Some benefit may be gained from adjusting the cov_factor.
        density = gaussian_kde(x)
        density.covariance_factor = lambda:cov_factor
        density._compute_covariance()
        return density

    #Combine X1 and X2, we'll use it later:
    cX = np.concatenate((X1,X2))

    if method == 'noiseless':
        ###This method works well when the feature is qualitative (rather than quantitative). Each unique value is
        ### treated as an individual bin.
        uX = np.unique(cX)
        A1 = len(X1) * (max(cX)-min(cX)) / len(uX)
        A2 = len(X2) * (max(cX)-min(cX)) / len(uX)
        bht = 0
        for x in uX:
            p1 = (X1==x).sum() / A1
            p2 = (X2==x).sum() / A2
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(uX)

    elif method == 'hist':
        ###Bin the values into a hardcoded number of bins (This is sensitive to N_BINS)
        N_BINS = 10
        #Bin the values:
        h1 = np.histogram(X1,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]
        h2 = np.histogram(X2,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]
        #Calc coeff from bin densities:
        bht = 0
        for i in range(N_BINS):
            p1 = h1[i]
            p2 = h2[i]
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/N_BINS

    elif method == 'autohist':
        ###Bin the values into bins automatically set by np.histogram:
        #Create bins from the combined sets:
        # bins = np.histogram(cX, bins='fd')[1]
        bins = np.histogram(cX, bins='doane')[1] #Seems to work best
        # bins = np.histogram(cX, bins='auto')[1]

        h1 = np.histogram(X1,bins=bins, density=True)[0]
        h2 = np.histogram(X2,bins=bins, density=True)[0]

        #Calc coeff from bin densities:
        bht = 0
        for i in range(len(h1)):
            p1 = h1[i]
            p2 = h2[i]
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(h1)

    elif method == 'continuous':
        ###Use a continuous density function to calculate the coefficient (This is the most consistent, but also slightly slow):
        N_STEPS = 200
        #Get density functions:
        d1 = get_density(X1)
        d2 = get_density(X2)
        #Calc coeff:
        xs = np.linspace(min(cX),max(cX),N_STEPS)
        bht = 0
        for x in xs:
            p1 = d1(x)
            p2 = d2(x)
            bht += sqrt(p1*p2)*(max(cX)-min(cX))/N_STEPS

    else:
        raise ValueError("The value of the 'method' parameter does not match any known method")

    return bht


def bhatta_dist(X1, X2, method='continuous'):
    #Calculate the Bhattacharyya distance between X1 and X2. X1 and X2 should be 1D numpy arrays representing the same
    # feature in two separate classes.

    def get_density(x, cov_factor=0.1):
        #Produces a continuous density function for the data in 'x'. Some benefit may be gained from adjusting the cov_factor.
        density = gaussian_kde(x)
        density.covariance_factor = lambda:cov_factor
        density._compute_covariance()
        return density

    #Combine X1 and X2, we'll use it later:
    cX = np.concatenate((X1,X2))

    if method == 'noiseless':
        ###This method works well when the feature is qualitative (rather than quantitative). Each unique value is
        ### treated as an individual bin.
        uX = np.unique(cX)
        A1 = len(X1) * (max(cX)-min(cX)) / len(uX)
        A2 = len(X2) * (max(cX)-min(cX)) / len(uX)
        bht = 0
        for x in uX:
            p1 = (X1==x).sum() / A1
            p2 = (X2==x).sum() / A2
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(uX)

    elif method == 'hist':
        ###Bin the values into a hardcoded number of bins (This is sensitive to N_BINS)
        N_BINS = 10
        #Bin the values:
        h1 = np.histogram(X1,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]
        h2 = np.histogram(X2,bins=N_BINS,range=(min(cX),max(cX)), density=True)[0]
        #Calc coeff from bin densities:
        bht = 0
        for i in range(N_BINS):
            p1 = h1[i]
            p2 = h2[i]
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/N_BINS

    elif method == 'autohist':
        ###Bin the values into bins automatically set by np.histogram:
        #Create bins from the combined sets:
        # bins = np.histogram(cX, bins='fd')[1]
        bins = np.histogram(cX, bins='doane')[1] #Seems to work best
        # bins = np.histogram(cX, bins='auto')[1]

        h1 = np.histogram(X1,bins=bins, density=True)[0]
        h2 = np.histogram(X2,bins=bins, density=True)[0]

        #Calc coeff from bin densities:
        bht = 0
        for i in range(len(h1)):
            p1 = h1[i]
            p2 = h2[i]
            bht += sqrt(p1*p2) * (max(cX)-min(cX))/len(h1)

    elif method == 'continuous':
        ###Use a continuous density function to calculate the coefficient (This is the most consistent, but also slightly slow):
        N_STEPS = 200
        #Get density functions:
        d1 = get_density(X1)
        d2 = get_density(X2)
        #Calc coeff:
        xs = np.linspace(min(cX),max(cX),N_STEPS)
        bht = 0
        for x in xs:
            p1 = d1(x)
            p2 = d2(x)
            bht += sqrt(p1*p2)*(max(cX)-min(cX))/N_STEPS

    else:
        raise ValueError("The value of the 'method' parameter does not match any known method")

    ###Lastly, convert the coefficient into distance:
    if bht==0:
        return float('Inf')
    else:
        return -np.log(bht)


def bhatta_dist2(x, Y, Y_selection=None, method='continuous'):
    #Same as bhatta_dist, but takes different inputs. Takes a feature 'x' and separates it by class ('Y').
    if Y_selection is None:
        Y_selection = list(set(Y))
    #Make sure Y_selection is just 2 classes:
    if len(Y_selection) != 2:
        raise ValueError("Use parameter Y_selection to select just 2 classes.")
    #Separate x into X1 and X2:
    X1 = np.array(x,dtype=np.float64)[Y==Y_selection[0]]
    X2 = np.array(x,dtype=np.float64)[Y==Y_selection[1]]
    #Plug X1 and X2 into bhatta_dist():
    return bhatta_dist(X1, X2, method=method)