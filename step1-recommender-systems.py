import sys

import numpy as np
import pandas as pd
import os.path
from random import randint

# -*- coding: utf-8 -*-
from numpy import savetxt

"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

# Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID': 'int', 'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int', 'gender': 'str', 'age': 'int', 'profession': 'int'},
                                names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)

#####
##
## CREATING UTILITY MATRIX AND SAVE AS CSV
##
#####

# df = pd.DataFrame(0, index=np.arange(1, users_description.shape[0] + 1),
#                   columns=np.arange(1, movies_description.shape[0] + 1))
# for i, row in ratings_description.iterrows():
#     df.loc[row['userID'], row['movieID']] = row['rating']
# df.to_csv('./data/utility_matrix.csv')

util_matrix = pd.read_csv('./data/utility_matrix.csv', index_col=0)
util_matrix.columns = util_matrix.columns.astype(int)
util_matrix = util_matrix.to_numpy()
print(util_matrix.shape)
print(util_matrix[:5])


def predict_collaborative_filtering(movies, users, ratings, predictions):
    # TO COMPLETE

    pass


############################################################################################################
##
## Latent factor
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    # find all the 0 entries
    mask = util_matrix == 0
    # mask all the 0 entries in the utility matrix
    masked_arr = np.ma.masked_array(util_matrix, mask)
    item_means = np.mean(masked_arr, axis=0)
    # if an item doesn't have any ratings, default to 0
    item_means = item_means.filled(0)
    util_masked = masked_arr.filled(item_means)
    x = np.tile(item_means, (util_masked.shape[0], 1))
    # remove the per item average from all entries
    # the above mentioned nan entries will be essentially zero now
    util_masked = util_masked - x

    k = 16
    u, s, vt = np.linalg.svd(util_masked)
    s = np.diag(s)
    s = s[0:k, 0:k]
    u = u[:, 0:k]
    vt = vt[0:k, :]
    print(u.shape, s.shape, vt.shape)
    # remember to add the average to the output
    predict_all = np.dot(np.dot(u, s), vt) + x
    result = np.empty([len(predictions), 2])

    for i, row in predictions.iterrows():
        output = predict_all[row['userID'] - 1, row['movieID'] - 1]
        # if output is 0, it means that a movie is not rated by anyone, set the movie rating to user's avg rating
        if output == 0:
            result[i] = [i + 1, np.mean(predict_all[row['userID'] - 1, :])]
        else:
            result[i] = [i + 1, predict_all[row['userID'] - 1, row['movieID'] - 1]]

    result = pd.DataFrame(result)
    result[[0]] = result[[0]].astype(int)
    result.rename(columns={0: 'Id', 1: 'Rating'}, inplace=True)
    print(result.head())
    print(np.where(result.isna()))
    return result


############################################################################################################
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    ## TO COMPLETE
    # # find all the 0 entries
    # mask = util_matrix == 0
    # # mask all the 0 entries in the utility matrix
    # masked_arr = np.ma.masked_array(util_matrix, mask)
    # item_means = np.mean(masked_arr, axis=0)
    # user_means = np.mean(masked_arr, axis=1)
    # item_means = item_means.filled(0)
    # user_means = user_means.filled(0)
    # rating_mean = np.mean(masked_arr)
    # bias_user = user_means - rating_mean
    # bias_item = item_means - rating_mean

    # if an item doesn't have any ratings, default to 0

    # util_masked = masked_arr.filled(item_means)
    # x = np.tile(item_means, (util_masked.shape[0], 1))
    # b_user = np.tile(bias_user, (util_masked.shape[1], 1)).transpose()
    # b_item = np.tile(bias_item, (util_masked.shape[0], 1))
    # print(b_user.shape, b_item.shape)
    # remove the per item average from all entries
    # the above mentioned nan entries will be essentially zero now
    # util_masked = util_masked - x
    # num of latent factors
    k = 9

    U = np.ones((util_matrix.shape[0], k))
    V = np.ones((k, util_matrix.shape[1]))
    util_masked = util_matrix.astype(float)
    # make all 0 entries into NaN
    util_masked[util_masked == 0] = np.nan

    # num of times you change the non-zero element in U and V
    times = 50

    for i in range(0, times):
        decompose_matrix_U(U, V, util_masked)
        decompose_matrix_V(U, V, util_masked)

    predict_all = np.dot(U, V)
    result = np.empty([len(predictions), 2])

    for i, row in predictions.iterrows():
        output = predict_all[row['userID'] - 1, row['movieID'] - 1]
        # if output is 0, it means that a movie is not rated by anyone, set the movie rating to user's avg rating
        if output == 0:
            result[i] = [i + 1, np.mean(predict_all[row['userID'] - 1, :])]
        else:
            result[i] = [i + 1, predict_all[row['userID'] - 1, row['movieID'] - 1]]

    result = pd.DataFrame(result)
    result[[0]] = result[[0]].astype(int)
    result.rename(columns={0: 'Id', 1: 'Rating'}, inplace=True)
    print(result.head())
    print(np.where(result.isna()))
    return result

def decompose_matrix_U(U, V, util_masked):
    for r in range(0, U.shape[0]):
        for s in range(0, V.shape[0]):
            m_array = np.array(util_masked[r, :])
            v_array = np.array(V[s, :])
            v_array[np.isnan(m_array)] = np.nan
            # print m_array
            # print v_array
            denominator = np.nansum(np.square(v_array))
            # print denominator
            sum_array = np.matmul(U[r, :], V[:]) - (U[r, s] * V[s, :])
            # # print sum_array
            numerator = np.nansum(V[s, :] * (m_array - sum_array))
            # numerator = np.nansum(V[s, :] * sum)
            # print numerator
            U[r, s] = float(numerator) / denominator

    return


def decompose_matrix_V(U, V, util_masked):
    for s in range(0, V.shape[1]):
        for r in range(0, U.shape[1]):
            m_array = np.array(util_masked[:, s])
            u_array = np.array(U[:, r])
            u_array[np.isnan(m_array)] = np.nan
            # print m_array
            # print v_array
            denominator = np.nansum(np.square(u_array))
            if denominator == 0:
                V[r, s] = 0
                continue
            # print denominator
            sum_array = np.matmul(U[:], V[:, s]) - (V[r, s] * U[:, r])
            # print sum_array
            numerator = np.nansum(U[:, r] * (m_array - sum_array))
            # print numerator
            V[r, s] = float(numerator) / denominator

    return

############################################################################################################
##
## Gradient descent
##
############################################################################################################

def predict_final_1(movies, users, ratings, predictions):
    ## TO COMPLETE
    # # find all the 0 entries
    # mask = util_matrix == 0
    # # mask all the 0 entries in the utility matrix
    # masked_arr = np.ma.masked_array(util_matrix, mask)
    # item_means = np.mean(masked_arr, axis=0)
    # user_means = np.mean(masked_arr, axis=1)
    # item_means = item_means.filled(0)
    # user_means = user_means.filled(0)
    # rating_mean = np.mean(masked_arr)
    # bias_user = user_means - rating_mean
    # bias_item = item_means - rating_mean

    # if an item doesn't have any ratings, default to 0

    # util_masked = masked_arr.filled(item_means)
    # x = np.tile(item_means, (util_masked.shape[0], 1))
    # b_user = np.tile(bias_user, (util_masked.shape[1], 1)).transpose()
    # b_item = np.tile(bias_item, (util_masked.shape[0], 1))
    # print(b_user.shape, b_item.shape)
    # remove the per item average from all entries
    # the above mentioned nan entries will be essentially zero now
    # util_masked = util_masked - x
    # num of latent factors
    k = 9

    U = np.ones((util_matrix.shape[0], k))
    V = np.ones((k, util_matrix.shape[1]))
    util_masked = util_matrix.astype(float)
    # make all 0 entries into NaN
    util_masked[util_masked == 0] = np.nan

    # num of times you change the non-zero element in U and V
    times = 50

    for i in range(0, times):
        decompose_matrix_U(U, V, util_masked)
        decompose_matrix_V(U, V, util_masked)

    predict_all = np.dot(U, V)
    result = np.empty([len(predictions), 2])

    for i, row in predictions.iterrows():
        output = predict_all[row['userID'] - 1, row['movieID'] - 1]
        # if output is 0, it means that a movie is not rated by anyone, set the movie rating to user's avg rating
        if output == 0:
            result[i] = [i + 1, np.mean(predict_all[row['userID'] - 1, :])]
        else:
            result[i] = [i + 1, predict_all[row['userID'] - 1, row['movieID'] - 1]]

    result = pd.DataFrame(result)
    result[[0]] = result[[0]].astype(int)
    result.rename(columns={0: 'Id', 1: 'Rating'}, inplace=True)
    print(result.head())
    print(np.where(result.isna()))
    return result

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    for step in range(steps):
        for i in range(P.shape[0]):
            for j in range(Q.shape[1]):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q




#####
##
## SAVE RESULTS
##
#####    

predictions = predict_final(movies_description, users_description, ratings_description,
                                     predictions_description)
# Save predictions
predictions.to_csv('./data/submission.csv', index=False)
