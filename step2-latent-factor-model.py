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

###########################################################################
#                   CREATING UTILITY MATRIX AND SAVE AS CSV

# df = pd.DataFrame(0, index=np.arange(1, users_description.shape[0] + 1),
#                   columns=np.arange(1, movies_description.shape[0] + 1))
# for i, row in ratings_description.iterrows():
#     df.loc[row['userID'], row['movieID']] = row['rating']
# df.to_csv('./data/utility_matrix.csv')

###########################################################################
#                   READ THE SAVED UTILITY MATRIX

util_matrix = pd.read_csv('./data/utility_matrix.csv', index_col=0)
util_matrix.columns = util_matrix.columns.astype(int)
util_matrix = util_matrix.to_numpy()
print(util_matrix.shape)
print(util_matrix[:5])


###########################################################################
#                   U-V DECOMPOSITION USING AN ITERATIVE
#                      APPROACH WITH REGULARIZATION

def predict_iterative(predictions):
    # num of latent factors
    k = 9

    # regularization alpha
    alpha = 0.2

    # initialize U V matrix with 1s
    U = np.ones((util_matrix.shape[0], k))
    V = np.ones((k, util_matrix.shape[1]))

    util_masked = util_matrix.astype(float)
    # make all 0 entries into NaN
    util_masked[util_masked == 0] = np.nan

    # num of times you change the non-zero element in U and V
    times = 50

    for i in range(0, times):
        decompose_matrix_U(U, V, util_masked, alpha)
        decompose_matrix_V(U, V, util_masked, alpha)

    predict_all = np.dot(U, V)
    result = np.empty([len(predictions), 2])

    for i, row in predictions.iterrows():
        output = predict_all[row['userID'] - 1, row['movieID'] - 1]
        # if output is 0, it means that a movie is not rated by anyone, set the movie rating to user's avg rating
        if output == 0:
            result[i] = [i + 1, np.mean(predict_all[row['userID'] - 1, :])]
        else:
            result[i] = [i + 1, predict_all[row['userID'] - 1, row['movieID'] - 1]]

    # format result
    result = pd.DataFrame(result)
    result[[0]] = result[[0]].astype(int)
    result.rename(columns={0: 'Id', 1: 'Rating'}, inplace=True)

    print(result.head())

    return result


def decompose_matrix_U(U, V, util_masked, alpha):
    for r in range(0, U.shape[0]):
        for s in range(0, V.shape[0]):
            m_array = np.array(util_masked[r, :])
            v_array = np.array(V[s, :])
            v_array[np.isnan(m_array)] = np.nan
            denominator = np.nansum(np.square(v_array)) + alpha
            sum_array = np.matmul(U[r, :], V[:]) - (U[r, s] * V[s, :])
            numerator = np.nansum(V[s, :] * (m_array - sum_array))
            U[r, s] = float(numerator) / denominator

    return


def decompose_matrix_V(U, V, util_masked, alpha):
    for s in range(0, V.shape[1]):
        for r in range(0, U.shape[1]):
            m_array = np.array(util_masked[:, s])
            u_array = np.array(U[:, r])
            u_array[np.isnan(m_array)] = np.nan
            denominator = np.nansum(np.square(u_array)) + alpha
            if denominator == 0:
                V[r, s] = 0
                continue
            sum_array = np.matmul(U[:], V[:, s]) - (V[r, s] * U[:, r])
            numerator = np.nansum(U[:, r] * (m_array - sum_array))
            V[r, s] = float(numerator) / denominator

    return


###########################################################################
#                   U-V DECOMPOSITION USING BATCH GRADIENT
#                 DESCENT WITH REGULARIZATION AND GLOBAL BIAS

def find_bias(matrix):
    # find all the 0 entries
    mask = matrix == 0

    # mask all the 0 entries in the utility matrix
    masked_arr = np.ma.masked_array(matrix, mask)
    item_means = np.mean(masked_arr, axis=0)
    user_means = np.mean(masked_arr, axis=1)
    global_mean = np.mean(masked_arr)

    # if an item doesn't have any ratings, default to 0
    item_means = item_means.filled(0)
    user_means = user_means.filled(0)

    return item_means, user_means, global_mean


def predict_batch_gradient_descent(predictions):
    item_means, user_means, global_mean = find_bias(util_matrix)
    print(item_means, user_means, global_mean)

    # num of latent factors
    k = 9

    U = np.ones((util_matrix.shape[0], k))
    V = np.ones((k, util_matrix.shape[1]))
    util_masked = util_matrix.astype(float)
    # make all 0 entries into NaN
    # util_masked[util_masked == 0] = np.nan

    U, V = matrix_factorization(util_masked, U, V, k, item_means, user_means, global_mean)

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

    return result


def matrix_factorization(R, P, Q, K, item_means, user_means, global_mean, steps=5000, alpha=0.0002, beta=0.01):
    for step in range(steps):
        for i in range(P.shape[0]):
            for j in range(Q.shape[1]):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j]) - global_mean - user_means[i] - item_means[j]
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j]
                                                     - 2 * beta * (P[i][k] + user_means[i] + item_means[j]))
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k]
                                                     - 2 * beta * (Q[k][j] + user_means[i] + item_means[j]))
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + beta * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q


###########################################################################
#                           SAVE RESULTS

output = predict_iterative(predictions_description)
# Save predictions
output.to_csv('./data/latent_factor.csv', index=False)
