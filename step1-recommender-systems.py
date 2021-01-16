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

dataframe_matrix = ratings_description.set_index(['userID', 'movieID']).rating.unstack(fill_value=0)


def create_data_matrix(ratings):
    data_matrix = np.zeros((6040, 3706))
    for i, row in ratings.iterrows():
        data_matrix[row['userID'] - 1, row['movieID'] - 1] = row['rating']
    np.save('data_matrix.npy', data_matrix)


# users-to-movies matrix
saved_data_matrix = np.load('data_matrix.npy')
overall_mean = saved_data_matrix[np.nonzero(saved_data_matrix)].mean()

print(overall_mean)
#####
##
## COLLABORATIVE FILTERING
##
#####
np.seterr('raise')


def cosine_similarity(vector_a, vector_b):
    denominator = (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0:
        return 0
    return np.dot(vector_a, vector_b) / denominator


def create_utility_matrix(m, file_name, similarity_function):
    utility_matrix = np.zeros((m.shape[0], m.shape[0]))

    # calculate similarity between each and every row
    for i in range(m.shape[0]):
        for j in range(i, m.shape[0]):
            if i != j:
                sim = similarity_function(m[i], m[j])
                utility_matrix[i, j] = sim
                utility_matrix[j, i] = sim

            if i == j:
                utility_matrix[i, j] = 1

    np.save(file_name, utility_matrix)


def normalize_data_matrix(data_matrix):
    mask = data_matrix == 0
    # mask all the 0 entries in the utility matrix
    masked_arr = np.ma.masked_array(data_matrix, mask)
    item_means = np.mean(masked_arr, axis=0)
    # if an item doesn't have any ratings, default to 0
    item_means = item_means.filled(0)
    util_masked = masked_arr.filled(item_means)
    x = np.tile(item_means, (util_masked.shape[0], 1))
    # remove the per item average from all entries
    # the above mentioned nan entries will be essentially zero now
    util_masked = util_masked - x

    return util_masked


movies_to_user_matrix = saved_data_matrix.T

normalized_data_matrix = normalize_data_matrix(saved_data_matrix)
normalized_movies_to_user_matrix = normalized_data_matrix.T
print(normalized_movies_to_user_matrix)
# create_utility_matrix(saved_data_matrix, 'user_user_matrix_1.npy', cosine_similarity)
# create_utility_matrix(movies_to_user_matrix, 'item_item_matrix.npy', cosine_similarity)
# create_utility_matrix(normalized_movies_to_user_matrix, 'normalized_item_item_matrix.npy', cosine_similarity)

user_user_matrix = np.load('user_user_matrix_1.npy')
item_item_matrix = np.load('item_item_matrix.npy')
normalized_item_item_matrix = np.load('normalized_item_item_matrix.npy')

print(user_user_matrix)

print(item_item_matrix)

print(normalized_item_item_matrix)

def calculate_base_line(global_mean, user, movie):
    user_deviation = np.mean(user, axis=0) - global_mean
    movie_deviation = np.mean(movie, axis=0) - global_mean

    return global_mean + user_deviation + movie_deviation


def find_k_nearest_neighbours(data_matrix, utility_matrix, row_index, column_index, k_neighbours):
    similarities = utility_matrix[row_index]

    # sort according to similarity
    sorted_similarities = pd.DataFrame(similarities).sort_values(by=[0], ascending=False)

    nearest_neighbours = dict()

    for index, row in sorted_similarities.iterrows():
        if index != row_index and data_matrix[index, column_index] != 0 and len(nearest_neighbours) < k_neighbours:
            nearest_neighbours[index] = row[0]
        if len(nearest_neighbours) >= k_neighbours:
            break

    return nearest_neighbours


# The index should be index of the matrix not of the ones given in the file (so before use should subtract by one
# each index)
def predict_one_rating(index_user, index_movie, k_neighbours, data_matrix, utility_matrix, mode, baseline):
    row_index = ""
    column_index = ""

    if mode == "user":
        row_index = index_user
        column_index = index_movie
    elif mode == "item":
        row_index = index_movie
        column_index = index_user

    nearest_neighbours = find_k_nearest_neighbours(data_matrix, utility_matrix, row_index, column_index, k_neighbours)

    similarities_sum = 0
    rating = 0

    if baseline == True and mode == "item":
        movie_to_predict = data_matrix[index_movie]
        user_to_predict = data_matrix[:, index_user]

        for index, similarity in nearest_neighbours.items():
            similarities_sum += similarity
            current_movie = data_matrix[index]

            rating += similarity * (data_matrix[index, column_index]
                                    - calculate_base_line(overall_mean, user_to_predict, current_movie))

    else:
        for index, similarity in nearest_neighbours.items():
            similarities_sum += similarity
            rating += similarity * data_matrix[index, column_index]

    if similarities_sum != 0:
        rating /= similarities_sum

    rating += calculate_base_line(overall_mean, user_to_predict, movie_to_predict)

    return rating


def predict_collaborative_filtering(movies, users, ratings, predictions, utility_matrix, mode, data_matrix):
    # TO COMPLETE
    result = np.empty([len(predictions), 2])

    for i, row in predictions.iterrows():
        result[i] = [i + 1, predict_one_rating(row['userID'] - 1, row['movieID'] - 1, 13,
                                               data_matrix, utility_matrix, mode, True)]

    result = pd.DataFrame(result)
    result[[0]] = result[[0]].astype(int)
    result.rename(columns={0: 'Id', 1: 'Rating'}, inplace=True)
    print(result.head())
    print(np.where(result.isna()))
    return result


#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
##
## RANDOM PREDICTORS
## //!!\\ TO CHANGE
##
#####

# By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


####
#
# SAVE RESULTS
#
####

# //!!\\ TO CHANGE by your prediction function
# predictions = predict_collaborative_filtering(movies_description, users_description,
#                                               ratings_description, predictions_description,
#                                               normalized_item_item_matrix, "item", movies_to_user_matrix)
# predictions.to_csv('./data/submission.csv', index=False)
