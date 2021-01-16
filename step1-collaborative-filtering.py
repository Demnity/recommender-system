import numpy as np
import pandas as pd

# -*- coding: utf-8 -*-
from numpy import savetxt

"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
#
# DATA IMPORT
#
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


# Create and save the data_matrix for use without re-computation
def create_data_matrix(ratings):
    data_matrix = np.zeros((6040, 3706))
    for i, row in ratings.iterrows():
        data_matrix[row['userID'] - 1, row['movieID'] - 1] = row['rating']
    np.save('data_matrix.npy', data_matrix)


# create_data_matrix(ratings_description)

# Load the saved data_matrix
saved_data_matrix = np.load('data_matrix.npy')


#####
#
# COLLABORATIVE FILTERING
#
#####

# Calculate the cosine similarity between two users
def cosine_similarity(vector_a, vector_b):
    denominator = (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0:
        return 0
    return np.dot(vector_a, vector_b) / denominator


# Create the utility matrix which contains the similarities between each and every user and save for later use
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


# create_utility_matrix(saved_data_matrix, 'user_user_matrix.npy', cosine_similarity)

# Load the saved similarity matrix
user_user_matrix = np.load('user_user_matrix.npy')


def find_k_nearest_neighbours(data_matrix, utility_matrix, row_index, column_index, k_neighbours):
    similarities = utility_matrix[row_index]

    # sort according to similarity
    sorted_similarities = pd.DataFrame(similarities).sort_values(by=[0], ascending=False)

    nearest_neighbours = dict()

    for index, row in sorted_similarities.iterrows():
        if index != row_index and data_matrix[index, column_index] != 0 \
                and len(nearest_neighbours) < k_neighbours and row[0] > 0:
            nearest_neighbours[index] = row[0]
        if len(nearest_neighbours) >= k_neighbours or row[0] <= 0:
            break

    return nearest_neighbours


# The index should be index of the matrix not of the ones given in the file (so before use should subtract by one
# each index)
def predict_one_rating(index_user, index_movie, k_neighbours, data_matrix, utility_matrix):
    row_index = index_user
    column_index = index_movie

    nearest_neighbours = find_k_nearest_neighbours(data_matrix, utility_matrix, row_index, column_index, k_neighbours)

    similarities_sum = 0
    rating = 0

    for index, similarity in nearest_neighbours.items():
        similarities_sum += similarity
        rating += similarity * data_matrix[index, column_index]

    if similarities_sum != 0:
        rating /= similarities_sum

    return rating


def predict_collaborative_filtering(predictions, utility_matrix, data_matrix):
    # TO COMPLETE
    result = np.empty([len(predictions), 2])

    for i, row in predictions.iterrows():
        result[i] = [i + 1, predict_one_rating(row['userID'] - 1, row['movieID'] - 1, 15, data_matrix, utility_matrix)]

    result = pd.DataFrame(result)
    result[[0]] = result[[0]].astype(int)
    result.rename(columns={0: 'Id', 1: 'Rating'}, inplace=True)

    return result


####
#
# SAVE RESULTS
#
####

# //!!\\ TO CHANGE by your prediction function
predictions = predict_collaborative_filtering(predictions_description, user_user_matrix, saved_data_matrix)
predictions.to_csv('./data/step1-CF-submission.csv', index=False)
