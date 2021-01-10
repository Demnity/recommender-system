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


saved_data_matrix = np.load('data_matrix.npy')

print(predictions_description)
#####
##
## COLLABORATIVE FILTERING
##
#####


def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))


def create_user_user_matrix(m):
    user_user_matrix = np.zeros((m.shape[0], m.shape[0]))

    #calculate similarity between each and every user
    for i in range(m.shape[0]):
        for j in range(i, m.shape[0]):
            if i != j:
                sim = cosine_similarity(m[i], m[j])
                user_user_matrix[i, j] = sim
                user_user_matrix[j, i] = sim

            if i == j:
                user_user_matrix[i, j] = 1

    np.save('user_user_matrix_1.npy', user_user_matrix)


user_user_matrix = np.load('user_user_matrix_1.npy')


# The index should be index of the matrix not of the ones given in the file (so before use should subtract by one
# each index)
def predict_one_rating(index_user, index_movie, k_neighbours, data_matrix):
    similarities = user_user_matrix[index_user]

    #sort according to similarity
    sorted_similarities = pd.DataFrame(similarities).sort_values(by=[0], ascending=False)

    nearest_neighbours = dict()

    for index, row in sorted_similarities.iterrows():
        if index != index_user and data_matrix[index, index_movie] != 0 and len(nearest_neighbours) < k_neighbours:
            nearest_neighbours[index] = row[0]
        if len(nearest_neighbours) >= k_neighbours:
            break

    similarities_sum = 0
    rating = 0

    for index, similarity in nearest_neighbours.items():
        similarities_sum += similarity
        rating += similarity * saved_data_matrix[index, index_movie]

    if similarities_sum != 0:
        rating /= similarities_sum

    return rating


def predict_collaborative_filtering(movies, users, ratings, predictions):
    # TO COMPLETE
    result = np.empty([len(predictions), 2])

    for i, row in predictions.iterrows():
        result[i] = [i + 1, predict_one_rating(row['userID'] - 1, row['movieID'] - 1, 10, saved_data_matrix)]
        print(result[i])
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


#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function
predictions = predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)


# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)
