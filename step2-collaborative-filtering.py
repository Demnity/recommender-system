import numpy as np
import pandas as pd

# -*- coding: utf-8 -*-
"""
FRAMEWORK FOR DATA MINING CLASS
### NOTES
This files is an example of what your code should look like. 
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
movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])

# Load the saved data_matrix
saved_data_matrix = np.load('data_matrix.npy')

# Calculate the mean of the entire matrix
overall_mean = saved_data_matrix[np.nonzero(saved_data_matrix)].mean()


# Calculate the cosine similarity between two users
def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two users/movies given their vectors

    :param vector_a: first vector of a movie/user

    :param vector_b: second vector of a movie/user

    :return: the cosine similarity of the movie and use
    """
    denominator = (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0:
        return 0
    return np.dot(vector_a, vector_b) / denominator


def normalize_data_matrix(data_matrix):
    """
    Function to normalize the data matrix by subtracting the non-zero element of each row by the row's mean.
    It is used for finding the Pearson correlation later on

    :param data_matrix: the matrix with the ratings, the rows can be either items (for item-item CF) or users
    (for user-user CF)

    :return: a normalized version of the data matrix
    """
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


def create_utility_matrix(data_matrix, file_name, similarity_function):
    """
    Create the utility matrix which contains the similarities between each and every user/items and save it for
    later use

    :param data_matrix: the matrix with the ratings, the rows can be either items (for item-item CF) or users
    (for user-user CF)

    :param file_name: name of the file

    :param similarity_function: function to calculate the similarity
    """

    utility_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))

    # calculate similarity between each and every row
    for i in range(data_matrix.shape[0]):
        for j in range(i, data_matrix.shape[0]):
            if i != j:
                sim = similarity_function(data_matrix[i], data_matrix[j])
                utility_matrix[i, j] = sim
                utility_matrix[j, i] = sim

            if i == j:
                utility_matrix[i, j] = 1

    np.save(file_name, utility_matrix)


movies_to_user_matrix = saved_data_matrix.T
normalized_data_matrix = normalize_data_matrix(saved_data_matrix)
normalized_movies_to_user_matrix = normalized_data_matrix.T

# create_utility_matrix(movies_to_user_matrix, 'item_item_matrix.npy', cosine_similarity)
# create_utility_matrix(normalized_movies_to_user_matrix, 'normalized_item_item_matrix.npy', cosine_similarity)

item_item_matrix = np.load('item_item_matrix.npy')
normalized_item_item_matrix = np.load('normalized_item_item_matrix.npy')


def calculate_base_line(global_mean, user, movie):
    """
    Calculate the global baseline given a vector of a user and movie

    :param global_mean: the mean of all the ratings

    :param user: a vector representing a user

    :param movie: a vector representing a movie

    :return: a baseline rating based on the deviation from the global mean
    """

    user_deviation = np.mean(user, axis=0) - global_mean
    movie_deviation = np.mean(movie, axis=0) - global_mean

    return global_mean + user_deviation + movie_deviation


def find_k_nearest_neighbours(data_matrix, utility_matrix, row_index, column_index, k_neighbours):
    """
    Find the k nearest neighbours of a movie/user given by the row_index

    :param data_matrix: the matrix with the ratings, the rows can be either items (for item-item CF) or users
    (for user-user CF)

    :param utility_matrix: the matrix with the similarities between items (for item-item CF) or users (for user-user CF)

    :param row_index: the index of the row to find the nearest neighbours for

    :param column_index: the index of the column for which a rating should be present

    :param k_neighbours: the number of neighbours to query for

    :return: a dictionary of the nearest neighbours with the key as the index and the value as the similarity
    """

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
def predict_one_rating(index_user, index_movie, k_neighbours, data_matrix, utility_matrix, mode, baseline):
    """
    Given the index of the user and the index of the movie predict the rating based on the collaborative filtering
    method

    :param index_user: index of the user to be predicted

    :param index_movie: index of the movie to be predicted

    :param k_neighbours: the number of neighbours

    :param data_matrix: the matrix with the ratings, the rows can be either items (for item-item CF) or users
    (for user-user CF)

    :param utility_matrix: the matrix with the similarities between items (for item-item CF) or users (for user-user CF)

    :param mode: the mode can be either "user" for user-user CF or "item" for item-item CF

    :param baseline: True or False indicating whether a global baseline should be taken into account or not

    :return: the predicted rating of the given user and movie
    """

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

        if similarities_sum != 0:
            rating /= similarities_sum

        # Add the global baseline for the movie
        rating += calculate_base_line(overall_mean, user_to_predict, movie_to_predict)

    else:
        for index, similarity in nearest_neighbours.items():
            similarities_sum += similarity
            rating += similarity * data_matrix[index, column_index]

        if similarities_sum != 0:
            rating /= similarities_sum

    return rating


def predict(prediction_list, k_neighbours, data_matrix, utility_matrix, mode, baseline):
    """
    Given a list of what to predict, predict the ratings of all the user-movie pairs
    :param prediction_list: list of what to predict

    :param k_neighbours: number of neighbours to take into account

    :param data_matrix: the matrix with the ratings, the rows can be either items (for item-item CF) or users
    (for user-user CF)

    :param utility_matrix: the matrix with the similarities between items (for item-item CF) or users (for user-user CF)

    :param mode: the mode can be either "user" for user-user CF or "item" for item-item CF

    :param baseline: True or False indicating whether a global baseline should be taken into account or not

    :return: a pandas Dataframe of all the predictions
    """
    # TO COMPLETE
    result = np.empty([len(prediction_list), 2])

    for i, row in prediction_list.iterrows():
        result[i] = [i + 1, predict_one_rating(row['userID'] - 1, row['movieID'] - 1, k_neighbours,
                                               data_matrix, utility_matrix, mode, baseline)]

    result = pd.DataFrame(result)
    result[[0]] = result[[0]].astype(int)
    result.rename(columns={0: 'Id', 1: 'Rating'}, inplace=True)
    return result


#####
##
## SAVE RESULTS
##
#####    

predictions = predict(predictions_description, 13, movies_to_user_matrix, normalized_item_item_matrix, "item", True)

predictions.to_csv('./data/step2-CF-submission.csv', index=False)
