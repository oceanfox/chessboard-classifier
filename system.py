"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import scipy
import numpy as np

from scipy.spatial import distance

N_DIMENSIONS = 10
K_NEAREST = 5

PIECES = {".": 64, "K": 1, "Q": 1, "B": 2, "N": 2, "R": 2, "P": 8, "k": 1, "q": 1, "b": 2, "n": 2, "r": 2, "p": 8}
def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # KNN Classifier
    predictions = []

    for test_instance in test:
        distances = distance.cdist(train, [test_instance], 'euclidean').flatten()
        indices = np.argsort(distances)[:K_NEAREST]
        nearest_labels = train_labels[indices]
        nearest_labels_list = nearest_labels.tolist()  # Convert numpy array to list
        predicted_label = max(set(nearest_labels_list), key=nearest_labels_list.count)
        predictions.append(predicted_label)

    return predictions


# The functions below must all be fprovided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    # reduced_data = data[:, 0:N_DIMENSIONS]
    reduced_data = np.dot((data - np.mean(data, axis=0)), model["eigenvectors"])

    # Subtract the mean
    # data -= np.mean(data, axis=0)

    # Compute the SVD
    # U, _, vt = np.linalg.svd(data, full_matrices=False)

    # Project the data onto the first N_DIMENSIONS singular vectors
    # reduced_data = U[:,:N_DIMENSIONS]

    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.



    # PCA
    model = {}
    model["labels_train"] = labels_train.tolist()

    covariance_matrix = np.cov(fvectors_train, ddof=False, rowvar=False)
    N = covariance_matrix.shape[0]

    _, eigenvectors = scipy.linalg.eigh(covariance_matrix, eigvals=(N - N_DIMENSIONS, N - 1))

    eigenvectors = np.fliplr(eigenvectors)

    model["eigenvectors"] = eigenvectors.tolist()

    # print(model)

    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()

    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    labels = []


    n = 64

    for i in range(0, len(fvectors_test), n):
        fvectors_test_board = fvectors_test[i:i + n]

        to_reeval = {".": [], "K": [], "Q": [], "B": [], "N": [], "R": [], "P": [], "k": [], "q": [], "b": [], "n": [], "r": [], "p": []}

        labels_board = []

        for i, square in enumerate(fvectors_test_board):
            # Calculate the distance to each training sample
            distances = distance.cdist(fvectors_train, square.reshape(1, -1), 'euclidean').flatten()
            indices = np.argsort(distances)[:K_NEAREST]
            nearest_labels = labels_train[indices]

            # Calculate the weighted count of each class using the defined piece counts
            if i < 8 or i > 55:
                nearest_labels = nearest_labels[nearest_labels != "p"]
                nearest_labels = nearest_labels[nearest_labels != "P"]

            nearest_labels_list = nearest_labels.tolist()

            # for label in nearest_labels:
            #     if label in class_weights:
            #         class_weights[label] += PIECES.get(label, 0)
            #     else:
            #         class_weights[label] = PIECES.get(label, 0)

            # Choose the label with the maximum weighted count
            predicted_label = max(nearest_labels_list, key=nearest_labels_list.count)

            labels_board.append(predicted_label)

        for label, count in PIECES.items():
            if labels_board.count(label) > count:
                for i, square in enumerate(labels_board):
                    if square == label:
                        to_reeval[square].append(i)

        for piece, positions in to_reeval.items():
            if len(positions) == 0:
                continue
            curr_pieces_labels = []
            for pos in positions:
                distances = distance.cdist(fvectors_train, fvectors_test_board[pos].reshape(1, -1), 'euclidean').flatten()
                indices = np.argsort(distances)[:K_NEAREST]
                nearest_labels_list = labels_train[indices].tolist()

                # print(curr_pieces_labels)
                curr_pieces_labels.append(list(zip(labels_train[indices], indices)))
                # predicted_label = max(nearest_labels_list, key=nearest_labels_list.count)
                # curr_pieces_labels[positions.index(pos)] = nearest_labels_list

            sorted_labels = sorted(curr_pieces_labels, key=lambda x: x[1])

            # print(f"{piece} {positions}")
            # for piece in sorted_labels:

            # print(max(sublist[-1][-1] for sublist in sorted_labels))
            # print([sorted(sublist, key=lambda x: x[1], reverse=True)[1] for sublist in sorted_labels])

            # print(sorted_labels)



        labels.append(labels_board)

    labels = np.array(labels).reshape(1600)

    return labels


    # return classify_squares(fvectors_test, model)
