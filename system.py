"""
PCA & KNN Chessboard classification system.

Solution outline for the COM2004/3004 assignment.

Name:           Arnav Kolhe
Student ID:     220131948

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.1
"""
from collections import Counter
from typing import List

import scipy
import numpy as np

from scipy.spatial import distance

from scipy.ndimage import gaussian_filter

N_DIMENSIONS = 10
K_NEAREST = 5

BOARD_SIZE = 64
ROW_SIZE = 8

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
    # Apply PCA
    pca_data = np.dot((data - np.mean(data, axis=0)), model["pca_eigenvectors"])

    # Apply LDA
    lda_data = np.dot((pca_data - np.mean(pca_data, axis=0)), model["lda_eigenvectors"])

    return lda_data

def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    N_DIMENSIONS_PCA = 60
    N_DIMENSIONS_LDA = 10

    model = {}
    model["labels_train"] = labels_train.tolist()

    # PCA
    covariance_matrix = np.cov(fvectors_train, ddof=False, rowvar=False)
    N = covariance_matrix.shape[0]

    _, pca_eigenvectors = scipy.linalg.eigh(covariance_matrix, eigvals=(N - N_DIMENSIONS_PCA, N - 1))
    pca_eigenvectors = np.fliplr(pca_eigenvectors)
    model["pca_eigenvectors"] = pca_eigenvectors.tolist()

    pca_data = np.dot((fvectors_train - np.mean(fvectors_train, axis=0)), pca_eigenvectors)

    # LDA
    mean_vectors = []
    for label in np.unique(labels_train):
        mean_vectors.append(np.mean(pca_data[labels_train == label], axis=0))

    mean_vectors = np.array(mean_vectors)
    overall_mean = np.mean(pca_data, axis=0)

    scatter_within = np.zeros((pca_data.shape[1], pca_data.shape[1]))

    for label in np.unique(labels_train):
        scatter_within += np.cov(pca_data[labels_train == label], rowvar=False)

    scatter_between = np.cov(mean_vectors, rowvar=False) - np.outer(overall_mean, overall_mean)

    _, lda_eigenvectors = scipy.linalg.eigh(scipy.linalg.pinv(scatter_within).dot(scatter_between), eigvals=(pca_data.shape[1] - N_DIMENSIONS_LDA, pca_data.shape[1] - 1))
    lda_eigenvectors = np.fliplr(lda_eigenvectors)

    model["lda_eigenvectors"] = lda_eigenvectors.tolist()

    lda_data = np.dot(pca_data, lda_eigenvectors)

    model["fvectors_train"] = lda_data.tolist()

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
        image = gaussian_filter(image, sigma=1)
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
            if i < ROW_SIZE or i >= BOARD_SIZE - ROW_SIZE:
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
                # curr_pieces_labels.append(list(zip(labels_train[indices], indices)))
                curr_pieces_labels.append(list(zip(nearest_labels_list, distances[indices])))
                # predicted_label = max(nearest_labels_list, key=nearest_labels_list.count)
                # curr_pieces_labels[positions.index(pos)] = nearest_labels_list

            sorted_labels = sorted(curr_pieces_labels, key=lambda x: x[1])

            flat_labels = [label for sublist in curr_pieces_labels for label, _ in sublist]

            label_counts = Counter(flat_labels)

            most_common_labels = [label for label, _ in label_counts.most_common(PIECES[piece])]

            for i, label in enumerate(most_common_labels):
                labels_board[positions[i]] = label

            # print(f"{piece} {positions}")
            # for piece in sorted_labels:
            # print(max(sublist[-1][-1] for sublist in sorted_labels))
            # print([sorted(sublist, key=lambda x: x[1], reverse=True)[1] for sublist in sorted_labels])

            # print(sorted_labels)



        labels.append(labels_board)

    labels = np.array(labels).reshape(1600)

    return labels


    # return classify_squares(fvectors_test, model)
