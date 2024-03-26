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
from collections import Counter, defaultdict
from typing import List

import scipy
import numpy as np

from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

N_DIMENSIONS = 10
K_NEAREST = 4

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


# CLASSIFICATION METHODS ------------------------------------------------------------------------------------------------------------
def get_nearest_labels_and_distances(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray):
    distances = distance.cdist(train, test.reshape(1,-1), 'euclidean').flatten()
    indices = np.argsort(distances)[:K_NEAREST]
    nearest_labels = train_labels[indices]
    nearest_distances = distances[indices]

    return nearest_labels, nearest_distances

def verify_piece_numbers(to_reeval, labels_board):
    for label, count in PIECES.items():
        if labels_board.count(label) > count:
            for i, square in enumerate(labels_board):
                if square == label:
                    to_reeval[square].append(i)
    return to_reeval

def remove_pawns(i, nearest_labels):
    # Remove pawns from the first and last rows
    if i < ROW_SIZE or i >= BOARD_SIZE - ROW_SIZE:
        nearest_labels = nearest_labels[nearest_labels != "p"]
        nearest_labels = nearest_labels[nearest_labels != "P"]
    return nearest_labels

def weigh_classes_from_distances(nearest_labels, nearest_distances):
    class_weights = defaultdict(float)

    # Calculate weighted frequency of each label
    for label, dist in zip(nearest_labels, nearest_distances):
        # Calculate weights from distances, you can use any function for weights
        # Here, we use the inverse of distance
        weight = 1.0 / dist if dist != 0 else 1.0
        class_weights[label] += weight
    return class_weights

# END CLASSIFICATION METHODS ------------------------------------------------------------------------------------------------------------

# PCA & LDA METHODS ------------------------------------------------------------------------------------------------------------

def run_pca(data: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """
    Run PCA on the input data using the PCA eigenvectors stored in the model.

    Parameters:
        data (np.ndarray): The input data to be dimensionally reduced.
        model (dict): A dictionary containing the PCA eigenvectors.

    Returns:
        np.ndarray: The dimensionally reduced data.
    """
    return np.dot((data - np.mean(data, axis=0)), eigenvectors)

def run_lda(data: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """
    Run LDA on the input data using the LDA eigenvectors stored in the model.

    Parameters:
        data (np.ndarray): The input data to be dimensionally reduced.
        model (dict): A dictionary containing the LDA eigenvectors.

    Returns:
        np.ndarray: The dimensionally reduced data.
    """
    return np.dot(data, eigenvectors)

# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.

def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """
    Reduce the dimensions of the input data using PCA and LDA.

    Parameters:
        data (np.ndarray): The input data to be dimensionally reduced.
        model (dict): A dictionary containing the PCA and LDA models.

    Returns:
        np.ndarray: The dimensionally reduced data.
    """

    # Apply PCA
    pca_data = run_pca((data - np.mean(data, axis=0)), np.array(model["pca_eigenvectors"]))

    # Apply LDA
    lda_data = run_lda((pca_data - np.mean(pca_data, axis=0)), np.array(model["lda_eigenvectors"]))

    return lda_data

def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """
    Process the training data to extract features using PCA and LDA.

    Args:
        fvectors_train (np.ndarray): The feature vectors of the training data.
        labels_train (np.ndarray): The labels of the training data.

    Returns:
        dict: A dictionary containing the processed training data, including the labels, PCA eigenvectors,
              LDA eigenvectors, and the transformed feature vectors.
    """
    # choose the number of dimensions for PCA and LDA
    N_DIMENSIONS_PCA = 50
    N_DIMENSIONS_LDA = N_DIMENSIONS

    model = {}
    model["labels_train"] = labels_train.tolist()

    
    # PCA
    # calculate the covariance matrix of the training data
    covariance_matrix = np.cov(fvectors_train, ddof=False, rowvar=False)

    # calculate the eigenvectors of the covariance matrix
    N = covariance_matrix.shape[0]
    _, pca_eigenvectors = scipy.linalg.eigh(covariance_matrix, eigvals=(N - N_DIMENSIONS_PCA, N - 1))

    # flip the order of the eigenvectors and add to the model
    pca_eigenvectors = np.fliplr(pca_eigenvectors)
    model["pca_eigenvectors"] = pca_eigenvectors.tolist()

    # project the training data using the eigenvectors
    pca_data = run_pca((fvectors_train - np.mean(fvectors_train, axis=0)), pca_eigenvectors)

    # LDA
    # calculate the mean vectors for each class label
    mean_vectors = []
    for label in np.unique(labels_train):
        mean_vectors.append(np.mean(pca_data[labels_train == label], axis=0))

    # convert to numpy array
    mean_vectors = np.array(mean_vectors)

    # calculate the overall mean
    overall_mean = np.mean(pca_data, axis=0)

    # calculate the within-class and between-class scatter matrices
    scatter_within = np.zeros((pca_data.shape[1], pca_data.shape[1]))

    # for each class label, calculate the within-class scatter matrix (S_W)
    for label in np.unique(labels_train):
        scatter_within += np.cov(pca_data[labels_train == label], rowvar=False)

    # calculate the between-class scatter matrix (S_B)
    scatter_between = np.cov(mean_vectors, rowvar=False) - np.outer(overall_mean, overall_mean)

    # calculate the eigenvectors of the matrix (S_W^-1 * S_B)
    _, lda_eigenvectors = scipy.linalg.eigh(np.linalg.pinv(scatter_within).dot(scatter_between), eigvals=(pca_data.shape[1] - N_DIMENSIONS_LDA, pca_data.shape[1] - 1))
    
    # flip the order of the eigenvectors and add to the model
    lda_eigenvectors = np.fliplr(lda_eigenvectors)

    # add the eigenvectors for lda to the model
    model["lda_eigenvectors"] = lda_eigenvectors.tolist()

    lda_data = run_lda(pca_data, lda_eigenvectors)

    # add the transformed feature vectors to the model
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
    test = np.array(fvectors_test)
    
    predictions = []

    for test_instance in test:
        # Calculate the distance to each training sample
        nearest_labels, nearest_distances = get_nearest_labels_and_distances(fvectors_train, labels_train, test_instance)
        
        # Calculate weighted frequency of each label
        class_weights = weigh_classes_from_distances(nearest_labels, nearest_distances)

        
        # Get the label with the highest weighted frequency
        predicted_label = max(class_weights.keys(), key=(lambda key: class_weights[key]))
        predictions.append(predicted_label)

    return predictions

def classify_boards(fvectors_test:np.ndarray, model: dict) -> List[str]:
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    labels = []
    

    for i in range(0, len(fvectors_test), BOARD_SIZE):
        fvectors_test_board = fvectors_test[i:i + BOARD_SIZE]
        to_reeval = defaultdict(list)
        labels_board = []

        for pos, square in enumerate(fvectors_test_board):
            nearest_labels, nearest_distances = get_nearest_labels_and_distances(fvectors_train, labels_train, square)

            nearest_labels = remove_pawns(pos, nearest_labels)

            # Calculate the weighted count of each class
            class_weights = weigh_classes_from_distances(nearest_labels, nearest_distances)

            # Choose the label with the maximum weighted count
            predicted_label = "." if (not class_weights) else max(class_weights.keys(), key=(lambda key: class_weights[key]))
            labels_board.append(predicted_label)

        to_reeval = verify_piece_numbers(to_reeval, labels_board)

        for positions in to_reeval.values():
            if len(positions) == 0:
                continue
            curr_pieces_labels = []
            for pos in positions:
                nearest_labels, nearest_distances = get_nearest_labels_and_distances(fvectors_train, labels_train, fvectors_test_board[pos])

                # Calculate the weighted count of each class
                class_weights = weigh_classes_from_distances(nearest_labels, nearest_distances)

                nearest_labels = remove_pawns(pos, nearest_labels)

                if len(class_weights.keys()) >= 2:
                    maximum = max(class_weights.keys(), key=(lambda key: class_weights[key]))
                    for k, v in class_weights.items(): 
                        if k != maximum and abs(v - class_weights[maximum]) < 0.1:
                            class_weights[maximum] = 0
                            break
                        

                # Choose the label with the maximum weighted count
                predicted_label =  "." if (not class_weights) else max(class_weights.keys(), key=(lambda key: class_weights[key]))
                curr_pieces_labels.append(predicted_label)

            # Add the reevaluated labels to the board
            for i, label in enumerate(curr_pieces_labels):
                labels_board[positions[i]] = label

        labels.append(labels_board)

    labels = np.array(labels).reshape(1600)

    return labels.tolist()