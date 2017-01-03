import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import timeit


def knnClassifier(neighbors, X_train_samples, Y_train_labels, X_test_samples):
    """
    Trains a k-nearest neighbors classifier and predict the classes based on X_test_samples samples

    Args:
        neighbors (int): The number of neighbors.
        X_train_samples (Array): Training samples.
        Y_train_labels (Array): Training class values (Class labels).
        X_test_samples (Array): Test/Validation samples, used to predict class labels.
    Returns:
        predicted_labels (Array): Predicted class values (Class labels)
    """

    #X_train_samples = np.array(X_train_samples)
    # Y_train_labels = np.array([0,2])
    # X_test_samples = np.array([[[2],[3]],[[1],[2]]])

    # nsamples, nx, ny = X_train_samples.shape
    # d2_train_dataset = X_train_samples.reshape((nsamples, nx * ny))
    #
    # nsamples, nx, ny = X_test_samples.shape
    # d2_test_dataset = X_test_samples.reshape((nsamples, nx * ny))

    #train = np.array(X_train_samples, dtype=dtype, order=order, copy=copy)

    knn_classifier = KNeighborsClassifier(n_neighbors=neighbors)

    start_time = timeit.default_timer()
    knn_classifier.fit(X_train_samples, Y_train_labels)
    elapsedTraining = (timeit.default_timer() - start_time) * 1000
    start_time = timeit.default_timer()
    predicted_labels = knn_classifier.predict(X_test_samples)
    elapsedTesting = (timeit.default_timer() - start_time) * 1000

    return elapsedTraining,elapsedTesting,predicted_labels