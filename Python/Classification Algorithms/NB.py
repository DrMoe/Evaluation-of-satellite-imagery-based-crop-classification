import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
import timeit

def nbClassifier(X_train, y_train, X_test):

    clf = GaussianNB()
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    elapsedTraining = (timeit.default_timer() - start_time) * 1000
    start_time = timeit.default_timer()

    predicted_labels = clf.predict(X_test)
    elapsedTesting = (timeit.default_timer() - start_time) * 1000

    return elapsedTraining,elapsedTesting,predicted_labels