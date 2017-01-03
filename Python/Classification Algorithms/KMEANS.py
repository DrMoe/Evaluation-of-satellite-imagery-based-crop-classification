import os
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix


def kmeansClassifier(X_train, X_test):
    clf = KMeans(n_clusters=5, random_state=0,init='k-means++', n_init=10, max_iter=3000)
    clf.fit(X_train)
    predicted_labels = clf.predict(X_test)
    return predicted_labels
