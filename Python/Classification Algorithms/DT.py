import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn import tree
import timeit

def dtClassifier(Depth, X_train, y_train, X_test):
    clf = tree.DecisionTreeClassifier(max_depth=Depth,criterion='entropy') # entropy gini
    #DecisionTreeRegressor(#max_depth=Depth)
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    elapsedTraining = (timeit.default_timer() - start_time) * 1000
    start_time = timeit.default_timer()

    predicted = clf.predict(X_test)
    elapsedTesting = (timeit.default_timer() - start_time) * 1000


    return elapsedTraining,elapsedTesting,predicted
