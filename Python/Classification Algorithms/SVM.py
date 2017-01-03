import os
from sklearn import svm

def svmClassifier( X_train, y_train, X_test,C,gamma):
    model = svm.SVC(C=C, class_weight=None,
        decision_function_shape='ovo', gamma=gamma, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    model.fit(X_train, y_train)
    model.predict(X_test)
    return model.predict(X_test)