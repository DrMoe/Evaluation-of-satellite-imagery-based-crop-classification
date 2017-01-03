import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import timeit



def MLP_lbfgs_Classifier(hidden_layers, alpha, activation_func, X_train, Y_train, X_test):
    """
    Classifies using a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.
    Using the the solver Limited-memory BFGS.

    Args:
        hidden_layers (tuple): Hidden layer size.
        alpha (float): L2 penalty (regularization term) parameter.
        activation_func {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}: Activation function for the hidden layer.
                            ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
                            ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
                            ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
                            ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

        X_train (Array): Training n_samples,n_features.
        Y_train (Array): n_samples that holds the targets values (class labels).
        X_test (Array): Test/Validation samples used in prediction.
    Returns:
        predicted (Array): Predicted target values from the X_test samples.
    """

    clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=hidden_layers, random_state=1,
                          activation=activation_func)
    clf.fit(X_train,Y_train)
    predicted = clf.predict(X_test)
    return predicted

def MLP_sgd_Classifier(hidden_layers, alpha, activation_func, X_train, Y_train, X_test):
    """
    Classifies using a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.
    Using the the solver stochastic gradient descent.

    Args:
        hidden_layers (tuple): Hidden layer size.
        alpha (float): L2 penalty (regularization term) parameter.
        activation_func {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}: Activation function for the hidden layer.
                            ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
                            ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
                            ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
                            ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

        X_train (Array): Training n_samples,n_features.
        Y_train (Array): n_samples that holds the targets values (class labels).
        X_test (Array): Test/Validation samples used in prediction.
    Returns:
        predicted (Array): Predicted target values from the X_test samples.
    """

    clf = MLPClassifier(solver='sgd', alpha=alpha, hidden_layer_sizes=hidden_layers, random_state=1,
                          activation=activation_func)
    clf.fit(X_train,Y_train)
    predicted = clf.predict(X_test)
    return predicted

def MLP_adam_Classifier(hidden_neuruns, layers, alpha, learning_rate, activation_func, X_train, Y_train, X_test):
    """
    Classifies using a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.
    Using the the solver stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba.

    Args:
        hidden_layers (tuple): Hidden layer size.
        alpha (float): L2 penalty (regularization term) parameter.
        activation_func {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}: Activation function for the hidden layer.
                            ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
                            ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
                            ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
                            ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

        X_train (Array): Training n_samples,n_features.
        Y_train (Array): n_samples that holds the targets values (class labels).
        X_test (Array): Test/Validation samples used in prediction.
    Returns:
        predicted (Array): Predicted target values from the X_test samples.
    """

    hls = []

    for i in range(0,layers):
        hls.append(hidden_neuruns)

    clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=tuple(hls), random_state=1,
                          activation=activation_func, max_iter=5000, learning_rate_init=learning_rate)
    start_time = timeit.default_timer()
    clf.fit(X_train,Y_train)
    elapsedTraining = (timeit.default_timer() - start_time) * 1000
    start_time = timeit.default_timer()
    predicted = clf.predict(X_test)
    elapsedTesting = (timeit.default_timer() - start_time) * 1000

    return elapsedTraining,elapsedTesting,predicted