import os
import errno
import datetime
import time
import csv
import sys
import socket
import timeit
import sys


from DT import dtClassifier
from CalculateScore import Calculate_Metrics
from printPDF import printPDF
from printCSV import printCSV
from loadTestSet import loadTestSet
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#from IPython.display import Image
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn import tree
import pydotplus
from sklearn.datasets import load_iris
from IPython.display import Image


# Training Set Path setup
traning_set = ''

# Load training set
test_set = loadTestSet(traning_set)
X, Y = test_set.loadTestSet()


clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=3)
clf.fit(X, Y)

dot_data1 = tree.export_graphviz(clf, out_file=None)
graph1 = pydotplus.graph_from_dot_data(dot_data1)
graph1.write_pdf("iris.pdf")

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=['Band 6', 'Band 7', 'Band 8', 'Band 8A', 'Band 11', 'Band 12'],
                         class_names=['Spring Barly', 'Winter Barly', 'Winter Wheat', 'Winter Rape', 'Maize'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("irisv2Gini.pdf")
Image(graph.create_png())

