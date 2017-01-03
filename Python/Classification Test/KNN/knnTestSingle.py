import os
import errno
import datetime
import time
import csv
import socket
import timeit
import sys

from KNN import knnClassifier
from CalculateScore import Calculate_Metrics
from printPDF import printPDF
from printCSV import printCSV
from loadTestSet import loadTestSet
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Test Data Setup
dates = ['08-05-16', '11-04-16', '15-09-16', '19-03-16']
datefiles = ['0805','1104','1509','1903', 'MergedSet']

t_size = 350

it = (t_size/5)

samples = t_size + it

x  = 8

## Overall configurations
hostname = socket.gethostname()
testStart = time.strftime("(%Y-%m-%d)")
testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
path = str(t_size) + '_' + '_Random' + '_' + testStart + '_' + hostname + '/' # Overall configurations
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise


## Test Description
test_description = ''

## Induvidual test configurations
testTime = datetime.datetime.now()
testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
csvfilename = time.strftime("(%Y-%m-%d)")
csvfile_name = str(x) + '_' + csvfilename + '.csv'
test_path = path + str(x) + '/'
test_path_indi = path + str(x) + '/' + 'Folds/'
try:
    os.makedirs(test_path)
except OSError:
    if not os.path.isdir(test_path):
        raise
try:
    os.makedirs(test_path_indi)
except OSError:
    if not os.path.isdir(test_path_indi):
        raise

# Training Set Path setup
t_path = ''

# Load training set
test_set = loadTestSet(t_path)
X, Y = test_set.loadTestSet()

# Set number of folds
folds = 6
kf = KFold(n_splits=folds)
kf.get_n_splits(X)

metrics_array = []
fold = 0

test_spec_dict = {'Classifier': 'KNN',
                  'Number of Neighbors': x, 'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                  'Sample Type': 'Bandset random field single pixel', 'Path': t_path,}

for train_index, test_index in kf.split(X):
    fold += 1
    start_time = timeit.default_timer()

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]


    # Classifies fields with different k values from 1 to 50.
    elapsedTraining, elapsedTesting, y_pred = knnClassifier(x, X_train, y_train, X_test)

    # Calculate the run time
    elapsed = (timeit.default_timer() - start_time) * 1000


    # Calculate the different accuracy scores
    cm_metrics = Calculate_Metrics(y_test, y_pred)
    metrics_dict = cm_metrics.calculateScore()


    metrics_dict.update({'Run Time(MSec)': elapsed})

    #Create metrics for each fold
    pdf_indi = printPDF(metrics_dict)
    pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
                            str(fold) + '_' + testTimeConvertet + '_' + hostname)

    metrics_array.append(metrics_dict)

# Calculate the overall score for all the folds
metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

# Create PDF
pdf = printPDF(metrics_mean_dict)
pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

# Create CSV
csv = printCSV(metrics_mean_dict, test_spec_dict)
csv.createCSV(path, csvfile_name)

print "Done"