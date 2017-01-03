import os
import errno
import datetime
import time
import csv
import sys
import socket
import timeit
import sys

from SVM import svmClassifier
from CalculateScore import Calculate_Metrics
from printPDF import printPDF
from printCSV import printCSV
from loadTestSet import loadTestSet
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

## Overall configurations
hostname = socket.gethostname()
testStart = datetime.datetime.now()
testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
path = testStartConvertet + '_' + hostname + '/'



try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

## Test Description
test_description = "SVM Grid search Test. From Date 08_05. With 25 pixels pr class. Using 6-folds cross validation"
# For NDVI MEAN and Random
# myArray=np.zeros((8, 7))
# C_range = 10. ** np.arange(-2, 4)
# gamma_range = 10. ** np.arange(-2, 5)

# Mearged og mean og random
myArray=np.zeros((10, 8))
C_range = 10. ** np.arange(-2, 8)
gamma_range = 10. ** np.arange(-12, -4)


for cost in range(C_range.shape[0]):
    for gamma in range(gamma_range.shape[0]):

        C = C_range[cost]
        gamma_value = gamma_range[gamma]
        print C
        print gamma_value
        x = [cost]
        ## Induvidual test configurations

        testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        csvfile_name = str(x) + '_' + testTimeConvertet + '.csv'
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

        print "SVM Test: " + str(x)  + "  Time: " + testTimeConvertet

        # Training Set Path setup
        traning_set = ''

        Size = '350'
        Type = ' 8$^{th}$ of May, 2016'

        Type2 = 'Mean Pixel'

        # Load training set
        test_set = loadTestSet(traning_set)
        X, Y = test_set.loadTestSet()

        # Set number of folds
        folds = 6
        kf = KFold(n_splits=folds)
        kf.get_n_splits(X)

        metrics_array = []
        fold = 0

        test_spec_dict = {'The value of Gamma': gamma_value, 'The value of Cost': C, 'Host': hostname,
                          'Date': testTimeConvertet, 'Sapmle Type': Type2, 'Samples Pr Class': Size, 'Classifier': 'SVM'}

        for train_index, test_index in kf.split(X):
            fold += 1
            start_time = timeit.default_timer()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Classifies fields with different k values from 1 to 50.
            y_pred = svmClassifier(X_train,y_train,X_test,C,gamma_value)

            # Calculate the different accuracy scores
            cm_metrics = Calculate_Metrics(y_test,y_pred)
            metrics_dict = cm_metrics.calculateScore()
            # Calculate the run time
            elapsed = (timeit.default_timer() - start_time) * 1000

            metrics_dict.update({'Run Time(MSec)': elapsed})

            # Create metrics for the fold
            #pdf_indi = printPDF(metrics_dict)
            #pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi, str(fold) + '_' + testTimeConvertet + '_' + hostname)

            metrics_array.append(metrics_dict)

        # Calculate the overall score for all the folds
        metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

        print "Acc " + str(float(metrics_mean_dict['accuracy_normalized']))
        print "STD " + str(float(metrics_mean_dict['accuracy_normalized_std']))

        myArray[cost][gamma] = float(metrics_mean_dict['accuracy_normalized'])*100
        confMat = myArray
        heatmap = plt.pcolormesh(confMat, vmin=30, vmax=100)

    #Create PDF
    #pdf = printPDF(metrics_mean_dict)
    #pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

    #Create CSV
    # csv = printCSV(metrics_mean_dict, test_spec_dict)
    # csv.createCSV(path, csvfile_name)

for y in range(confMat.shape[0]):
    for x in range(confMat.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.2f' % confMat[y, x],
                 horizontalalignment='center',
                 verticalalignment='center', )

heat= plt.colorbar(heatmap)
heat.set_ticks([30, 40,50,60,70,80,90, 100])
heat.set_label('%')
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.90)
plt.ylabel('C')
plt.xlabel('Gamma')
plt.title(Size + ' Training samples (Pr. crop).' '\n Mean merged training data set(Id:5) ')
my_xticks = ['1E-12','1E-11','1E-10','1E-09','1E-08','1E-07','1E-06','1E-05']
plt.xticks(np.arange(len(gamma_range))+0.5, my_xticks, rotation=45)
#plt.yticks(np.arange(len(C_range))+0.5, C_range, rotation=45)
my_yticks = ['0.01','0.1','1','10','100','1000','10000','100000','1000000','10000000']
plt.yticks(np.arange(len(C_range))+0.5, my_yticks, rotation=35)
plt.savefig(Size+'Samples'+Type+".pdf")
plt.show()

print "Done"