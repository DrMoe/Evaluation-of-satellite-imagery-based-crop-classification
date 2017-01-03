import os
import errno
import datetime
import time
import csv
import sys
import socket
import timeit
import sys
import re
import matplotlib.pyplot as plt
import numpy as np

from DT import dtClassifier
from CalculateScore import Calculate_Metrics
from printPDF import printPDF
from printCSV import printCSV
from loadTestSet import loadTestSet
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

class runDT_Test:

    def __init__(self):
        print ""

    def runTest(self,path,treeSize,trainingFile,test_description,type,traning_size,create_pdf=False):

        ## Overall configurations
        hostname = socket.gethostname()
        #testStart = datetime.datetime.now()
        #testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        # path = testStartConvertet + '_' + hostname + '/'

        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise

        ## Induvidual test configurations
        testTime = datetime.datetime.now()
        testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        csvfile_name = str(treeSize) + '_' + testTimeConvertet + '_' + str(traning_size) + '.csv'
        test_path = path + str(treeSize) + '/'
        test_path_indi = path + str(treeSize) + '/' + 'Folds/'

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

        print "DT Test: " + str(treeSize) + "  Time: " + testTimeConvertet

        # Training Set Path setup
        traning_set = trainingFile
        # Load training set
        test_set = loadTestSet(traning_set)
        X, Y = test_set.loadTestSet()
        # Set number of folds
        folds = 6
        kf = KFold(n_splits=folds)
        kf.get_n_splits(X)

        metrics_array = []
        fold = 0

        test_spec_dict = {'Classifier': 'DT',
                          'Depth': treeSize, 'Date': testTimeConvertet, 'Host': hostname,
                          'Training Set': traning_set, 'Sample Type': type}

        elapsedTraining1 = 0
        elapsedTesting2 = 0

        for train_index, test_index in kf.split(X):
            fold += 1
            start_time = timeit.default_timer()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Classifies fields with different k values from 1 to 50.
            elapsedTraining, elapsedTesting,y_pred = dtClassifier(treeSize, X_train, y_train, X_test)
            elapsedTraining1 += elapsedTraining
            elapsedTesting2 += elapsedTesting

            # Calculate the different accuracy scores
            cm_metrics = Calculate_Metrics(y_test, y_pred)
            metrics_dict = cm_metrics.calculateScore()
            # Calculate the run time
            elapsed = (timeit.default_timer() - start_time) * 1000


            metrics_dict.update({'Run Time(MSec)': elapsed})

            if create_pdf:
                # Create metrics for the fold
                pdf_indi = printPDF(metrics_dict)
                pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
                                         str(fold) + '_' + testTimeConvertet + '_' + hostname)

            metrics_array.append(metrics_dict)

        # Calculate the overall score for all the folds
        metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

        print 'Time in ms for training: ' + str(elapsedTraining1 / 6)
        print 'Time in ms for testing: ' + str(elapsedTesting2 / 6)

        if create_pdf:
            # Create PDF
            pdf = printPDF(metrics_mean_dict)
            pdf.create_pdf(test_spec_dict, test_description, test_path, str(treeSize) + '_' + testTimeConvertet + '_' + hostname)

        # Create CSV
        csv = printCSV(metrics_mean_dict, test_spec_dict)
        csv.createCSV(path, csvfile_name)

        return csvfile_name



    def combineResults(self,path,traning_size):

        data = []
        i = 0

        numbers = re.compile(r'(\d+)')

        def numericalSort(value):
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts

        for file in sorted(os.listdir(path + '/Overall/'), key=numericalSort):
            if file.endswith(str(traning_size) + '.csv'):
                with open(path + '/Overall/' + file, 'rb') as csvfile:
                    print "Current File Being Processed is: " + file
                    fileReader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    fileReader.next()

                    for row in fileReader:
                        i += 1
                        row = [i] + row
                        data.append(row)

        file = path + str(traning_size) + '_Combined_Results.csv'

        with open(file, "w") as text_file:
            writer = csv.writer(text_file)
            writer.writerow(
                ['Nr', 'Accuracy', 'Kappa', 'F1 Score', 'Run Time (MSec)', 'Accuracy Std', 'Kappa Std', 'F1 Std'])
            writer.writerows(data)

    def plot(self,path,title,figfile):

        dates = ['08-05-16', '11-04-16', '15-09-16', '19-03-16']
        datestitle = ['at 8$^{th}$ of May, 2016', 'at 11$^{th}$ of April, 2016', 'at 15$^{th}$ of Sep, 2016',
                      'at 19$^{th}$ of March, 2016']
        datefiles = ['0805', '1104', '1509', '1903', 'MergedSet']
        filled_markers = ['|-', 'v-', '^-', '<-', '>-', '4-', 's-', 'p-', '*-', '1-', '+-', 'D-', '.-', 'o-']

        for i in range(1,15):

            training_nr = 25*i
            csvfilename = str(path) + str(training_nr) + '_Combined_Results.csv'

            with open(csvfilename, 'rb') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                csvreader.next()  # Skip the first line as it contains text
                y_results = []
                x = []
                for row in csvreader:
                    row_value = float(row[1])*100
                    y_results.append(row_value)
                    x.append(float(row[0]))

                plt.plot(x[0:20], y_results[0:20], filled_markers[i-1],label=str(training_nr))

        plt.xticks((np.arange(len(x))+1)[1::2])
        plt.ylim([10,90])
        plt.xlim([0, 20.5])
        plt.title(title)
        plt.ylabel('Accuracy(%)')
        plt.xlabel('Maximum depth size')
        plt.grid(True)
        lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=7, prop={'size': 10})
        plt.savefig(figfile, dpi=600, format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()

