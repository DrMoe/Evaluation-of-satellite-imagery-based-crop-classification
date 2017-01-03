import os
import errno
import datetime
import time
import csv
import sys
import socket
import timeit
import sys

from NB import nbClassifier
from CalculateScore import Calculate_Metrics
from printPDF import printPDF
from printCSV import printCSV
from loadTestSet import loadTestSet
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score # Import section
import numpy as np
import matplotlib.pyplot as plt


# Test Data Setup
dates = ['08-05-16', '11-04-16', '15-09-16', '19-03-16']
datestitle = ['at 8$^{th}$ of May, 2016', 'at 11$^{th}$ of April, 2016', 'at 15$^{th}$ of Sep, 2016',
              'at 19$^{th}$ of March, 2016']
datefiles = ['0805', '1104', '1509', '1903', 'MergedSet']
filled_markers = ['|-', 'v-', '^-', '<-', '>-', '4-', 's-', 'p-', '*-', '1-', '+-', 'D-', '.-', 'o-']

## Random Pixel Values
for set in range(0, 4):
    y_results = []
    x = []
    for i in range(1, 15):

        samples = i * 30

        t_size = (samples/6)*5


        ## Overall configurations
        hostname = socket.gethostname()
        testStart = time.strftime("(%Y-%m-%d)")
        testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        path = str(t_size) + '_' + dates[set] + '_Random' + '_' + testStart + '_' + hostname + '/' # Overall configurations
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


        ## Test Description
        test_description = 'Naive Bayes Test. From Date' + dates[set] + '.With -- Random pixels pr class. Using 6-folds cross validation'

        ## Induvidual test configurations
        testTime = datetime.datetime.now()
        testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        csvfilename = time.strftime("(%Y-%m-%d)")
        csvfile_name = str(t_size) + '_' + csvfilename + '.csv'
        test_path = path + str(t_size) + '/'
        test_path_indi = path + str(t_size) + '/' + 'Folds/'
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

        print "Naive Bayes Test: " + str(t_size) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

        # Training Set Path setup
        t_path = '../Classify Set/Random Pixels/' + dates[set] + '/'
        training_file = str(samples) + 'SamplesPrClassBandset_Random_' + datefiles[set] + '_resampled.csv'
        traning_set = t_path + str(samples) + 'SamplesPrClassBandset_Random_' + datefiles[set] + '_resampled.csv'

        # Load training set
        test_set = loadTestSet(traning_set)
        X, Y = test_set.loadTestSet()

        # Set number of folds
        folds = 6
        kf = KFold(n_splits=folds)
        kf.get_n_splits(X)

        metrics_array = []
        fold = 0

        test_spec_dict = {'Classifier': 'Naive Bayes',
                          'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                          'Training Set': traning_set, 'Sample Type': 'Bandset random field single pixel', 'Path': t_path, 'File': training_file}

        for train_index, test_index in kf.split(X):
            fold += 1
            start_time = timeit.default_timer()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Naive Bayes
            y_pred = nbClassifier(X_train, y_train, X_test)

            # Calculate the different accuracy scores
            cm_metrics = Calculate_Metrics(y_test, y_pred)
            metrics_dict = cm_metrics.calculateScore()
            # Calculate the run time
            elapsed = (timeit.default_timer() - start_time) * 1000

            metrics_dict.update({'Run Time(MSec)': elapsed})

            # Create metrics for each fold
            # pdf_indi = printPDF(metrics_dict)
            # pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
            #                         str(fold) + '_' + testTimeConvertet + '_' + hostname)

            metrics_array.append(metrics_dict)

        # Calculate the overall score for all the folds
        metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

        # # Create PDF
        # pdf = printPDF(metrics_mean_dict)
        # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

        # Create CSV
        csv = printCSV(metrics_mean_dict, test_spec_dict)
        csv.createCSV(path, csvfile_name)

        # Get Scores
        y_results.append((metrics_mean_dict['accuracy_normalized'])*100)
        x.append(t_size)

    plt.plot(x, y_results, filled_markers[set], label=datefiles[set])

title = 'Bandset with random pixels from - \n ' + '19$^{th}$ of March' + ', ' + '11$^{th}$ of April' + ', ' + '8$^{th}$ of May' + ', ' + '15$^{th}$ of Sep, 2016'
figfile = 'CombiBandsetRandomPixel_Accuracy_AllDates.pdf'
#plt.xticks(np.arange(25,375,25))
plt.ylim([10, 95])
# plt.xlim([0, 16])
plt.title(title)
plt.ylabel('Accuracy(%)')
plt.xlabel('Training Size')
plt.grid(True)
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=7, prop={'size': 10})
plt.savefig(figfile, dpi=600, format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.clf()
print "Done"

# Mean Pixel Values
for set in range(0, 4):
    y_results = []
    x = []
    for i in range(1, 15):

        samples = i * 30

        t_size = (samples/6)*5


        ## Overall configurations
        hostname = socket.gethostname()
        testStart = time.strftime("(%Y-%m-%d)")
        testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        path = str(t_size) + '_' + dates[set] + '_Mean' + '_' + testStart + '_' + hostname + '/' # Overall configurations
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


        ## Test Description
        test_description = 'Naive Bayes Test. From Date' + dates[set] + '.With -- Mean pixels pr class. Using 6-folds cross validation'

        ## Induvidual test configurations
        testTime = datetime.datetime.now()
        testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        csvfilename = time.strftime("(%Y-%m-%d)")
        csvfile_name = str(t_size) + '_' + csvfilename + '.csv'
        test_path = path + str(t_size) + '/'
        test_path_indi = path + str(t_size) + '/' + 'Folds/'
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

        print "Naive Bayes Test: " + str(t_size) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

        # Training Set Path setup
        t_path = '../Classify Set/Mean Pixel Value/' + dates[set] + '/'
        training_file = str(samples) + 'SamplesPrClassBandset_Mean_' + datefiles[set] + '_resampled.csv'
        traning_set = t_path + str(samples) + 'SamplesPrClassBandset_Mean_' + datefiles[set] + '_resampled.csv'

        # Load training set
        test_set = loadTestSet(traning_set)
        X, Y = test_set.loadTestSet()

        # Set number of folds
        folds = 6
        kf = KFold(n_splits=folds)
        kf.get_n_splits(X)

        metrics_array = []
        fold = 0

        test_spec_dict = {'Classifier': 'Naive Bayes',
                          'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                          'Training Set': traning_set, 'Sample Type': 'Bandset Mean field single pixel', 'Path': t_path, 'File': training_file}

        for train_index, test_index in kf.split(X):
            fold += 1
            start_time = timeit.default_timer()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Naive Bayes
            y_pred = nbClassifier(X_train, y_train, X_test)

            # Calculate the different accuracy scores
            cm_metrics = Calculate_Metrics(y_test, y_pred)
            metrics_dict = cm_metrics.calculateScore()
            # Calculate the run time
            elapsed = (timeit.default_timer() - start_time) * 1000

            metrics_dict.update({'Run Time(MSec)': elapsed})

            # Create metrics for each fold
            # pdf_indi = printPDF(metrics_dict)
            # pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
            #                         str(fold) + '_' + testTimeConvertet + '_' + hostname)

            metrics_array.append(metrics_dict)

        # Calculate the overall score for all the folds
        metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

        # # Create PDF
        # pdf = printPDF(metrics_mean_dict)
        # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

        # Create CSV
        csv = printCSV(metrics_mean_dict, test_spec_dict)
        csv.createCSV(path, csvfile_name)

        # Get Scores
        y_results.append((metrics_mean_dict['accuracy_normalized'])*100)
        x.append(t_size)

    plt.plot(x, y_results, filled_markers[set], label=datefiles[set])

title = 'Bandset with mean pixels from - \n ' + '19$^{th}$ of March' + ', ' + '11$^{th}$ of April' + ', ' + '8$^{th}$ of May' + ', ' + '15$^{th}$ of Sep, 2016'
figfile = 'CombiBandsetMeanPixel_Accuracy_AllDates.pdf'
#plt.xticks(np.arange(25,375,25))
plt.ylim([10, 90])
# plt.xlim([0, 16])
plt.title(title)
plt.ylabel('Accuracy(%)')
plt.xlabel('Training Size')
plt.grid(True)
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=7, prop={'size': 10})
plt.savefig(figfile, dpi=600, format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.clf()
print "Done"

## Merged Mean Pixel Values
y_results = []
x = []
for i in range(1, 15):

    samples = i * 30

    t_size = (samples/6)*5

    elapsedTraining1 = 0
    elapsedTesting2 = 0
    ## Overall configurations
    hostname = socket.gethostname()
    testStart = time.strftime("(%Y-%m-%d)")
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
    path = str(t_size) + '_0805_1104_1509_1903' + '_Combi_Mean' + '_' + testStart + '_' + hostname + '/' # Overall configurations
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


    ## Test Description
    test_description = 'Naive Bayes Test. From Date.With -- Merged Mean pixels pr class. Using 6-folds cross validation'

    ## Induvidual test configurations
    testTime = datetime.datetime.now()
    testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
    csvfilename = time.strftime("(%Y-%m-%d)")
    csvfile_name = str(t_size) + '_' + csvfilename + '.csv'
    test_path = path + str(t_size) + '/'
    test_path_indi = path + str(t_size) + '/' + 'Folds/'
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

    print "Naive Bayes Test: " + str(t_size) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

    # Training Set Path setup
    t_path = '../Classify Set/Mean Pixel Value/Merged Sets/All/'
    training_file = str(samples) + 'SamplesPrClassRandom_0805_1104_1509_1903_Mean_resampled.csv'
    traning_set = t_path + str(samples) + 'SamplesPrClassRandom_0805_1104_1509_1903_Mean_resampled.csv'

    # Load training set
    test_set = loadTestSet(traning_set)
    X, Y = test_set.loadTestSet()

    # Set number of folds
    folds = 6
    kf = KFold(n_splits=folds)
    kf.get_n_splits(X)

    metrics_array = []
    fold = 0

    test_spec_dict = {'Classifier': 'Naive Bayes',
                      'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                      'Training Set': traning_set, 'Sample Type': 'Bandset Merged Mean field single pixel', 'Path': t_path, 'File': training_file}

    for train_index, test_index in kf.split(X):
        fold += 1
        start_time = timeit.default_timer()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # Naive Bayes
        elapsedTraining, elapsedTesting,y_pred = nbClassifier(X_train, y_train, X_test)
        elapsedTraining1 += elapsedTraining
        elapsedTesting2 += elapsedTesting
        # Calculate the different accuracy scores
        cm_metrics = Calculate_Metrics(y_test, y_pred)
        metrics_dict = cm_metrics.calculateScore()
        # Calculate the run time
        elapsed = (timeit.default_timer() - start_time) * 1000

        metrics_dict.update({'Run Time(MSec)': elapsed})

        # # Create metrics for each fold
        # pdf_indi = printPDF(metrics_dict)
        # pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
        #                         str(fold) + '_' + testTimeConvertet + '_' + hostname)

        metrics_array.append(metrics_dict)

    # Calculate the overall score for all the folds
    metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)
    print 'Time in ms for training: ' + str(elapsedTraining1 / 6)
    print 'Time in ms for testing: ' + str(elapsedTesting2 / 6)
    # # # Create PDF
    # pdf = printPDF(metrics_mean_dict)
    # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

    # Create CSV
    csv = printCSV(metrics_mean_dict, test_spec_dict)
    csv.createCSV(path, csvfile_name)

    # Get Scores
    y_results.append((metrics_mean_dict['accuracy_normalized'])*100)
    x.append(t_size)

plt.plot(x, y_results, filled_markers[set], label='Mean merged')

print "Done"


# Merged Random Pixel Values
y_results = []
x = []
for i in range(1, 15):

    samples = i * 30

    t_size = (samples/6)*5


    ## Overall configurations
    hostname = socket.gethostname()
    testStart = time.strftime("(%Y-%m-%d)")
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
    path = str(t_size) + '_0805_1104_1509_1903' + '_Combi_Random' + '_' + testStart + '_' + hostname + '/' # Overall configurations
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


    ## Test Description
    #test_description = 'Naive Bayes Test. From Date' + dates[set] + '.With -- Merged Mean pixels pr class. Using 6-folds cross validation'

    ## Induvidual test configurations
    testTime = datetime.datetime.now()
    testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
    csvfilename = time.strftime("(%Y-%m-%d)")
    csvfile_name = str(t_size) + '_' + csvfilename + '.csv'
    test_path = path + str(t_size) + '/'
    test_path_indi = path + str(t_size) + '/' + 'Folds/'
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

    print "Naive Bayes Test: " + str(t_size) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

    # Training Set Path setup
    t_path = '../Classify Set/Random Pixels/Merged Sets/0805_1104_1509_1903/'
    training_file = str(samples) + 'SamplesPrClassBandset_Random_0805_1104_1509_1903_resampled.csv'
    traning_set = t_path + str(samples) + 'SamplesPrClassBandset_Random_0805_1104_1509_1903_resampled.csv'

    # Load training set
    test_set = loadTestSet(traning_set)
    X, Y = test_set.loadTestSet()

    # Set number of folds
    folds = 6
    kf = KFold(n_splits=folds)
    kf.get_n_splits(X)

    metrics_array = []
    fold = 0

    test_spec_dict = {'Classifier': 'Naive Bayes',
                      'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                      'Training Set': traning_set, 'Sample Type': 'Bandset Merged Random field single pixel', 'Path': t_path, 'File': training_file}

    for train_index, test_index in kf.split(X):
        fold += 1
        start_time = timeit.default_timer()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # Naive Bayes
        y_pred = nbClassifier(X_train, y_train, X_test)

        # Calculate the different accuracy scores
        cm_metrics = Calculate_Metrics(y_test, y_pred)
        metrics_dict = cm_metrics.calculateScore()
        # Calculate the run time
        elapsed = (timeit.default_timer() - start_time) * 1000

        metrics_dict.update({'Run Time(MSec)': elapsed})

        # Create metrics for each fold
        # pdf_indi = printPDF(metrics_dict)
        # pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
        #                         str(fold) + '_' + testTimeConvertet + '_' + hostname)

        metrics_array.append(metrics_dict)

    # Calculate the overall score for all the folds
    metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

    # # Create PDF
    # pdf = printPDF(metrics_mean_dict)
    # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

    # Create CSV
    csv = printCSV(metrics_mean_dict, test_spec_dict)
    csv.createCSV(path, csvfile_name)

    # Get Scores
    y_results.append((metrics_mean_dict['accuracy_normalized'])*100)
    x.append(t_size)

plt.plot(x, y_results, filled_markers[set], label='Random merged')
title = 'Mean merged and random merged data sets (Id:5) and (Id:10) at - \n ' + '19$^{th}$ of March' + ', ' + '11$^{th}$ of April' + ', ' + '8$^{th}$ of May' + ', ' + '15$^{th}$ of Sep, 2016'
figfile = 'MergedBandsetMeanRandomPixel_Accuracy_AllDates.pdf'
#plt.xticks(np.arange(25,375,25))
plt.ylim([10, 90])
# plt.xlim([0, 16])
plt.title(title)
plt.ylabel('Accuracy(%)')
plt.xlabel('Training Size')
plt.grid(True)
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=7, prop={'size': 10})
plt.savefig(figfile, dpi=600, format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.clf()
print "Done"

# NDVI Random Pixel Values
for set in range(0, 4):
    y_results = []
    x = []
    for i in range(1, 15):

        samples = i * 30

        t_size = (samples/6)*5


        ## Overall configurations
        hostname = socket.gethostname()
        testStart = time.strftime("(%Y-%m-%d)")
        testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        path = str(t_size) + '_' + dates[set] + '_NDVIRandom' + '_' + testStart + '_' + hostname + '/' # Overall configurations
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


        ## Test Description
        test_description = 'Naive Bayes Test. From Date' + dates[set] + '.With -- Random pixels pr class. Using 6-folds cross validation'

        ## Induvidual test configurations
        testTime = datetime.datetime.now()
        testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        csvfilename = time.strftime("(%Y-%m-%d)")
        csvfile_name = str(t_size) + '_' + csvfilename + '.csv'
        test_path = path + str(t_size) + '/'
        test_path_indi = path + str(t_size) + '/' + 'Folds/'
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

        print "Naive Bayes Test: " + str(t_size) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

        # Training Set Path setup
        t_path = '../Classify Set/NDVI/' + dates[set] + '/Single Random/'
        training_file = str(samples) + 'SamplesPrClassNDVI_Random_' + datefiles[set] + '_resampled.csv'
        traning_set = t_path + str(samples) + 'SamplesPrClassNDVI_Random_' + datefiles[set] + '_resampled.csv'

        # Load training set
        test_set = loadTestSet(traning_set)
        X, Y = test_set.loadTestSet()

        # Set number of folds
        folds = 6
        kf = KFold(n_splits=folds)
        kf.get_n_splits(X)

        metrics_array = []
        fold = 0

        test_spec_dict = {'Classifier': 'Naive Bayes',
                          'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                          'Training Set': traning_set, 'Sample Type': 'Bandset random field single pixel', 'Path': t_path, 'File': training_file}

        for train_index, test_index in kf.split(X):
            fold += 1
            start_time = timeit.default_timer()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Naive Bayes
            y_pred = nbClassifier(X_train, y_train, X_test)

            # Calculate the different accuracy scores
            cm_metrics = Calculate_Metrics(y_test, y_pred)
            metrics_dict = cm_metrics.calculateScore()
            # Calculate the run time
            elapsed = (timeit.default_timer() - start_time) * 1000

            metrics_dict.update({'Run Time(MSec)': elapsed})

            # Create metrics for each fold
            # pdf_indi = printPDF(metrics_dict)
            # pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
            #                         str(fold) + '_' + testTimeConvertet + '_' + hostname)

            metrics_array.append(metrics_dict)

        # Calculate the overall score for all the folds
        metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

        # # Create PDF
        # pdf = printPDF(metrics_mean_dict)
        # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

        # Create CSV
        csv = printCSV(metrics_mean_dict, test_spec_dict)
        csv.createCSV(path, csvfile_name)

        # Get Scores
        y_results.append((metrics_mean_dict['accuracy_normalized'])*100)
        x.append(t_size)

    plt.plot(x, y_results, filled_markers[set], label=datefiles[set])

title = 'NDVI with random pixels from - \n ' + '19$^{th}$ of March' + ', ' + '11$^{th}$ of April' + ', ' + '8$^{th}$ of May' + ', ' + '15$^{th}$ of Sep, 2016'
figfile = 'CombiBandsetNDVIRandomPixel_Accuracy_AllDates.pdf'
#plt.xticks(np.arange(25,375,25))
plt.ylim([10, 90])
# plt.xlim([0, 16])
plt.title(title)
plt.ylabel('Accuracy(%)')
plt.xlabel('Training Size')
plt.grid(True)
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=7, prop={'size': 10})
plt.savefig(figfile, dpi=600, format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.clf()
print "Done"


## NDVI Mean Pixel Values
for set in range(0, 4):
    y_results = []
    x = []
    for i in range(1, 15):

        samples = i * 30

        t_size = (samples/6)*5


        ## Overall configurations
        hostname = socket.gethostname()
        testStart = time.strftime("(%Y-%m-%d)")
        testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        path = str(t_size) + '_' + dates[set] + '_NDVIMean' + '_' + testStart + '_' + hostname + '/' # Overall configurations
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


        ## Test Description
        test_description = 'Naive Bayes Test. From Date' + dates[set] + '.With -- Random pixels pr class. Using 6-folds cross validation'

        ## Induvidual test configurations
        testTime = datetime.datetime.now()
        testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        csvfilename = time.strftime("(%Y-%m-%d)")
        csvfile_name = str(t_size) + '_' + csvfilename + '.csv'
        test_path = path + str(t_size) + '/'
        test_path_indi = path + str(t_size) + '/' + 'Folds/'
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

        print "Naive Bayes Test: " + str(t_size) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

        # Training Set Path setup
        t_path = '../Classify Set/NDVI/' + dates[set] + '/Mean/'
        training_file = str(samples) + 'SamplesPrClassNDVI_' + datefiles[set] + '_Mean_resampled.csv'
        traning_set = t_path + str(samples) + 'SamplesPrClassNDVI_' + datefiles[set] + '_Mean_resampled.csv'

        # Load training set
        test_set = loadTestSet(traning_set)
        X, Y = test_set.loadTestSet()

        # Set number of folds
        folds = 6
        kf = KFold(n_splits=folds)
        kf.get_n_splits(X)

        metrics_array = []
        fold = 0

        test_spec_dict = {'Classifier': 'Naive Bayes',
                          'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                          'Training Set': traning_set, 'Sample Type': 'Bandset random field single pixel', 'Path': t_path, 'File': training_file}

        for train_index, test_index in kf.split(X):
            fold += 1
            start_time = timeit.default_timer()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Naive Bayes
            y_pred = nbClassifier(X_train, y_train, X_test)

            # Calculate the different accuracy scores
            cm_metrics = Calculate_Metrics(y_test, y_pred)
            metrics_dict = cm_metrics.calculateScore()
            # Calculate the run time
            elapsed = (timeit.default_timer() - start_time) * 1000

            metrics_dict.update({'Run Time(MSec)': elapsed})

            # Create metrics for each fold
            # pdf_indi = printPDF(metrics_dict)
            # pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
            #                         str(fold) + '_' + testTimeConvertet + '_' + hostname)

            metrics_array.append(metrics_dict)

        # Calculate the overall score for all the folds
        metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

        # # Create PDF
        # pdf = printPDF(metrics_mean_dict)
        # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

        # Create CSV
        csv = printCSV(metrics_mean_dict, test_spec_dict)
        csv.createCSV(path, csvfile_name)

        # Get Scores
        y_results.append((metrics_mean_dict['accuracy_normalized'])*100)
        x.append(t_size)

    plt.plot(x, y_results, filled_markers[set], label=datefiles[set])

title = 'NDVI with mean pixels from - \n ' + '19$^{th}$ of March' + ', ' + '11$^{th}$ of April' + ', ' + '8$^{th}$ of May' + ', ' + '15$^{th}$ of Sep, 2016'
figfile = 'CombiBandsetNDVIMeanPixel_Accuracy_AllDates.pdf'
#plt.xticks(np.arange(25,375,25))
plt.ylim([10, 90])
# plt.xlim([0, 16])
plt.title(title)
plt.ylabel('Accuracy(%)')
plt.xlabel('Training Size')
plt.grid(True)
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=7, prop={'size': 10})
plt.savefig(figfile, dpi=600, format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.clf()
print "Done"

## Merged NDVI Mean Pixel Values
y_results = []
x = []
for i in range(1, 15):

    samples = i * 30

    t_size = (samples/6)*5


    ## Overall configurations
    hostname = socket.gethostname()
    testStart = time.strftime("(%Y-%m-%d)")
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
    path = str(t_size) + '_0805_1104_1509_1903' + '_Combi_NDVIMean' + '_' + testStart + '_' + hostname + '/' # Overall configurations
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


    ## Test Description
    test_description = 'Naive Bayes Test. From Date' + dates[set] + '.With -- Merged Mean pixels pr class. Using 6-folds cross validation'

    ## Induvidual test configurations
    testTime = datetime.datetime.now()
    testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
    csvfilename = time.strftime("(%Y-%m-%d)")
    csvfile_name = str(t_size) + '_' + csvfilename + '.csv'
    test_path = path + str(t_size) + '/'
    test_path_indi = path + str(t_size) + '/' + 'Folds/'
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

    print "Naive Bayes Test: " + str(t_size) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

    # Training Set Path setup
    t_path = '../Classify Set/NDVI/Merged Sets/All/Mean/'
    training_file = str(samples) + 'SamplesPrClassNDVI_Random_0805_1104_1509_1903_Mean_resampled.csv'
    traning_set = t_path + str(samples) + 'SamplesPrClassNDVI_Random_0805_1104_1509_1903_Mean_resampled.csv'

    # Load training set
    test_set = loadTestSet(traning_set)
    X, Y = test_set.loadTestSet()

    # Set number of folds
    folds = 6
    kf = KFold(n_splits=folds)
    kf.get_n_splits(X)

    metrics_array = []
    fold = 0

    test_spec_dict = {'Classifier': 'Naive Bayes',
                      'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                      'Training Set': traning_set, 'Sample Type': 'Bandset Merged Mean field single pixel', 'Path': t_path, 'File': training_file}

    for train_index, test_index in kf.split(X):
        fold += 1
        start_time = timeit.default_timer()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # Naive Bayes
        y_pred = nbClassifier(X_train, y_train, X_test)

        # Calculate the different accuracy scores
        cm_metrics = Calculate_Metrics(y_test, y_pred)
        metrics_dict = cm_metrics.calculateScore()
        # Calculate the run time
        elapsed = (timeit.default_timer() - start_time) * 1000

        metrics_dict.update({'Run Time(MSec)': elapsed})

        # Create metrics for each fold
        # pdf_indi = printPDF(metrics_dict)
        # pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
        #                         str(fold) + '_' + testTimeConvertet + '_' + hostname)

        metrics_array.append(metrics_dict)

    # Calculate the overall score for all the folds
    metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

    # # Create PDF
    # pdf = printPDF(metrics_mean_dict)
    # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

    # Create CSV
    csv = printCSV(metrics_mean_dict, test_spec_dict)
    csv.createCSV(path, csvfile_name)

    # Get Scores
    y_results.append((metrics_mean_dict['accuracy_normalized'])*100)
    x.append(t_size)

plt.plot(x, y_results, filled_markers[set], label='Mean')

print "Done"


## Merged NDVI Random Pixel Values
y_results = []
x = []
for i in range(1, 15):

    samples = i * 30

    t_size = (samples/6)*5


    ## Overall configurations
    hostname = socket.gethostname()
    testStart = time.strftime("(%Y-%m-%d)")
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
    path = str(t_size) + '_0805_1104_1509_1903' + '_Combi_NDVI_Random' + '_' + testStart + '_' + hostname + '/' # Overall configurations
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


    ## Test Description
    test_description = 'Naive Bayes Test. From Date' + dates[set] + '.With -- Merged Mean pixels pr class. Using 6-folds cross validation'

    ## Induvidual test configurations
    testTime = datetime.datetime.now()
    testTimeConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
    csvfilename = time.strftime("(%Y-%m-%d)")
    csvfile_name = str(t_size) + '_' + csvfilename + '.csv'
    test_path = path + str(t_size) + '/'
    test_path_indi = path + str(t_size) + '/' + 'Folds/'
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

    print "Naive Bayes Test: " + str(t_size) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

    # Training Set Path setup
    t_path = '../Classify Set/NDVI/Merged Sets/All/Single Random/'
    training_file = str(samples) + 'SamplesPrClassNDVI_Random_0805_1104_1509_1903__resampled.csv'
    traning_set = t_path + str(samples) + 'SamplesPrClassNDVI_Random_0805_1104_1509_1903__resampled.csv'

    # Load training set
    test_set = loadTestSet(traning_set)
    X, Y = test_set.loadTestSet()

    # Set number of folds
    folds = 6
    kf = KFold(n_splits=folds)
    kf.get_n_splits(X)

    metrics_array = []
    fold = 0

    test_spec_dict = {'Classifier': 'Naive Bayes',
                      'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                      'Training Set': traning_set, 'Sample Type': 'Bandset Merged Random field single pixel', 'Path': t_path, 'File': training_file}

    for train_index, test_index in kf.split(X):
        fold += 1
        start_time = timeit.default_timer()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # Naive Bayes
        y_pred = nbClassifier(X_train, y_train, X_test)

        # Calculate the different accuracy scores
        cm_metrics = Calculate_Metrics(y_test, y_pred)
        metrics_dict = cm_metrics.calculateScore()
        # Calculate the run time
        elapsed = (timeit.default_timer() - start_time) * 1000

        metrics_dict.update({'Run Time(MSec)': elapsed})

        # Create metrics for each fold
        # pdf_indi = printPDF(metrics_dict)
        # pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
        #                         str(fold) + '_' + testTimeConvertet + '_' + hostname)

        metrics_array.append(metrics_dict)

    # Calculate the overall score for all the folds
    metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

    # # Create PDF
    # pdf = printPDF(metrics_mean_dict)
    # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

    # Create CSV
    csv = printCSV(metrics_mean_dict, test_spec_dict)
    csv.createCSV(path, csvfile_name)

    # Get Scores
    y_results.append((metrics_mean_dict['accuracy_normalized'])*100)
    x.append(t_size)

plt.plot(x, y_results, filled_markers[set], label='Random')

title = 'Merged NDVI with mean and random pixels at - \n ' + '19$^{th}$ of March' + ', ' + '11$^{th}$ of April' + ', ' + '8$^{th}$ of May' + ', ' + '15$^{th}$ of Sep, 2016'
figfile = 'MergedNDVIMeanRandomPixel_Accuracy_AllDates.pdf'
#plt.xticks(np.arange(25,375,25))
plt.ylim([10, 90])
# plt.xlim([0, 16])
plt.title(title)
plt.ylabel('Accuracy(%)')
plt.xlabel('Training Size')
plt.grid(True)
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=7, prop={'size': 10})
plt.savefig(figfile, dpi=600, format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.clf()
print "Done"