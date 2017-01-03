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
from sklearn.model_selection import cross_val_score # Import section

# Test Data Setup
dates = ['08-05-16', '11-04-16', '15-09-16', '19-03-16']
datefiles = ['0805','1104','1509','1903', 'MergedSet']

## Random Pixel Values
for set in range(0, 4):
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
        test_description = 'KNN Test with variable k-size, from 1 to 40. From Date ' + dates[set] + ' .With ' + str(t_size) + ' Bandset random pixels pr class. Using 6-folds cross validation'

        for x in range(1, 41):
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

            print "KNN Test: " + str(x) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

            # Training Set Path setup
            t_path = '../../Classify Set/Random Pixels/' + dates[set] + '/'
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

            test_spec_dict = {'Classifier': 'KNN',
                              'Number of Neighbors': x, 'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                              'Training Set': traning_set, 'Sample Type': 'Bandset random field single pixel', 'Path': t_path, 'File': training_file}

            for train_index, test_index in kf.split(X):
                fold += 1
                start_time = timeit.default_timer()

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]


                # Classifies fields with different k values from 1 to 50.
                y_pred = knnClassifier(x, X_train, y_train, X_test)

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


# Mean Pixel Values
for set in range(0, 4):
    for i in range(14, 15):

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
        test_description = 'KNN Test with variable k-size, from 1 to 60. From Date ' + dates[set] + ' .With ' + str(t_size) + ' Bandset mean pixels pr class. Using 6-folds cross validation'

        for x in range(1, 21):
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

            print "KNN Test: " + str(x) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

            # Training Set Path setup
            t_path = '../../Classify Set/Mean Pixel Value/' + dates[set] + '/'
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

            test_spec_dict = {'Classifier': 'KNN',
                              'Number of Neighbors': x, 'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                              'Training Set': traning_set, 'Sample Type': 'Bandset mean field single pixel', 'Path': t_path, 'File': training_file}

            for train_index, test_index in kf.split(X):
                fold += 1
                start_time = timeit.default_timer()

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                # Classifies fields with different k values from 1 to 50.
                y_pred = knnClassifier(x, X_train, y_train, X_test)

                # Calculate the different accuracy scores
                cm_metrics = Calculate_Metrics(y_test, y_pred)
                metrics_dict = cm_metrics.calculateScore()
                # Calculate the run time
                elapsed = (timeit.default_timer() - start_time) * 1000

                metrics_dict.update({'Run Time(MSec)': elapsed})

                # Create metrics for each fold
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

## NDVI Random Pixel Values
for set in range(0, 4):
    for i in range(14, 15):

        samples = i * 30

        t_size = (samples/6)*5


        ## Overall configurations
        hostname = socket.gethostname()
        testStart = time.strftime("(%Y-%m-%d)")
        testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        path = str(t_size) + '_' + dates[set] + '_NDVI_Random' + '_' + testStart + '_' + hostname + '/' # Overall configurations
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


        ## Test Description
        test_description = 'KNN Test with variable k-size, from 1 to 60. From Date ' + dates[set] + ' .With ' + str(t_size) + ' NDVI random pixels pr class. Using 6-folds cross validation'

        for x in range(1, 41):
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

            print "KNN Test: " + str(x) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

            # Training Set Path setup
            t_path = '../../Classify Set/NDVI/' + dates[set] + '/Single Random/'
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

            test_spec_dict = {'Classifier': 'KNN',
                              'Number of Neighbors': x, 'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                              'Training Set': traning_set, 'Sample Type': 'NDVI random field single pixel', 'Path': t_path, 'File': training_file}

            for train_index, test_index in kf.split(X):
                fold += 1
                start_time = timeit.default_timer()

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                # Classifies fields with different k values from 1 to 50.
                y_pred = knnClassifier(x, X_train, y_train, X_test)

                # Calculate the different accuracy scores
                cm_metrics = Calculate_Metrics(y_test, y_pred)
                metrics_dict = cm_metrics.calculateScore()
                # Calculate the run time
                elapsed = (timeit.default_timer() - start_time) * 1000

                metrics_dict.update({'Run Time(MSec)': elapsed})

                # Create metrics for each fold
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

## NDVI Mean Pixel Values
for set in range(0, 4):
    for i in range(14, 15):

        samples = i * 30

        t_size = (samples/6)*5


        ## Overall configurations
        hostname = socket.gethostname()
        testStart = time.strftime("(%Y-%m-%d)")
        testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
        path = str(t_size) + '_' + dates[set] + '_NDVI_Mean' + '_' + testStart + '_' + hostname + '/' # Overall configurations
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise


        ## Test Description
        test_description = 'KNN Test with variable k-size, from 1 to 60. From Date ' + dates[set] + ' .With ' + str(t_size) + ' NDVI mean pixels pr class. Using 6-folds cross validation'

        for x in range(1, 41):
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

            print "KNN Test: " + str(x) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

            # Training Set Path setup
            t_path = '../../Classify Set/NDVI/' + dates[set] + '/Mean/'
            training_file = str(samples) + 'SamplesPrClassNDVI_' + datefiles[set] + '_Mean' + '_resampled.csv'
            traning_set = t_path + str(samples) + 'SamplesPrClassNDVI_' + datefiles[set] + '_Mean' + '_resampled.csv'

            # Load training set
            test_set = loadTestSet(traning_set)
            X, Y = test_set.loadTestSet()

            # Set number of folds
            folds = 6
            kf = KFold(n_splits=folds)
            kf.get_n_splits(X)

            metrics_array = []
            fold = 0

            test_spec_dict = {'Classifier': 'KNN',
                              'Number of Neighbors': x, 'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                              'Training Set': traning_set, 'Sample Type': 'NDVI mean field single pixel', 'Path': t_path, 'File': training_file}

            for train_index, test_index in kf.split(X):
                fold += 1
                start_time = timeit.default_timer()

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                # Classifies fields with different k values from 1 to 50.
                y_pred = knnClassifier(x, X_train, y_train, X_test)

                # Calculate the different accuracy scores
                cm_metrics = Calculate_Metrics(y_test, y_pred)
                metrics_dict = cm_metrics.calculateScore()
                # Calculate the run time
                elapsed = (timeit.default_timer() - start_time) * 1000

                metrics_dict.update({'Run Time(MSec)': elapsed})

                # Create metrics for each fold
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

# Mean Combi bands Pixel Values
for i in range(1, 15):

    samples = i * 30

    t_size = (samples/6)*5


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
    test_description = 'KNN Test with variable k-size, from 1 to 60. From Date 0805_1104_1509_1903' + ' .With ' + str(t_size) + ' Combi bandset Mean pixels pr class. Using 6-folds cross validation'
    elapsedTraining1=0
    elapsedTesting2 = 0
    for x in range(7, 8):
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

        print "KNN Test: " + str(x) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

        # Training Set Path setup
        t_path = '../../Classify Set/Mean Pixel Value/Merged Sets/All/'
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

        test_spec_dict = {'Classifier': 'KNN',
                          'Number of Neighbors': x, 'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                          'Training Set': traning_set, 'Sample Type': 'Combi bandset Mean field single pixel', 'Path': t_path, 'File': training_file}

        for train_index, test_index in kf.split(X):
            fold += 1

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            start_time = timeit.default_timer()

            # Classifies fields with different k values from 1 to 50.
            elapsedTraining, elapsedTesting,y_pred = knnClassifier(x, X_train, y_train, X_test)
            elapsedTraining1 += elapsedTraining
            elapsedTesting2 += elapsedTesting
            # Calculate the different accuracy scores
            cm_metrics = Calculate_Metrics(y_test, y_pred)
            metrics_dict = cm_metrics.calculateScore()

            # Calculate the run time
            elapsed = (timeit.default_timer() - start_time) * 1000

            print "Time: " + str(elapsed)
            metrics_dict.update({'Run Time(MSec)': elapsed})

            # Create metrics for each fold
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
        # elapsedTraining1 =elapsedTraining1/6
        # elapsedTesting2 = elapsedTesting2/6
        print 'Time in ms for training: ' + str(elapsedTraining1/6)
        print 'Time in ms for testing: ' + str(elapsedTesting2/6)
print "Done"


## Random Combi bands Pixel Values
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
    test_description = 'KNN Test with variable k-size, from 1 to 60. From Date 0805_1104_1509_1903' + ' .With ' + str(t_size) + ' Combi bandset Random pixels pr class. Using 6-folds cross validation'

    for x in range(1, 61):
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

        print "KNN Test: " + str(x) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

        # Training Set Path setup
        t_path = '../../Classify Set/Random Pixels/Merged Sets/0805_1104_1509_1903/'
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

        test_spec_dict = {'Classifier': 'KNN',
                          'Number of Neighbors': x, 'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                          'Training Set': traning_set, 'Sample Type': 'Combi bandset Random field single pixel', 'Path': t_path, 'File': training_file}

        for train_index, test_index in kf.split(X):
            fold += 1
            start_time = timeit.default_timer()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Classifies fields with different k values from 1 to 50.
            y_pred = knnClassifier(x, X_train, y_train, X_test)

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

        # Create PDF
        # pdf = printPDF(metrics_mean_dict)
        # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

        # Create CSV
        csv = printCSV(metrics_mean_dict, test_spec_dict)
        csv.createCSV(path, csvfile_name)

print "Done"

## Mean Combi bands Pixel Values
for i in range(1, 15):

    samples = i * 30

    t_size = (samples/6)*5


    ## Overall configurations
    hostname = socket.gethostname()
    testStart = time.strftime("(%Y-%m-%d)")
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
    path = str(t_size) + '_0805_1104_1509_1903' + '_Combi_NDVI_Mean' + '_' + testStart + '_' + hostname + '/' # Overall configurations
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


    ## Test Description
    test_description = 'KNN Test with variable k-size, from 1 to 60. From Date 0805_1104_1509_1903' + ' .With ' + str(t_size) + ' Combi bandset NDVI Mean pixels pr class. Using 6-folds cross validation'

    for x in range(1, 61):
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

        print "KNN Test: " + str(x) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

        # Training Set Path setup
        t_path = '../../Classify Set/NDVI/Merged Sets/0805_1104_1509_1903/Mean/'
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

        test_spec_dict = {'Classifier': 'KNN',
                          'Number of Neighbors': x, 'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                          'Training Set': traning_set, 'Sample Type': 'Combi bandset NDVI Mean field single pixels', 'Path': t_path, 'File': training_file}

        for train_index, test_index in kf.split(X):
            fold += 1
            start_time = timeit.default_timer()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Classifies fields with different k values from 1 to 50.
            y_pred = knnClassifier(x, X_train, y_train, X_test)

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

        # Create PDF
        # pdf = printPDF(metrics_mean_dict)
        # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

        # Create CSV
        csv = printCSV(metrics_mean_dict, test_spec_dict)
        csv.createCSV(path, csvfile_name)

print "Done"

## Mean Combi bands Pixel Values
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
    test_description = 'KNN Test with variable k-size, from 1 to 60. From Date 0805_1104_1509_1903' + ' .With ' + str(t_size) + ' Combi bandset NDVI Random pixels pr class. Using 6-folds cross validation'

    for x in range(1, 61):
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

        print "KNN Test: " + str(x) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

        # Training Set Path setup
        t_path = '../../Classify Set/NDVI/Merged Sets/0805_1104_1509_1903/Single Random/'
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

        test_spec_dict = {'Classifier': 'KNN',
                          'Number of Neighbors': x, 'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                          'Training Set': traning_set, 'Sample Type': 'Combi bandset NDVI Random field single pixels', 'Path': t_path, 'File': training_file}

        for train_index, test_index in kf.split(X):
            fold += 1
            start_time = timeit.default_timer()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Classifies fields with different k values from 1 to 50.
            y_pred = knnClassifier(x, X_train, y_train, X_test)

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

        # Create PDF
        # pdf = printPDF(metrics_mean_dict)
        # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

        # Create CSV
        csv = printCSV(metrics_mean_dict, test_spec_dict)
        csv.createCSV(path, csvfile_name)

print "Done"


## Mean UnCut Pixel Values
for i in range(1, 15):

    samples = i * 30

    t_size = (samples/6)*5


    ## Overall configurations
    hostname = socket.gethostname()
    testStart = time.strftime("(%Y-%m-%d)")
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")
    path = str(t_size) + '_' + dates[0] + '_Mean_UnCut' + '_' + testStart + '_' + hostname + '/' # Overall configurations
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


    ## Test Description
    test_description = 'KNN Test with variable k-size, from 1 to 60. From Date ' + dates[0] + ' .With ' + str(t_size) + ' Bandset Mean UnCut pixels pr class. Using 6-folds cross validation'

    for x in range(1, 61):
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

        print "KNN Test: " + str(x) + ' | ' + str(samples) + "  Time: " + testTimeConvertet

        # Training Set Path setup
        t_path = '../../Classify Set/UnCut Fields/Mean Pixel Values/' + dates[0] + '/'
        training_file = str(samples) + 'SamplesPrClassBandset_NoCut_Mean_' + datefiles[0] + '_resampled.csv'
        traning_set = t_path + str(samples) + 'SamplesPrClassBandset_NoCut_Mean_' + datefiles[0] + '_resampled.csv'

        # Load training set
        test_set = loadTestSet(traning_set)
        X, Y = test_set.loadTestSet()

        # Set number of folds
        folds = 6
        kf = KFold(n_splits=folds)
        kf.get_n_splits(X)

        metrics_array = []
        fold = 0

        test_spec_dict = {'Classifier': 'KNN',
                          'Number of Neighbors': x, 'Date': testTimeConvertet, 'Host': hostname, 'Training Size': t_size,
                          'Training Set': traning_set, 'Sample Type': 'Bandset Mean UnCut field single pixel', 'Path': t_path, 'File': training_file}

        for train_index, test_index in kf.split(X):
            fold += 1
            start_time = timeit.default_timer()

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            # Classifies fields with different k values from 1 to 50.
            y_pred = knnClassifier(x, X_train, y_train, X_test)

            # Calculate the different accuracy scores
            cm_metrics = Calculate_Metrics(y_test, y_pred)
            metrics_dict = cm_metrics.calculateScore()
            # Calculate the run time
            elapsed = (timeit.default_timer() - start_time) * 1000

            metrics_dict.update({'Run Time(MSec)': elapsed})

            #Create metrics for each fold
            # pdf_indi = printPDF(metrics_dict)
            # pdf_indi.create_pdf_indi(test_spec_dict, test_description, test_path_indi,
            #                         str(fold) + '_' + testTimeConvertet + '_' + hostname)

            metrics_array.append(metrics_dict)

        # Calculate the overall score for all the folds
        metrics_mean_dict = cm_metrics.calculate_mean_score(metrics_array, folds)

        # Create PDF
        # pdf = printPDF(metrics_mean_dict)
        # pdf.create_pdf(test_spec_dict, test_description, test_path, str(x) + '_' + testTimeConvertet + '_' + hostname)

        # Create CSV
        csv = printCSV(metrics_mean_dict, test_spec_dict)
        csv.createCSV(path, csvfile_name)

print "Done"