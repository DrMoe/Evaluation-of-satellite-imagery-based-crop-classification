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
from runDT_Test import runDT_Test

dates = ['08-05-16', '11-04-16', '15-09-16', '19-03-16']
datestitle = [', at -8$^{th}$ of May, 2016', 'at 11$^{th}$ of April, 2016', 'at 15$^{th}$ of Sep, 2016',
              ', at - 19$^{th}$ of March, 2016']
datefiles = ['0805', '1104', '1509', '1903', 'MergedSet']

## Mean Single field pixel
for set in range(0, 4):

    DT_Test = runDT_Test()

    ## Test Type
    type = 'Mean Pixel Value'
    ## Create PDF
    create_pdf = False

    ## Overall configurations
    hostname = socket.gethostname()
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")

    path = type + ' - ' + str(datefiles[set]) + '/' + testStartConvertet + '_' + hostname + '/'

    for i in range(1, 15):

        traning_size = 25 * i
        samples = i * 30
        t_size = (samples / 6) * 5

        ## Test Description
        test_description = 'DT Test with variable depth-size, from 1 to 20. From Date ' + dates[set] + ' .With ' + str(
            t_size) + ' mean pixels pr class. Using 6-folds cross validation'

        ## Training Set
        traning_set = '../../Classify Set/Mean Pixel Value/' + dates[set] + '/' + str(
            samples) + 'SamplesPrClassBandset_Mean_' + datefiles[set] + '_resampled.csv'

        title = 'Mean merged data set(Id:5)' + datestitle[set]
        figfile = 'CombiBandsetMeanPixel_Accuracy_' + datefiles[set] +'.pdf'

        for x in range(1, 21):

            csvfile_name = DT_Test.runTest(path,x,traning_set,test_description,type,traning_size,create_pdf)

        DT_Test.combineResults(path,traning_size)

    DT_Test.plot(path,title,figfile)

# Random pixels single
for set in range(0, 4):

    DT_Test = runDT_Test()

    ## Test Type
    type = 'Random Pixel Value'
    ## Create PDF
    create_pdf = False

    ## Overall configurations
    hostname = socket.gethostname()
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")

    path = type + ' - ' + str(datefiles[set]) + '/' + testStartConvertet + '_' + hostname + '/'

    for i in range(1, 15):

        traning_size = 25 * i
        samples = i * 30
        t_size = (samples / 6) * 5

        ## Test Description
        test_description = 'DT Test with variable depth-size, from 1 to 20. From Date ' + dates[set] + ' .With ' + str(
            t_size) + ' random pixels pr class. Using 6-folds cross validation'

        ## Training Set
        traning_set = '../../Classify Set/Random Pixels/' + dates[set] + '/' + str(
            samples) + 'SamplesPrClassBandset_Random_' + datefiles[set] + '_resampled.csv'

        title = 'Bandset with random pixels ' + datestitle[set]
        figfile = 'CombiBandsetRandomPixel_Accuracy_' + datefiles[set] +'.pdf'

        for x in range(1, 21):

            csvfile_name = DT_Test.runTest(path,x,traning_set,test_description,type,traning_size,create_pdf)

        DT_Test.combineResults(path,traning_size)

    DT_Test.plot(path,title,figfile)



## Merged Mean Pixels
for set in range(4, 5):

    DT_Test = runDT_Test()

    ## Test Type
    type = 'Merged Mean Pixel Value'
    ## Create PDF
    create_pdf = False

    ## Overall configurations
    hostname = socket.gethostname()
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")

    path = type + ' - ' + str(datefiles[set]) + '/' + testStartConvertet + '_' + hostname + '/'

    for i in range(1, 15):

        traning_size = 25 * i
        samples = i * 30
        t_size = (samples / 6) * 5

        ## Test Description
        test_description = 'DT Test with variable depth-size, from 1 to 20. From Date 0805_1104_1509_1903.With ' + str(
            t_size) + ' Combi Bandset using mean pixels pr class. Using 6-folds cross validation'

        ## Training Set
        traning_set = '../../Classify Set/Mean Pixel Value/Merged Sets/All/' + str(
            samples) + 'SamplesPrClassRandom_0805_1104_1509_1903_Mean_resampled.csv'

        title = 'Mean merged data set(Id:5) at - \n ' + '19$^{th}$ of March' + ', ' + '11$^{th}$ of April' + ', ' + '8$^{th}$ of May' + ', ' + '15$^{th}$ of Sep, 2016'
        figfile = 'MergedCombiBandsetMeanPixel_Accuracy_' + datefiles[set] +'.pdf'

        for x in range(6, 7):

            csvfile_name = DT_Test.runTest(path,x,traning_set,test_description,type,traning_size,create_pdf)

        DT_Test.combineResults(path,traning_size)

    DT_Test.plot(path,title,figfile)

# Merged Random Pixels
for set in range(4, 5):

    DT_Test = runDT_Test()

    ## Test Type
    type = 'Merged Random Pixel Value'
    ## Create PDF
    create_pdf = False

    ## Overall configurations
    hostname = socket.gethostname()
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")

    path = type + ' - ' + str(datefiles[set]) + '/' + testStartConvertet + '_' + hostname + '/'

    for i in range(1, 15):

        traning_size = 25 * i
        samples = i * 30
        t_size = (samples / 6) * 5

        ## Test Description
        test_description = 'DT Test with variable depth-size, from 1 to 20. From Date 0805_1104_1509_1903.With ' + str(
            t_size) + ' Combi Bandset using random pixels pr class. Using 6-folds cross validation'

        ## Training Set
        traning_set = '../../Classify Set/Random Pixels/Merged Sets/0805_1104_1509_1903/' + str(
            samples) + 'SamplesPrClassBandset_Random_0805_1104_1509_1903_resampled.csv'

        title = 'Merged Bandset with random pixels at - \n ' + '19$^{th}$ of March' + ', ' + '11$^{th}$ of April' + ', ' + '8$^{th}$ of May' + ', ' + '15$^{th}$ of Sep, 2016'
        figfile = 'MergedCombiBandsetRandomPixel_Accuracy_' + datefiles[set] +'.pdf'

        for x in range(1, 21):

            csvfile_name = DT_Test.runTest(path,x,traning_set,test_description,type,traning_size,create_pdf)

        DT_Test.combineResults(path,traning_size)

    DT_Test.plot(path,title,figfile)


Merged NDVI Mean Pixels
for set in range(4, 5):

    DT_Test = runDT_Test()

    ## Test Type
    type = 'Merged NDVI Mean Pixel Value'
    ## Create PDF
    create_pdf = False

    ## Overall configurations
    hostname = socket.gethostname()
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")

    path = type + ' - ' + str(datefiles[set]) + '/' + testStartConvertet + '_' + hostname + '/'

    for i in range(1, 15):

        traning_size = 25 * i
        samples = i * 30
        t_size = (samples / 6) * 5

        ## Test Description
        test_description = 'DT Test with variable depth-size, from 1 to 20. From Date 0805_1104_1509_1903.With ' + str(
            t_size) + ' Combi Bandset using NDVI mean pixels pr class. Using 6-folds cross validation'

        ## Training Set
        traning_set = '../../Classify Set/NDVI/Merged Sets/All/Mean/' + str(
            samples) + 'SamplesPrClassNDVI_Random_0805_1104_1509_1903_Mean_resampled.csv'

        title = 'Merged Bandset with NDVI mean pixels at - \n ' + '19$^{th}$ of March' + ', ' + '11$^{th}$ of April' + ', ' + '8$^{th}$ of May' + ', ' + '15$^{th}$ of Sep, 2016'
        figfile = 'MergedCombiBandsetNDVIMeanPixel_Accuracy_' + datefiles[set] +'.pdf'

        for x in range(1, 21):

            csvfile_name = DT_Test.runTest(path, x, traning_set, test_description, type, traning_size, create_pdf)

        DT_Test.combineResults(path,traning_size)

    DT_Test.plot(path,title,figfile)


## Merged NDVI Random Pixels
for set in range(4, 5):

    DT_Test = runDT_Test()

    ## Test Type
    type = 'Merged NDVI Random Pixel Value'
    ## Create PDF
    create_pdf = False

    ## Overall configurations
    hostname = socket.gethostname()
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")

    path = type + ' - ' + str(datefiles[set]) + '/' + testStartConvertet + '_' + hostname + '/'

    for i in range(1, 15):

        traning_size = 25 * i
        samples = i * 30
        t_size = (samples / 6) * 5

        ## Test Description
        test_description = 'DT Test with variable depth-size, from 1 to 20. From Date 0805_1104_1509_1903.With ' + str(
            t_size) + ' Combi Bandset using NDVI random pixels pr class. Using 6-folds cross validation'

        ## Training Set
        traning_set = '../../Classify Set/NDVI/Merged Sets/All/Single Random/' + str(
            samples) + 'SamplesPrClassNDVI_Random_0805_1104_1509_1903__resampled.csv'

        title = 'Merged Bandset with NDVI random pixels at - \n ' + '19$^{th}$ of March' + ', ' + '11$^{th}$ of April' + ', ' + '8$^{th}$ of May' + ', ' + '15$^{th}$ of Sep, 2016'
        figfile = 'MergedCombiBandsetNDVIRandomPixel_Accuracy_' + datefiles[set] +'.pdf'

        for x in range(1, 21):

            csvfile_name = DT_Test.runTest(path,x,traning_set,test_description,type,traning_size,create_pdf)

        DT_Test.combineResults(path,traning_size)

    DT_Test.plot(path,title,figfile)


# NDVI Mean pixels single
for set in range(0, 4):

    DT_Test = runDT_Test()

    ## Test Type
    type = 'NDVI Mean Pixel Value'
    ## Create PDF
    create_pdf = False

    ## Overall configurations
    hostname = socket.gethostname()
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")

    path = type + ' - ' + str(datefiles[set]) + '/' + testStartConvertet + '_' + hostname + '/'

    for i in range(1, 15):

        traning_size = 25 * i
        samples = i * 30
        t_size = (samples / 6) * 5

        ## Test Description
        test_description = 'DT Test with variable depth-size, from 1 to 20. From Date ' + dates[set] + ' .With ' + str(
            t_size) + ' NDVI mean pixels pr class. Using 6-folds cross validation'

        ## Training Set
        traning_set = '../../Classify Set/NDVI/' + dates[set] + '/Mean/' + str(
            samples) + 'SamplesPrClassNDVI_' + datefiles[set] + '_Mean_resampled.csv'

        title = 'Bandset with NDVI mean pixels ' + datestitle[set]
        figfile = 'CombiBandsetNDVIMeanPixel_Accuracy_' + datefiles[set] +'.pdf'

        for x in range(1, 21):

            csvfile_name = DT_Test.runTest(path,x,traning_set,test_description,type,traning_size,create_pdf)

        DT_Test.combineResults(path,traning_size)

    DT_Test.plot(path,title,figfile)


# NDVI Random pixels single
for set in range(0, 4):

    DT_Test = runDT_Test()

    ## Test Type
    type = 'NDVI random Pixel Value'
    ## Create PDF
    create_pdf = False

    ## Overall configurations
    hostname = socket.gethostname()
    testStartConvertet = time.strftime("(%Y-%m-%d)-(%H-%M-%S)")

    path = type + ' - ' + str(datefiles[set]) + '/' + testStartConvertet + '_' + hostname + '/'

    for i in range(1, 15):

        traning_size = 25 * i
        samples = i * 30
        t_size = (samples / 6) * 5

        ## Test Description
        test_description = 'DT Test with variable depth-size, from 1 to 20. From Date ' + dates[set] + ' .With ' + str(
            t_size) + ' NDVI random pixels pr class. Using 6-folds cross validation'

        ## Training Set
        traning_set = '../../Classify Set/NDVI/' + dates[set] + '/Single Random/' + str(
            samples) + 'SamplesPrClassNDVI_Random_' + datefiles[set] + '_resampled.csv'

        title = 'Bandset with NDVI random pixels ' + datestitle[set]
        figfile = 'CombiBandsetNDVIRandomPixel_Accuracy_' + datefiles[set] +'.pdf'

        for x in range(1, 21):

            csvfile_name = DT_Test.runTest(path,x,traning_set,test_description,type,traning_size,create_pdf)

        DT_Test.combineResults(path,traning_size)

     DT_Test.plot(path,title,figfile)