import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from loadTestSet import loadTestSet

path = ''

for i in range(0, 15):

    training_nr = 30 * i
    csvfilename = path + str(training_nr) + ''
    outputfile = str(training_nr) + ''

    # Load training set
    test_set = loadTestSet(csvfilename)
    X, Y = test_set.loadTestSet()

    #X = MinMaxScaler().fit_transform(X)
    X = StandardScaler().fit_transform(X)
    X_df = pd.DataFrame(data=X.astype(float))
    Y_df = pd.DataFrame(data=Y.astype(int))

    result = pd.concat([X_df, Y_df], axis=1)

    result.to_csv(outputfile, sep=';', header=['Band_6_0406_Mean','Band_7_1104_Mean','Band_8_0406_Mean','Band_8A_0406_Mean','Band_11_0406_Mean','Band_12_0406_Mean', 'AfgKode'], float_format='%.10f', index=False)

print "Done"


