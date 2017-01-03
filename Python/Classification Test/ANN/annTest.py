import os
import errno
import datetime
import time
import csv
import sys
import socket
import timeit
import sys


from loadTestSet import loadTestSet
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from time import time
import csv

if __name__ == '__main__':

    for date in range(0,1):

            dates = ['08-05-16', '11-04-16', '15-09-16', '19-03-16']
            datefiles = ['0805', '1104', '1509', '1903', 'MergedSet']

            for t in range(1, 15):

                training_file = 30 * t
                training_nr = 25 * t

                print '\n Data set: ' + str(training_nr) + ' - Mean Merged ' + dates[date] + ' with Mean Std \n'

                # Training Set Path setup
                traning_set = ''

                test_results_file = 'Test_Results_' + str(training_file) + '_.csv'
                best_results_file = 'Best_Results_' + str(training_file) + '_.csv'


                # Load training set
                test_set = loadTestSet(traning_set)
                X, Y = test_set.loadTestSet()
                t0 = time()
                parameters = dict(activation=('relu', 'tanh'),
                                  alpha=[1e-07, 1e-06, 1e-05, 1e-04],
                                  learning_rate_init=[1e-03],
                                  hidden_layer_sizes=[(160,),(180,), (200,), (220,), (240,),(260,), (280,), (300,), (320,), (340,),
                                                      (160,160,),(180,180,), (200,200,),(220,220,), (240,240,),(260,260,), (280,280,), (300,300,), (320,320,), (340,340,),
                                                      (160,160,160,),(180, 180,180,), (200, 200,200,), (220, 220,220,), (240, 240,240,), (260, 260,260,), (280, 280,280,), (300, 300,300,), (320, 320,320,),(340, 340,340,)])

                estimator = MLPClassifier(solver='adam', random_state=1, max_iter=5000)

                grid = GridSearchCV(estimator, parameters, cv=6, n_jobs=-1, verbose=1)
                grid.fit(X, Y)

                #print sorted(clf.cv_results_.keys())
                print("done in %0.3fs" % (time() - t0))
                print("Best estimator found by grid search:")
                print(grid.best_estimator_)

                print sorted(grid.cv_results_.keys())

                result = grid.cv_results_['mean_test_score']

                mean_test_score = grid.cv_results_['mean_test_score']
                mean_fit_time = grid.cv_results_['mean_fit_time']
                mean_score_time = grid.cv_results_['mean_score_time']
                mean_train_score = grid.cv_results_['mean_train_score']
                param_activation = grid.cv_results_['param_activation']
                param_alpha = grid.cv_results_['param_alpha']
                param_hidden_layer_sizes = grid.cv_results_['param_hidden_layer_sizes']
                param_learning_rate_init = grid.cv_results_['param_learning_rate_init']
                params = grid.cv_results_['params']
                rank_test_score = grid.cv_results_['rank_test_score']
                split0_test_score = grid.cv_results_['split0_test_score']
                split0_train_score = grid.cv_results_['split0_train_score']
                split1_test_score = grid.cv_results_['split1_test_score']
                split1_train_score = grid.cv_results_['split1_train_score']
                split2_test_score = grid.cv_results_['split2_test_score']
                split2_train_score = grid.cv_results_['split2_train_score']
                std_fit_time = grid.cv_results_['std_fit_time']
                std_score_time = grid.cv_results_['std_score_time']
                std_test_score = grid.cv_results_['std_test_score']
                std_train_score = grid.cv_results_['std_train_score']

                with open(test_results_file, 'wb') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=';',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(['Mean Score', 'Mean fit time', 'Mean Score time', 'Mean train score', 'Rank',
                                         'Activation Function', 'Hidden Units/Layers', 'alpha', 'Learning Rate',
                                         'Std Fit time', 'Std Score time', 'Std Test Score', 'Std Train Score'])

                    for i in range(0,len(mean_test_score)):
                        spamwriter.writerow([mean_test_score[i], mean_fit_time[i], mean_score_time[i], mean_train_score[i],
                                             rank_test_score[i], param_activation[i],param_hidden_layer_sizes[i],param_alpha[i],
                                             param_learning_rate_init[i], std_fit_time[i], std_score_time[i], std_test_score[i], std_train_score[i]])

                with open(best_results_file, 'wb') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=';',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(['Best'])
                    spamwriter.writerow([grid.best_estimator_])

                print "Done"
