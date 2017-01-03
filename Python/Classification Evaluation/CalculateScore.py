import timeit
import os
import errno
import socket
import datetime
import time
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, cohen_kappa_score
import sklearn.metrics as skm


class Calculate_Metrics:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculateScore(self):
        """
        Calculates various accuracy scores based on the confusion matrix.

        Args:
            confusionMatrixFile (string): The file containing the confusion matrix (.csv)
        Returns:
            Metrics Dictionary
        """

        metrics_dict = self.compute_scores(self.y_true, self.y_pred)

        return metrics_dict

    def compute_scores(self,y_true, y_pred):
        con = confusion_matrix(y_true, y_pred)

        array_dimension = con.shape
        row_dimensions = array_dimension[0]

        # Create a matrix contaning TP(0), FP(1), FN(2), TN(3) for all classes
        TP_FP_FN_TN_List = []
        for x in range(0, row_dimensions):
            TP_FP_FN_TN_List.append(self.process_cm(con, x, to_print=False))
        TP_FP_FN_TN_Array = np.vstack((TP_FP_FN_TN_List))

        # Compute TNR(True Negative Rate)/Specificity
        # TNR = TN / TN + FP
        TNR_List = []
        Accuracy_List = []
        for x in range(0, row_dimensions):
            TNR_List.append(
                float(TP_FP_FN_TN_Array[x][3]) / (float(TP_FP_FN_TN_Array[x][3] + float(TP_FP_FN_TN_Array[x][1]))))
            Accuracy_List.append((float(TP_FP_FN_TN_Array[x][0]) + float(TP_FP_FN_TN_Array[x][3]))
                                 / (float(TP_FP_FN_TN_Array[x][0]) + float(TP_FP_FN_TN_Array[x][1])
                                    + float(TP_FP_FN_TN_Array[x][2]) + float(TP_FP_FN_TN_Array[x][3])))
        TNR_Array = np.hstack((TNR_List))
        Acc_Array = np.hstack((Accuracy_List))

        # Compute F1 Score
        f1_score_all = f1_score(y_true, y_pred, average=None)
        f1_score_macro = f1_score(y_true, y_pred, average='macro')
        f1_score_micro = f1_score(y_true, y_pred, average='micro')
        f1_score_weighted = f1_score(y_true, y_pred, average='weighted')

        # Compute Recall Score
        recall_all = recall_score(y_true, y_pred, average=None)
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')

        # Compute Precision Score
        precision_all = precision_score(y_true, y_pred, average=None)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')

        # Compute Kappa Score
        kappa_all = cohen_kappa_score(y_true, y_pred)
        kappa_linear = cohen_kappa_score(y_true, y_pred, weights='linear')
        kappa_quadratic = cohen_kappa_score(y_true, y_pred, weights='quadratic')

        # Compute Accuracy Score
        accuracy_all = skm.accuracy_score(y_true, y_pred, normalize=False)
        accuracy_normalized = skm.accuracy_score(y_true, y_pred, normalize=True)

        # Compute Hamming Loss
        hamming_loss = skm.hamming_loss(y_true, y_pred)

        # Compute Jaccard Score
        jaccard_all = skm.jaccard_similarity_score(y_true, y_pred, normalize=False)
        jaccard_normalized = skm.jaccard_similarity_score(y_true, y_pred, normalize=True)

        # Compute Zero One Loss
        zero_one_all = skm.zero_one_loss(y_true, y_pred, normalize=False)
        zero_one_normalize = skm.zero_one_loss(y_true, y_pred, normalize=True)

        class_code = (1, 10, 11, 22, 216)

        metrics_dict = {'TpFpFnTn': TP_FP_FN_TN_Array, 'TNR': TNR_Array, 'Acc_Indi': Acc_Array,
                        'f1_score_all': f1_score_all,'f1_score_macro': f1_score_macro, 'f1_score_micro': f1_score_micro,
                        'f1_score_weighted': f1_score_weighted, 'recall_all': recall_all, 'recall_macro': recall_macro,
                        'recall_micro': recall_micro, 'recall_weighted': recall_weighted,
                        'precision_all': precision_all, 'precision_macro': precision_macro,
                        'precision_micro': precision_micro, 'precision_weighted': precision_weighted,
                        'kappa_all': kappa_all, 'kappa_linear': kappa_linear, 'kappa_quadratic': kappa_quadratic,
                        'accuracy_all': accuracy_all, 'accuracy_normalized': accuracy_normalized,
                        'hamming_loss': hamming_loss, 'jaccard_all': jaccard_all,
                        'jaccard_normalized': jaccard_normalized, 'zero_one_all': zero_one_all,
                        'zero_one_normalize': zero_one_normalize, 'con_matrix': con,'class_code': class_code}

        return metrics_dict

    def process_cm(self,confusion_mat, i=0, to_print=True):
        """
        Processes the confusion matrix generated by scikit.
        Finds True positives, False positives, False negatives and True negatives

        Args:
            confusionMatrixFile (array): The confusion matrix
        Returns:
            TP, FP, FN, TN (int): Contains the number
        """

        # i means which class to choose to do one-vs-the-rest calculation
        # rows are actual obs whereas columns are predictions
        TP = confusion_mat[i, i]  # correctly labeled as i
        FP = confusion_mat[:, i].sum() - TP  # incorrectly labeled as i
        FN = confusion_mat[i, :].sum() - TP  # incorrectly labeled as non-i
        TN = confusion_mat.sum().sum() - TP - FP - FN
        if to_print:
            print('TP: {}'.format(TP))
            print('FP: {}'.format(FP))
            print('FN: {}'.format(FN))
            print('TN: {}'.format(TN))
        return TP, FP, FN, TN

    def calculate_mean_score(self,metrics_array,folds):
        """
        Calculates the mean score of each score in the metrics_array.

        Args:
            metrics_array (array): Multiple arrays that contains metrics for each fold
        Returns:
            metrics_mean_dict (array): Mean metrics array
        """

        # Initialize the different variables
        f1_score_macro, f1_score_micro, f1_score_weighted = 0,0,0
        recall_macro,recall_micro,recall_weighted,precision_macro,precision_micro,precision_weighted = 0,0,0,0,0,0
        kappa_all,kappa_linear,kappa_quadratic,accuracy_all,accuracy_normalized,hamming_loss,jaccard_all = 0,0,0,0,0,0,0
        jaccard_normalized,zero_one_all,zero_one_normalize, run_time = 0,0,0,0

        for x in range(0,folds):
            f1_score_macro += metrics_array[x]['f1_score_macro']
            f1_score_micro += metrics_array[x]['f1_score_micro']
            f1_score_weighted += metrics_array[x]['f1_score_weighted']
            recall_macro += metrics_array[x]['recall_macro']
            recall_micro += metrics_array[x]['recall_micro']
            recall_weighted += metrics_array[x]['recall_weighted']
            precision_macro += metrics_array[x]['precision_macro']
            precision_micro += metrics_array[x]['precision_micro']
            precision_weighted += metrics_array[x]['precision_weighted']
            kappa_all += metrics_array[x]['kappa_all']
            kappa_linear += metrics_array[x]['kappa_linear']
            kappa_quadratic += metrics_array[x]['kappa_quadratic']
            accuracy_all += metrics_array[x]['accuracy_all']
            accuracy_normalized += metrics_array[x]['accuracy_normalized']
            hamming_loss += metrics_array[x]['hamming_loss']
            jaccard_all += metrics_array[x]['jaccard_all']
            jaccard_normalized += metrics_array[x]['jaccard_normalized']
            zero_one_all += metrics_array[x]['zero_one_all']
            zero_one_normalize += metrics_array[x]['zero_one_normalize']
            run_time += metrics_array[x]['Run Time(MSec)']

        # Compute the mean score
        f1_score_macro = f1_score_macro / float(folds)
        f1_score_micro = f1_score_micro / float(folds)
        f1_score_weighted = f1_score_weighted / float(folds)
        recall_macro = recall_macro / float(folds)
        recall_micro = recall_micro / float(folds)
        recall_weighted = recall_weighted / float(folds)
        precision_macro = precision_macro / float(folds)
        precision_micro = precision_micro / float(folds)
        precision_weighted = precision_weighted / float(folds)
        kappa_all = kappa_all / float(folds)
        kappa_linear = kappa_linear / float(folds)
        kappa_quadratic = kappa_quadratic / float(folds)
        accuracy_all = accuracy_all / float(folds)
        accuracy_normalized = accuracy_normalized / float(folds)
        hamming_loss = hamming_loss / float(folds)
        jaccard_all = jaccard_all / float(folds)
        jaccard_normalized = jaccard_normalized / float(folds)
        zero_one_all = zero_one_all / float(folds)
        zero_one_normalize = zero_one_normalize / float(folds)
        run_time = run_time / float(folds)

        accuracy_normalized_array = []
        kappa_all_array = []
        f1_score_micro_array = []
        for x in range(0,folds):
            accuracy_normalized_array.append(metrics_array[x]['accuracy_normalized'])
            kappa_all_array.append(metrics_array[x]['kappa_all'])
            f1_score_micro_array.append(metrics_array[x]['f1_score_micro'])
        accuracy_normalized_std = np.std(accuracy_normalized_array)
        kappa_all_std = np.std(kappa_all_array)
        f1_score_micro_std = np.std(f1_score_micro_array)

        # Finds the correct number of classes
        array_dimension = metrics_array[0]['con_matrix'].shape
        row_dimensions = array_dimension[0]

        TP_FP_FN_TN_Array = []
        for i in range(0,row_dimensions):
            TP,FP,FN,TN = 0,0,0,0
            for x in range(0,folds):
                TP += metrics_array[x]['TpFpFnTn'][i][0]
                FP += metrics_array[x]['TpFpFnTn'][i][1]
                FN += metrics_array[x]['TpFpFnTn'][i][2]
                TN += metrics_array[x]['TpFpFnTn'][i][3]
            TP = TP / float(folds)
            FP = FP / float(folds)
            FN = FN / float(folds)
            TN = TN / float(folds)
            TP_FP_FN_TN_Array.append((TP,FP,FN,TN))
        TP_FP_FN_TN_Array = np.vstack((TP_FP_FN_TN_Array))

        samples = 0
        for x in range(0,5):
            samples += metrics_array[x]['con_matrix'][0][x]

        # Confidence intalval
        ci = accuracy_normalized -  (accuracy_normalized - 1.96 * (accuracy_normalized_std/np.sqrt(samples)))



        TNR_List = []
        for i in range(0, row_dimensions):
            value = 0
            for x in range(0, folds):
                value += metrics_array[x]['TNR'][i]
            value = value / float(folds)
            TNR_List.append(value)

        Acc_List = []
        for i in range(0, row_dimensions):
            value = 0
            for x in range(0, folds):
                value += metrics_array[x]['Acc_Indi'][i]
            value = value / float(folds)
            Acc_List.append(value)

        F1_score_all_List = []
        for i in range(0, row_dimensions):
            value = 0
            for x in range(0, folds):
                value += metrics_array[x]['f1_score_all'][i]
            value = value / float(folds)
            F1_score_all_List.append(value)

        recall_all_List = []
        for i in range(0, row_dimensions):
            value = 0
            for x in range(0, folds):
                value += metrics_array[x]['recall_all'][i]
            value = value / float(folds)
            recall_all_List.append(value)

        precision_all_List = []
        for i in range(0, row_dimensions):
            value = 0
            for x in range(0, folds):
                value += metrics_array[x]['precision_all'][i]
            value = value / float(folds)
            precision_all_List.append(value)

        class_code = (1, 10, 11, 22, 216)

        metrics_mean_dict = {'TpFpFnTn': TP_FP_FN_TN_Array, 'TNR': TNR_List, 'Acc_Indi': Acc_List,
                            'f1_score_all': F1_score_all_List,'f1_score_macro': f1_score_macro, 'f1_score_micro': f1_score_micro,
                            'f1_score_weighted': f1_score_weighted, 'recall_all': recall_all_List, 'recall_macro': recall_macro,
                            'recall_micro': recall_micro, 'recall_weighted': recall_weighted,
                            'precision_all': precision_all_List, 'precision_macro': precision_macro,
                            'precision_micro': precision_micro, 'precision_weighted': precision_weighted,
                            'kappa_all': kappa_all, 'kappa_linear': kappa_linear, 'kappa_quadratic': kappa_quadratic,
                            'accuracy_all': accuracy_all, 'accuracy_normalized': accuracy_normalized,
                            'hamming_loss': hamming_loss, 'jaccard_all': jaccard_all, 'Run Time(MSec)': run_time,
                            'jaccard_normalized': jaccard_normalized, 'zero_one_all': zero_one_all,
                            'zero_one_normalize': zero_one_normalize,'class_code': class_code, 'con_matrix': metrics_array[0]['con_matrix'],
                             'accuracy_normalized_std': accuracy_normalized_std, 'kappa_all_std': kappa_all_std, 'f1_score_micro_std': f1_score_micro_std,
                             'confidence_level': ci}

        return metrics_mean_dict