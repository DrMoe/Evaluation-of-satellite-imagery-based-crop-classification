import os
import csv

class printCSV:

    def __init__(self,metrics_dict,test_spec_dict):
        self.metrics_dict = metrics_dict
        self.test_spec_dict = test_spec_dict

    def createCSV(self,path,csvfile_name,count="",append=False):

        try:
            os.makedirs(path + 'Overall/')
        except OSError:
            if not os.path.isdir(path):
                raise
        if append:
            mode = 'a'
        else:
            mode = 'wb'

        if count == "":
            with open(path + 'Overall/' + csvfile_name, mode) as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                if not append:
                    spamwriter.writerow(['Accuracy', 'Kappa', 'F1 Score', 'Run Time (MSec)', 'Accuracy Std', 'Kappa Std', 'F1 Std'])
                spamwriter.writerow([self.metrics_dict['accuracy_normalized'], self.metrics_dict['kappa_all'],
                                     self.metrics_dict['f1_score_micro'], self.metrics_dict['Run Time(MSec)'],
                                     self.metrics_dict['accuracy_normalized_std'],self.metrics_dict['kappa_all_std'],self.metrics_dict['f1_score_micro_std']])
        else:
            with open(path + 'Overall/' + csvfile_name, mode) as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                if not append:
                    spamwriter.writerow(['Count','Accuracy', 'Kappa', 'F1 Score', 'Run Time (MSec)', 'Accuracy Std', 'Kappa Std', 'F1 Std'])
                spamwriter.writerow([str(count), self.metrics_dict['accuracy_normalized'], self.metrics_dict['kappa_all'],
                                     self.metrics_dict['f1_score_micro'], self.metrics_dict['Run Time(MSec)'],
                                     self.metrics_dict['accuracy_normalized_std'],self.metrics_dict['kappa_all_std'],self.metrics_dict['f1_score_micro_std']])