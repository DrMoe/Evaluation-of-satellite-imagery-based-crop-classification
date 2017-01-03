import timeit
import os
import errno
import socket
import datetime
import time
import csv
import numpy as np
import shutil
from pylatex import Document, Section, Subsection, Description, Tabular, MultiColumn,\
    MultiRow, Itemize, Enumerate, Command, NoEscape

class printPDF:

    def __init__(self, metrics_dict):
        self.metrics_dict = metrics_dict

    def create_pdf(self, test_spec_dict, test_description, path, name):
        array_dimension = self.metrics_dict['con_matrix'].shape
        row_dimensions = array_dimension[0]

        array = self.metrics_dict['TpFpFnTn'][0]
        code = self.metrics_dict['class_code'][0]

        doc = Document("metrics")

        with doc.create(Section('Test description')):
            doc.append(test_description)
            with doc.create(Description()) as desc:
                for key, value in test_spec_dict.iteritems():
                    desc.add_item(key, value)

        section = Section('Metrics overview')

        test1 = Subsection('Rate matrix')

        # Create TN, TP, FP, FN table
        table1 = Tabular('cccccc')
        table1.add_hline()
        table1.add_row(("Class",'True-Positive','False-Positive','False-Negative','True-Negative', 'Accuracy'))
        table1.add_hline()
        for x in range(0, row_dimensions):
            table1.add_row([self.metrics_dict['class_code'][x], self.metrics_dict['TpFpFnTn'][x][0],
                            self.metrics_dict['TpFpFnTn'][x][1], self.metrics_dict['TpFpFnTn'][x][2],
                            self.metrics_dict['TpFpFnTn'][x][3], self.metrics_dict['Acc_Indi'][x]])
        table1.add_hline()
        test1.append(table1)

        test3 = Subsection('Class metrics')

        table3 = Tabular('ccccc')
        table3.add_hline()
        table3.add_row(("Class",'True-Positive Rate (TPR)','Precision','True-Negative Rate (TNR)','F1-Score'))
        table3.add_hline()
        for x in range(0, row_dimensions):
            table3.add_row([self.metrics_dict['class_code'][x], self.metrics_dict['recall_all'][x],
                            self.metrics_dict['precision_all'][x], self.metrics_dict['TNR'][x],
                            self.metrics_dict['f1_score_all'][x]])
        table3.add_hline()
        test3.append(table3)

        test2 = Subsection('Other')

        table2 = Tabular('cc')
        table2.add_hline()
        table2.add_row(("Class", "Value"))
        table2.add_hline()
        table2.add_row(["F1 Micro (Globally)", self.metrics_dict['f1_score_micro']])
        table2.add_row(["F1 Macro (Each label)", self.metrics_dict['f1_score_macro']])
        table2.add_row(["F1 Weighted (Each label)", self.metrics_dict['f1_score_weighted']])
        table2.add_row(["F1 Micro (Globally) Std", self.metrics_dict['f1_score_micro_std']])
        table2.add_hline()
        table2.add_row(["Recall Micro (Globally)", self.metrics_dict['recall_micro']])
        table2.add_row(["Recall Macro (Each label)", self.metrics_dict['recall_macro']])
        table2.add_row(["Recall Weighted (Each label)", self.metrics_dict['recall_weighted']])
        table2.add_hline()
        table2.add_row(["Precision Micro (Globally)", self.metrics_dict['precision_micro']])
        table2.add_row(["Precision Macro (Each label)", self.metrics_dict['precision_macro']])
        table2.add_row(["Precision Weighted (Each label)", self.metrics_dict['precision_weighted']])
        table2.add_hline()
        table2.add_row(["Kappa", self.metrics_dict['kappa_all']])
        table2.add_row(["Kappa (Linear weighted)", self.metrics_dict['kappa_linear']])
        table2.add_row(["Kappa (Quadratic weighted)", self.metrics_dict['kappa_quadratic']])
        table2.add_row(["Kappa Std", self.metrics_dict['kappa_all_std']])
        table2.add_hline()
        table2.add_row(["Accuracy (Correct classified)", self.metrics_dict['accuracy_all']])
        table2.add_row(["Accuracy (Normalized)", self.metrics_dict['accuracy_normalized']])
        table2.add_row(["Accuracy (Normalized) Std", self.metrics_dict['accuracy_normalized_std']])
        table2.add_row(["Confidence Level(95%)", self.metrics_dict['confidence_level']])
        table2.add_hline()
        table2.add_row(["Jaccard (Sum)", self.metrics_dict['jaccard_all']])
        table2.add_row(["Jaccard (Average)", self.metrics_dict['jaccard_normalized']])
        table2.add_hline()
        table2.add_row(["Zero-one classification loss (Misclassifications)", self.metrics_dict['zero_one_all']])
        table2.add_row(["Zero-one classification loss (Fraction of misclassifications)", self.metrics_dict['zero_one_normalize']])
        table2.add_hline()
        table2.add_row(["Hamming loss", self.metrics_dict['hamming_loss']])
        table2.add_hline()
        table2.add_row(["Run Time (MSec)", self.metrics_dict['Run Time(MSec)']])
        test2.append(table2)

        section.append(test1)
        section.append(test3)
        section.append(test2)
        doc.append(section)

        try:
            doc.generate_pdf(name + '_' + 'Metrics', compiler='pdflatex')
        except Exception:
            print ""

        shutil.move(name + '_' + 'Metrics' + '.pdf', path)

        try:
            os.remove(name + '_' + 'Metrics' + '.tex')
        except OSError:
            pass

        try:
            os.remove(name + '_' + 'Metrics' + '.log')
        except OSError:
            pass

        try:
            os.remove(name + '_' + 'Metrics' + '.aux')
        except OSError:
            pass

        return

    def create_pdf_indi(self, test_spec_dict, test_description, path, name):
        array_dimension = self.metrics_dict['con_matrix'].shape
        row_dimensions = array_dimension[0]

        array = self.metrics_dict['TpFpFnTn'][0]
        code = self.metrics_dict['class_code'][0]

        doc = Document("metrics")

        with doc.create(Section('Test description')):
            doc.append(test_description)
            with doc.create(Description()) as desc:
                for key, value in test_spec_dict.iteritems():
                    desc.add_item(key, value)

        section = Section('Metrics overview')

        test4 = Subsection('Confusion Matrix')

        crop_array = np.array(
            ['Spring Barly(1)', 'Winter Barley(10)', 'Winter Wheat(11)', 'Winter Rape(22)', 'Maize(216)'])

        # Create TN, TP, FP, FN table
        table4 = Tabular('cccccc')
        table4.add_hline()
        table4.add_row(('', 'Spring Barly', 'Winter Barley', 'Winter Wheat', 'Winter Rape', 'Maize'))
        table4.add_hline()
        for x in range(0, row_dimensions):
            table4.add_row([crop_array[x], self.metrics_dict['con_matrix'][x][0],
                            self.metrics_dict['con_matrix'][x][1], self.metrics_dict['con_matrix'][x][2],
                            self.metrics_dict['con_matrix'][x][3], self.metrics_dict['con_matrix'][x][4]])
        table4.add_hline()
        test4.append(table4)

        test1 = Subsection('Rate matrix')

        # Create TN, TP, FP, FN table
        table1 = Tabular('cccccc')
        table1.add_hline()
        table1.add_row(("Class", 'True-Positive', 'False-Positive', 'False-Negative', 'True-Negative', 'Accuracy'))
        table1.add_hline()
        for x in range(0, row_dimensions):
            table1.add_row([self.metrics_dict['class_code'][x], self.metrics_dict['TpFpFnTn'][x][0],
                            self.metrics_dict['TpFpFnTn'][x][1], self.metrics_dict['TpFpFnTn'][x][2],
                            self.metrics_dict['TpFpFnTn'][x][3], self.metrics_dict['Acc_Indi'][x]])
        table1.add_hline()
        test1.append(table1)

        test3 = Subsection('Class metrics')

        table3 = Tabular('ccccc')
        table3.add_hline()
        table3.add_row(("Class", 'True-Positive Rate (TPR)', 'Precision', 'True-Negative Rate (TNR)', 'F1-Score'))
        table3.add_hline()
        for x in range(0, row_dimensions):
            table3.add_row([self.metrics_dict['class_code'][x], self.metrics_dict['recall_all'][x],
                            self.metrics_dict['precision_all'][x], self.metrics_dict['TNR'][x],
                            self.metrics_dict['f1_score_all'][x]])
        table3.add_hline()
        test3.append(table3)

        test2 = Subsection('Other')

        table2 = Tabular('cc')
        table2.add_hline()
        table2.add_row(("Class", "Value"))
        table2.add_hline()
        table2.add_row(["F1 Micro (Globally)", self.metrics_dict['f1_score_micro']])
        table2.add_row(["F1 Macro (Each label)", self.metrics_dict['f1_score_macro']])
        table2.add_row(["F1 Weighted (Each label)", self.metrics_dict['f1_score_weighted']])
        table2.add_hline()
        table2.add_row(["Recall Micro (Globally)", self.metrics_dict['recall_micro']])
        table2.add_row(["Recall Macro (Each label)", self.metrics_dict['recall_macro']])
        table2.add_row(["Recall Weighted (Each label)", self.metrics_dict['recall_weighted']])
        table2.add_hline()
        table2.add_row(["Precision Micro (Globally)", self.metrics_dict['precision_micro']])
        table2.add_row(["Precision Macro (Each label)", self.metrics_dict['precision_macro']])
        table2.add_row(["Precision Weighted (Each label)", self.metrics_dict['precision_weighted']])
        table2.add_hline()
        table2.add_row(["Kappa", self.metrics_dict['kappa_all']])
        table2.add_row(["Kappa (Linear weighted)", self.metrics_dict['kappa_linear']])
        table2.add_row(["Kappa (Quadratic weighted)", self.metrics_dict['kappa_quadratic']])
        table2.add_hline()
        table2.add_row(["Accuracy (Correct classified)", self.metrics_dict['accuracy_all']])
        table2.add_row(["Accuracy (Normalized)", self.metrics_dict['accuracy_normalized']])
        table2.add_hline()
        table2.add_row(["Jaccard (Sum)", self.metrics_dict['jaccard_all']])
        table2.add_row(["Jaccard (Average)", self.metrics_dict['jaccard_normalized']])
        table2.add_hline()
        table2.add_row(["Zero-one classification loss (Misclassifications)", self.metrics_dict['zero_one_all']])
        table2.add_row(
            ["Zero-one classification loss (Fraction of misclassifications)", self.metrics_dict['zero_one_normalize']])
        table2.add_hline()
        table2.add_row(["Hamming loss", self.metrics_dict['hamming_loss']])
        test2.append(table2)

        section.append(test4)
        section.append(test1)
        section.append(test3)
        section.append(test2)
        doc.append(section)

        try:
            doc.generate_pdf(name + '_' + 'Metrics', compiler='pdflatex')
        except Exception:
            print ""

        shutil.move(name + '_' + 'Metrics' + '.pdf', path)

        try:
            os.remove(name + '_' + 'Metrics' + '.tex')
        except OSError:
            pass

        try:
            os.remove(name + '_' + 'Metrics' + '.log')
        except OSError:
            pass

        try:
            os.remove(name + '_' + 'Metrics' + '.aux')
        except OSError:
            pass

        return