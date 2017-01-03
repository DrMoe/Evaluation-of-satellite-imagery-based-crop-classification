import csv
import os
import numpy as np

class loadTestSet:

    def __init__(self, testset_csv):
        self.testset_csv = testset_csv

    def test_lenght(self,fields, field):
        for i in range(1,20):
            nr = field + '_' + str(i)
            if nr in fields:
                if len(fields[nr]) < 10:
                    return True, nr
            else:
                return False, nr


    def loadTestSet_FieldObjects(self,samples, pixels_pr_field):
        with open(self.testset_csv, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            csvreader.next()  # Skip the first line as it contains text

            fields_1 = {}
            fields_10 = {}
            fields_11 = {}
            fields_22 = {}
            fields_216 = {}

            for row in csvreader:
                columns = len(row)
                Field = row[columns - 1]
                AfgKode = row[columns - 2]

                if AfgKode == '1':
                    if Field in fields_1:
                        if len(fields_1[Field]) >= 10:
                            test, nr = self.test_lenght(fields_1,Field)
                            if test:
                                fields_1[nr].append(row)
                            else:
                                fields_1[nr] = [row]
                        else:
                            fields_1[Field].append(row)
                    else:
                        fields_1[Field] = [row]


                if AfgKode == '10':
                    if Field in fields_10:
                        if len(fields_10[Field]) >= 10:
                            test, nr = self.test_lenght(fields_10,Field)
                            if test:
                                fields_10[nr].append(row)
                            else:
                                fields_10[nr] = [row]
                        else:
                            fields_10[Field].append(row)
                    else:
                        fields_10[Field] = [row]



                if AfgKode == '11':
                    if Field in fields_11:
                        if len(fields_11[Field]) >= 10:
                            test, nr = self.test_lenght(fields_11,Field)
                            if test:
                                fields_11[nr].append(row)
                            else:
                                fields_11[nr] = [row]
                        else:
                            fields_11[Field].append(row)
                    else:
                        fields_11[Field] = [row]



                if AfgKode == '22':
                    if Field in fields_22:
                        if len(fields_22[Field]) >= 10:
                            test, nr = self.test_lenght(fields_22,Field)
                            if test:
                                fields_22[nr].append(row)
                            else:
                                fields_22[nr] = [row]
                        else:
                            fields_22[Field].append(row)
                    else:
                        fields_22[Field] = [row]



                if AfgKode == '216':
                    if Field in fields_216:
                        if len(fields_216[Field]) >= 10:
                            test, nr = self.test_lenght(fields_216,Field)
                            if test:
                                fields_216[nr].append(row)
                            else:
                                fields_216[nr] = [row]
                        else:
                            fields_216[Field].append(row)
                    else:
                        fields_216[Field] = [row]


        spectral_nr = columns - 2

        class_value_1 = []
        field1 = np.empty((samples, pixels_pr_field, spectral_nr))

        field, field_pixels = 0, 0
        for key, value in fields_1.iteritems():
            for pixel in value:
                for i in range(0, spectral_nr):
                    field1[field][field_pixels][i] = float(pixel[i])
                field_pixels += 1
                if field_pixels == pixels_pr_field:
                    field_pixels = 0
                    break
            class_value_1.append((int(pixel[spectral_nr])))
            field += 1

        class_value_10 = []
        field10 = np.empty((samples, pixels_pr_field, spectral_nr))

        field, field_pixels = 0, 0
        for key, value in fields_10.iteritems():
            for pixel in value:
                for i in range(0, spectral_nr):
                    field10[field][field_pixels][i] = float(pixel[i])
                field_pixels += 1
                if field_pixels == pixels_pr_field:
                    field_pixels = 0
                    break
            class_value_10.append((int(pixel[spectral_nr])))
            field += 1

        class_value_11 = []
        field11 = np.empty((samples, pixels_pr_field, spectral_nr))

        field, field_pixels = 0, 0
        for key, value in fields_11.iteritems():
            for pixel in value:
                for i in range(0, spectral_nr):
                    field11[field][field_pixels][i] = float(pixel[i])
                field_pixels += 1
                if field_pixels == pixels_pr_field:
                    field_pixels = 0
                    break
            class_value_11.append((int(pixel[spectral_nr])))
            field += 1

        class_value_22 = []
        field22 = np.empty((samples, pixels_pr_field, spectral_nr))

        field, field_pixels = 0, 0
        for key, value in fields_22.iteritems():
            for pixel in value:
                for i in range(0, spectral_nr):
                    field22[field][field_pixels][i] = float(pixel[i])
                field_pixels += 1
                if field_pixels == pixels_pr_field:
                    field_pixels = 0
                    break
            class_value_22.append((int(pixel[spectral_nr])))
            field += 1

        class_value_216 = []
        field216 = np.empty((samples, pixels_pr_field, spectral_nr))

        field, field_pixels = 0, 0
        for key, value in fields_216.iteritems():
            for pixel in value:
                for i in range(0, spectral_nr):
                    field216[field][field_pixels][i] = float(pixel[i])
                field_pixels += 1
                if field_pixels == pixels_pr_field:
                    field_pixels = 0
                    break
            class_value_216.append((int(pixel[spectral_nr])))
            field += 1

        test_set = []
        class_set = []

        for i in range(0, samples):
            test_set.append(field1[i])
            class_set.append(class_value_1[i])
            test_set.append(field10[i])
            class_set.append(class_value_10[i])
            test_set.append(field11[i])
            class_set.append(class_value_11[i])
            test_set.append(field22[i])
            class_set.append(class_value_22[i])
            test_set.append(field216[i])
            class_set.append(class_value_216[i])

        test_set = np.asanyarray(test_set)
        nsamples, nx, ny = test_set.shape

        X = test_set.reshape((nsamples, nx * ny))
        Y = np.asanyarray(class_set)

        return X,Y

    def loadTestSet(self):
        with open(self.testset_csv, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            csvreader.next()  # Skip the first line as it contains text

            raster_value = []
            class_value = []

            for row in csvreader:
                columns = len(row)
                value = []
                for i in range(0, columns - 1):
                    singlevalue = float(row[i])
                    value.extend([singlevalue])
                raster_value.append(value)
                class_value.append((int(row[columns - 1])))

        x_data = np.asarray(raster_value)
        y_labels = np.asarray(class_value)

        return x_data, y_labels