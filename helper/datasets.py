import csv

import numpy
from sklearn.preprocessing import StandardScaler

import featureedit

def csv2numpy(csv_in):
    # Parse CSV file
    csv_rows = list(csv.reader(open(csv_in, 'rb')))
    classes = {'FALSE':0, 'TRUE':1}
    rownum = 0
    # Count exact number of data points
    TOTAL_ROWS = 0
    for row in csv_rows:
        if row[0] in classes:
            # Count line if it begins with a class label (boolean)
            TOTAL_ROWS += 1
    # X = vector of data points, y = label vector
    X = numpy.array(numpy.zeros((TOTAL_ROWS, featureedit.FeatureDescriptor.get_feature_count())), dtype=numpy.float64, order='C')
    y = numpy.array(numpy.zeros(TOTAL_ROWS), dtype=numpy.float64, order='C')
    file_names = []
    for row in csv_rows:
        # Skip line if it doesn't begin with a class label (boolean)
        if row[0] not in classes:
            continue
        # Read class label from first row
        y[rownum] = classes[row[0]]
        featnum = 0
        file_names.append(row[1])
        for featval in row[2:]:
            if featval in classes:
                # Convert booleans to integers
                featval = classes[featval]
            X[rownum, featnum] = float(featval)
            featnum += 1
        rownum += 1
    return X, y, file_names

def numpy2csv(csv_out, X, y, file_names=None):
    we_opened_csvfile = type(csv_out) == str
    csvfile = open(csv_out, 'wb+') if we_opened_csvfile else csv_out
    # Write header
    csvfile.write('class')
    if file_names:
        csvfile.write(',filename')
    names = featureedit.FeatureDescriptor.get_feature_names()
    for name in names:
        csvfile.write(',{}'.format(name))
    csvfile.write('\n')
    descs = featureedit.FeatureDescriptor.get_feature_descriptions()
    # Write data
    for i in range(0, X.shape[0]):
        csvfile.write('{}'.format('TRUE' if bool(y[i]) else 'FALSE'))
        if file_names:
            csvfile.write(',{}'.format(file_names[i]))
        for j in range(0, X.shape[1]):
            feat_type = descs[names[j]]['type']
            feat_val = X[i, j]
            if feat_type == bool:
                feat_val = 'TRUE' if feat_val >= 0.5 else 'FALSE'
            elif feat_type == int:
                feat_val = int(round(feat_val))
            csvfile.write(',{}'.format(feat_val))
        csvfile.write('\n')
    
    if we_opened_csvfile:
        csvfile.close()

# Ham chuan hoa du lieu cua mot file .csv
def standardize_csv(csv_in, csv_out, standardizer=None):
    X, y, file_names = csv2numpy(csv_in)
#     X = X.todense()
    if standardizer is None:
        standardizer = StandardScaler(copy=False)
        standardizer.fit(X)
    standardizer.transform(X)
    numpy2csv(csv_out, X, y, file_names)
    del X
    return standardizer 
