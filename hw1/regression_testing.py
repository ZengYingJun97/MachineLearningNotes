import csv
import sys

import numpy as np
import pandas as pd

w_set = np.load('weight.npy')
mean_x = np.load('mean.npy')
std_x = np.load('std.npy')

#Testing
inputFileName = sys.argv[1]
testData = pd.read_csv(inputFileName, header=None, encoding='big5')
testData = testData.iloc[:, 2:]
testData[testData == 'NR'] = 0
test_data = testData.to_numpy()
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i : 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
ans_y = np.dot(test_x, w_set)

#Ans
outputFileName = sys.argv[2]
with open(outputFileName, mode='w', newline='') as ans_file:
    csv_writer = csv.writer(ans_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        csv_writer.writerow(['id_' + str(i), ans_y[i][0]])