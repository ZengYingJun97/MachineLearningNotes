import csv

import numpy as np
import pandas as pd
import math

# 载入数据
data = pd.read_csv('../data/train.csv', encoding='big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

#提取特征
month_data = {}
for month in range(12):
    sample  = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour >= 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]

#标准化
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

#训练
size = 18 * 9 + 1
w_set = np.zeros([size, 1])
x_set = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([size, 1])
eps = 0.000000001
for t in range(iter_time):
    pre_y = np.dot(x_set, w_set)
    loss = np.sqrt(np.sum(np.power(pre_y - y, 2)) / (471 * 12))
    gradient = 2 * np.dot(x_set.transpose(), pre_y - y)
    adagrad += gradient ** 2
    w_set = w_set - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w_set)

#测试
testData = pd.read_csv('../data/test.csv', header=None, encoding='big5')
test_data = testData.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
test_x
w_set = np.load('weight.npy')
ans_y = np.dot(test_x, w_set)

#保存数据
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id, value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)