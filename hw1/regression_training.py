import sys

import numpy as np
import pandas as pd

#Preprocessing
data = pd.read_csv('./data/train.csv', encoding='big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
input_data = data.to_numpy()

#Extract Features
month_data = {}
for month in range(12):
    temp = np.empty([18, 480])
    for day in range(20):
        temp[:, day * 24 : (day + 1) * 24] = input_data[(month * 20 + day) * 18 : (month * 20 + day + 1) * 18, :]
    month_data[month] = temp
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if hour > 14 and day == 19:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]

#Normalize
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

#Training
size = 18 * 9 + 1
x_set = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
w_set = np.zeros([size, 1])
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([size, 1])
eps = 0.000000001
for t in range(iter_time):
    pre_y = np.dot(x_set, w_set)
    loss = np.sqrt(np.sum(np.power(pre_y - y, 2)) / (12 * 471))
    gradient = 2 * np.dot(x_set.transpose(), pre_y - y)
    adagrad += gradient ** 2
    w_set = w_set - learning_rate * gradient / np.sqrt(adagrad + eps)

#Save
np.save('weight.npy', w_set)
np.save('mean.npy', mean_x)
np.save('std.npy', std_x)