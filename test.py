import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

a=np.array([1,2,3])
b=np.array([4,5,6])
print(np.sum(a<=2.5))

from sklearn.preprocessing import MinMaxScaler
scale=list(range(49,73))
df = pd.read_csv("res01.csv", sep=',',usecols=scale)
data_all=np.array(df).astype(float)
idx=[]
data_prep=[]
scaler = MinMaxScaler()
for line in data_all:
    for i in range(24):
        if line[i]>0 :
            idx.append(i-1)
            break
j=0
for line in data_all:
    line=line[idx[j]:]
    j+=1
    line=np.reshape(line,(-1,1))
    line=scaler.fit_transform(np.array(line))
    print(line)
    data_prep.append(line)
data_fin=[]
sequence_length=5
for line in data_prep:
    for i in range(len(line) - sequence_length - 1):
        data_fin.append(line[i: i + sequence_length + 1])
data_fin=np.array(data_fin)




#
# def load_data(file_name, sequence_length=5, split=0.6):
#     scale = list(range(50, 73))
#     df = pd.read_csv("out.csv", sep=',', usecols=scale)
#     data_all = np.array(df).astype(float)
#     idx = []
#     data_prep = []
#     scaler = MinMaxScaler()
#     for line in data_all:
#         for i in range(24):
#             if line[i] > 0:
#                 idx.append(i - 1)
#                 break
#     j = 0
#     for line in data_all:
#         line = line[idx[j]:]
#         j += 1
#         line = np.reshape(line, (-1, 1))
#         line = scaler.fit_transform(np.array(line))
#         print(line)
#         data_prep.append(line)
#     data_fin = []
#     sequence_length = 5
#     for line in data_prep:
#         for i in range(len(line) - sequence_length - 1):
#             data_fin.append(line[i: i + sequence_length + 1])
#     data_fin = np.array(data_fin)
#     reshaped_data = np.array(data_fin).astype('float64')
#     np.random.shuffle(reshaped_data)
#     # 对x进行统一归一化，而y则不归一化
#     print(reshaped_data.shape)
#     x = reshaped_data[:, :-1]
#     print(x.shape)
#     y = reshaped_data[:, -1]
#     split_boundary = int(reshaped_data.shape[0] * split)
#     train_x = x[: split_boundary]
#     test_x = x[split_boundary:]
#
#     train_y = y[: split_boundary]
#     test_y = y[split_boundary:]
#
#     return train_x, train_y, test_x, test_y, scaler
#
#
# train_x, train_y, test_x, test_y, scaler = load_data('out.csv')