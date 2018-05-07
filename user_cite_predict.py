#source_code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import *
from keras.layers import LSTM, Dense, Activation

#seqlength 可以改！
# def load_data(file_name, sequence_length=5, split=0.8):
#     df = pd.read_csv(file_name, sep=',', usecols=[1])
#     data_all = np.array(df).astype(float)
#     print(data_all)
#     scaler = MinMaxScaler()
#     data_all = scaler.fit_transform(data_all)
#     data = []
#     for i in range(len(data_all) - sequence_length - 1):
#         data.append(data_all[i: i + sequence_length + 1])
#     reshaped_data = np.array(data).astype('float64')
#     np.random.shuffle(reshaped_data)
#     x = reshaped_data[:, :-1]
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

def load_data(file_name, sequence_length=5, cv=0.8,test=0.9):
    scale = list(range(49, 73))
    df = pd.read_csv(file_name, sep=',', usecols=scale)
    data_all = np.array(df).astype(float)
    idx = []
    data_prep = []
    scaler = MinMaxScaler()

    for line in data_all:
        for i in range(25):
            if line[i] > 0:
                idx.append(i - 1)
                break

    j = 0
    for line in data_all:
        line = line[idx[j]:]
        j += 1
        line = np.reshape(line, (-1, 1))
        line = scaler.fit_transform(np.array(line))
        # print(np.sum(line))
        data_prep.append(line)

    data_fin = []
    for line in data_prep:
        for i in range(len(line) - sequence_length - 1):
            data_fin.append(line[i: i + sequence_length + 1])
    data_fin = np.array(data_fin)

    reshaped_data = np.array(data_fin).astype('float64')
    np.random.shuffle(reshaped_data)
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    cv_boundary = int(reshaped_data.shape[0] * cv)
    test_bound=int(reshaped_data.shape[0] * test)
    train_x = x[: cv_boundary]
    cv_x = x[cv_boundary:test_bound]
    test_x=x[test_bound:]

    train_y = y[: cv_boundary]
    cv_y=y[cv_boundary:test_bound]
    test_y = y[test_bound:]

    return train_x, train_y, test_x, test_y, cv_x, cv_y, scaler

def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    print(model.layers)
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='Adam')
    return model


def train_model(train_x, train_y, test_x, test_y,cv_x,cv_y):
    model = build_model()
    try:
        # batch_size for optimize
        model.fit(train_x, train_y, batch_size=512, nb_epoch=30, validation_split=0.1)
        model.save('my_model01.h5')
        #prediction
        predict_cv = model.predict(cv_x)
        predict_cv = np.reshape(predict_cv, (predict_cv.size, ))
        predict_t=model.predict(test_x)
        predict_t=np.reshape(predict_t, (predict_t.size, ))
    except KeyboardInterrupt:
        print(predict_cv)
        print(predict_t)
        print(test_y)
    # print(predict_cv)
    # print(test_y)
    try:
        fig = plt.figure(1)
        plt.plot(predict_cv, 'r:')
        plt.plot(test_y, 'g-')
        plt.legend(['predict', 'true'])
    except Exception as e:
        print(e)
    return predict_cv, predict_t,cv_y,test_y

def evaluation(y_pd,y_true,thd=0.3):
    err=(y_pd-y_true)/(y_true+6)
    mape=np.sum(np.abs(err))/y_true.shape[0]
    acc=np.sum(err<=thd)/y_true.shape[0]

    return mape,acc

def use_mode(file_model,test):
    model=load_model(file_model)
    predict=model.predict(test)
    return predict

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, cv_x,cv_y,scaler= load_data('.csv')
    # train_x, train_y, test_x, test_y= load_data('international-airline-passengers.csv')
    print(train_x.shape)
    print(train_y.shape)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))#设置成3维
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    # predict_t=use_mode(test_x)
    predict_cv, predict_t, cv_y, test_y = train_model(train_x, train_y, test_x, test_y, cv_x, cv_y)
    # sum_t=np.sum(test_x,axis=1)
    # retrieve_x=scaler.inverse_transform([[i] for i in test_x])  #将test_x还原
    predict_t = scaler.inverse_transform(predict_t)
    test_y = scaler.inverse_transform(test_y)


    # t1=predict_t+sum_t
    # t2=test_y+sum_t
    # for i in t2:
    #     print(i)
    # mape, acc = evaluation(t1, t2)
    # print(mape,acc)
    # print(mape,acc)


    # cv_x = np.reshape(cv_x, (cv_x.shape[0], cv_x.shape[1], 1))
    # predict_cv, predict_t, cv_y, test_y = train_model(train_x, train_y, test_x, test_y,cv_x,cv_y)

    # print (predict_cv.shape)
    # predict_t = scaler.inverse_transform([[i] for i in predict_t])
    # # predict_y =[[i] for i in predict_y]
    # test_y = scaler.inverse_transform(test_y)

    # caculate indication
    mape,acc=evaluation(predict_t,test_y)
    print(mape)
    print(acc)


