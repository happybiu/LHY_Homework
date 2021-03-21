import numpy as np
import math

with open('./data/X_train') as f:
    next(f)                              # 跳过第一行
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

with open('./data/Y_train') as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

with open('./data/X_test') as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

def normalize(x_data):
    mean = np.mean(x_data,axis=0)
    std = np.std(x_data, axis=0)
    for i in range(x_data.shape[0]):
        x_data[i, :] = (x_data[i, :] - mean) / (std + 0.000001)
    return x_data

def split_train_val(x_data, ratio):
    train_data = x_data[:math.floor((len(x_data)*ratio)), :]
    val_data = x_data[math.floor(len(x_data)*ratio):, :]
    return train_data, val_data

# def cross_entropy(X_train, Y_train, predict):
#     return -((Y_train*np.log(predict)+(1-Y_train)*np.log(1-predict))

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

X_train = normalize(X_train)
X_train, X_val = split_train_val(X_train, ratio=0.9)
Y_train, Y_val = split_train_val(Y_train, ratio=0.9)
X_test= normalize(X_test)

w = np.ones([510])
b = np.ones([1])

learning_rate = 0.01
X_train = np.concatenate((X_train, np.ones([X_train.shape[0],1])), axis=1)

f = np.dot(X_train,np.concatenate((w,b),axis=0))




for epoch in range(100):
    cross_entropy = -( np.dot(np.log(_sigmoid(f)),Y_train) + np.dot(np.log(1-_sigmoid(f)),(1-Y_train)) )
    gradient = np.dot(X_train.T, Y_train - _sigmoid(f))
    w = w - learning_rate*gradient
    b = b - learning_rate*gradient


