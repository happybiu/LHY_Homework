# 缺少考虑的地方：
# astype(float)  这一步骤的作用是？我check了下，x.dtype本来就是float,
# eps
# 没有划分验证集

import pandas as pd
import numpy as np

data = pd.read_csv('train.csv', encoding='big5')
data = data.iloc[:, 3:]       
data[data=='NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    x = np.empty([18,24*20])
    for day in range(20):
        x[:, day*24: (day+1)*24] = raw_data[(month*20+day)*18:(month*20+day+1)*18, :]
    month_data[month] = x

x = np.empty([471*12, 18*9])
target_data = np.empty([471*12, 1])
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month*471+day*24+hour] = month_data[month][:, day*24+hour:day*24+hour+9].reshape(1,-1)
            target_data[month*471+day*24+hour][0] = month_data[month][9,day*24+hour+9]

for i in range(471*12):
    mean = np.mean(x[i])
    std = np.std(x[i])
    for j in range(18*9):
        x[i][j] = (x[i][j]- mean) / std

w = np.ones([18*9+1, 1])
x = np.concatenate((np.ones([471*12,1]),x),axis=1)
learning_rate = 0.01
adagrad = 0

for epoch in range(100):
    loss = np.sqrt(np.power(np.dot(x,w)-target_data, 2).sum()/471/12)
    if epoch%10 == 0: 
        print("epoch=%d:  loss=%.2f\n"%(epoch, loss))
    gradient = 2*np.dot(x.transpose(),np.dot(x,w)-target_data)
    adagrad += np.power(gradient, 2)
    w = w - learning_rate * gradient / np.sqrt(adagrad)

test_data = pd.read_csv('test.csv', header=None, encoding='big5')
test_data[test_data == 'NR'] = 0
test_data = test_data.iloc[:,2:]
test_data = test_data.to_numpy()

test_x = np.empty([240,18*9])
for i in range(240):
    test_x[i, :] = test_data[i * 18 : (i + 1) * 18, :].reshape(1, -1)

for i in range(240):
    mean = np.mean(test_x[i])
    std = np.std(test_x[i])
    for j in range(18*9):
        test_x[i][j] = (test_x[i][j]- mean) / std

test_x = np.concatenate((np.ones([240,1]),test_x),axis=1)
print("before: test_x.dtype =", test_x.dtype)
test_x.astype(float)
print("after: test_x.dtype =", test_x.dtype)
predict = np.dot(test_x, w)
predict = pd.DataFrame(predict)
predict.to_csv('predict2.csv')

