import pandas as pd
import numpy as np

# 处理原始数据,得到x_train_data和y_train_data
raw_train_data = pd.read_csv('./data/raw/train.csv')
x_train_data = raw_train_data
x_train_data[x_train_data=='NR'] = 0
x_train_data = x_train_data.iloc[:,3:]
x_train_data = x_train_data.to_numpy()
list_train_data = np.empty([12*20*16,18*9])
y_train_data = np.empty([12*20*16])
for month in range(12):
    for day in range(20):
        for hour in range(16):
            list_train_data[month*20 + day*16 + hour] = x_train_data[(month*20+day)*18 : (month*20+day+1)*18 , hour:hour+9].flatten()
            y_train_data[month*20 + day*16 + hour] = x_train_data[9,hour+10]

# 建立模型
w = np.zeros([18*9])
b = np.zeros([1])
predict_y = np.dot(x,w) + b

# 构建损失函数
loss = 


