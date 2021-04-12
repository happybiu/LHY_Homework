import pandas as pd
import numpy as np

# 读取数据
raw_train_data = pd.read_csv('./data/raw/train.csv', encoding = 'big5')

# 处理原始数据：处理表格中的字符串等,得到全部是数字的x_train_data和y_train_data
x_train_data = raw_train_data
x_train_data[x_train_data=='NR'] = 0
x_train_data = x_train_data.iloc[:,3:]
x_train_data = x_train_data.to_numpy()

# 处理x_train_data成list_train_data，并且将一个样本的所有feature标准化
list_train_data = np.empty([12*20*15,18*9], dtype = float)
y_train_data = np.empty([12*20*15, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(15):
            index = month*20 + day*15 + hour
            list_train_data[month*20 + day*15 + hour] = x_train_data[(month*20+day)*18 : (month*20+day+1)*18 , hour:hour+9].flatten()
            y_train_data[month*20 + day*15 + hour] = x_train_data[9, hour+9]
            # 标准化
            train_mean = np.mean(list_train_data[month*20 + day*15 + hour])
            train_std = np.std(list_train_data[month*20 + day*15 + hour])
            if train_std != 0:
                list_train_data[month*20 + day*15 + hour] = (list_train_data[month*20 + day*15 + hour] - train_mean) / train_std

epoch = 1000
lr = 100

# 训练模型
w = np.ones([18*9,1])
b = np.ones([1])
w_sum_grad2 = 0
b_sum_grad2 = 0
eps = 0.0000000001
b_list_train_data = np.ones([12*20*15,1])
for i in range(epoch):
    # 建立模型
    predict_y = np.dot(list_train_data, w) + b
    # 定义损失函数
    loss = np.sqrt(np.sum( np.power(y_train_data-predict_y,2))/12/20/15)
    print(loss)
    # 找到最小化损失函数的解
    w_gradient = 2*np.dot(list_train_data.T, (predict_y-y_train_data))
    b_gradient = 2*np.dot(b_list_train_data.T, (predict_y-y_train_data))
    w_sum_grad2 += w_gradient ** 2
    b_sum_grad2 += b_gradient **2
    w = w - lr * w_gradient / np.sqrt(w_sum_grad2 + eps)
    b = b - lr * b_gradient / np.sqrt(b_sum_grad2 + eps)

print("w=", w)
print("b=", b)

# 测试

# 准备原始数据
raw_test_data = pd.read_csv('./data/raw/test.csv', encoding = 'big5', header = None)
# 清理原始数据
x_test_data = raw_test_data.iloc[:, 2:]
x_test_data[x_test_data == 'NR'] = 0
x_test_data = x_test_data.to_numpy()
# 准备list_test_data
list_test_data = np.empty([240, 18*9], dtype = float)
for i in range(240):
    list_test_data[i, :] = x_test_data[i*18:(i+1)*18, :].flatten()
    test_mean = np.mean(list_test_data[i, :])
    test_std = np.std(list_test_data[i, :])
    # 标准化
    if test_std != 0:
        list_test_data[i, :] = (list_test_data[i, :] - test_mean) / test_std

# 将测试数据丢入模型进行预测
predict_test_data = np.dot(list_test_data, w) + b
print(predict_test_data)
df_predict_test_data = pd.DataFrame(predict_test_data)
df_predict_test_data.to_csv('./data/mycode_predict_test_data.csv')


