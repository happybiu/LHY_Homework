<!-- TOC -->

- [机器学习常用的包](#机器学习常用的包)
  - [numpy](#numpy)
    - [介绍](#介绍)
    - [ndarray的属性](#ndarray的属性)
    - [ndarray的数据类型](#ndarray的数据类型)
    - [基本操作](#基本操作)
      - [创建ndarray：](#创建ndarray)
      - [range & np.arange & np.linspce](#range--nparange--nplinspce)
      - [形状修改](#形状修改)
      - [转置](#转置)
      - [类型转换](#类型转换)
      - [数组去重](#数组去重)
      - [数组运算](#数组运算)
    - [矩阵mat (尽量别用)](#矩阵mat-尽量别用)
      - [mat和ndarray在处理乘法时的比较](#mat和ndarray在处理乘法时的比较)
      - [矩阵的创建](#矩阵的创建)
      - [矩阵的逆](#矩阵的逆)
      - [矩阵连乘](#矩阵连乘)
    - [统计函数](#统计函数)
    - [比较和逻辑函数](#比较和逻辑函数)
    - [all & any](#all--any)
    - [其他操作](#其他操作)
    - [IO操作](#io操作)
      - [写入](#写入)
      - [读取](#读取)
      - [numpy 其他操作](#numpy-其他操作)
  - [Pandas](#pandas)
    - [介绍](#介绍-1)
    - [两种基本数据结构：Series & DataFrame](#两种基本数据结构series--dataframe)
    - [写入csv文件](#写入csv文件)
    - [读取csv文件](#读取csv文件)
    - [对DataFrame的操作](#对dataframe的操作)
  - [Matplotlib](#matplotlib)
    - [介绍](#介绍-2)
    - [画图](#画图)
    - [使用子图](#使用子图)
    - [散点图](#散点图)
    - [直方图](#直方图)
    - [标题、标签、图例 & 图片保存](#标题标签图例--图片保存)
    - [Seaborn](#seaborn)
  - [Scikit-learn](#scikit-learn)
    - [sklearn 表格](#sklearn-表格)
    - [sklearn数据集](#sklearn数据集)
    - [数据预处理](#数据预处理)
      - [min-max标准化 MinMaxScaler](#min-max标准化-minmaxscaler)
      - [Z-score标准化 StandardScaler](#z-score标准化-standardscaler)
      - [归一化](#归一化)
      - [二值化](#二值化)
      - [标签编码](#标签编码)
      - [独热编码](#独热编码)
    - [数据集的划分](#数据集的划分)
    - [定义模型](#定义模型)
      - [估计器（`Estimator`）](#估计器estimator)
      - [转换器（`Transformer`）](#转换器transformer)
    - [模型评估](#模型评估)
      - [交叉验证](#交叉验证)
    - [保存和加载模型：joblib](#保存和加载模型joblib)
  - [os 模块](#os-模块)
  - [cv2 模块](#cv2-模块)
  - [PIL Image对象](#pil-image对象)
    - [Image.open()和cv2.imread()的区别](#imageopen和cv2imread的区别)
  - [机器学习完整例子示范](#机器学习完整例子示范)
    - [步骤](#步骤)
    - [解释说明](#解释说明)
  - [pytorch](#pytorch)
    - [介绍](#介绍-3)
    - [历史发展](#历史发展)
    - [Tensorflow & Pytorch & keras的比较](#tensorflow--pytorch--keras的比较)
    - [查看pytorch版本](#查看pytorch版本)
    - [官方文档/官方手册](#官方文档官方手册)
    - [Tensor 张量](#tensor-张量)
      - [Tensor(张量)创建](#tensor张量创建)
      - [获取tensor的size](#获取tensor的size)
      - [运算：加减乘除、矩阵乘法、矩阵求逆](#运算加减乘除矩阵乘法矩阵求逆)
      - [改变tensor的维度和大小](#改变tensor的维度和大小)
      - [获得数值](#获得数值)
      - [ndarray和tensor的互相转换](#ndarray和tensor的互相转换)
      - [CUDA 张量](#cuda-张量)
    - [Autograd 自动求导机制](#autograd-自动求导机制)
      - [autograd.Variable](#autogradvariable)
        - [Variable创建](#variable创建)
      - [requires_grad = True](#requires_grad--true)
      - [梯度 反向传播](#梯度-反向传播)
      - [另一个autograd的例子](#另一个autograd的例子)
      - [with torch.no_grad()](#with-torchno_grad)
    - [数据加载与预处理](#数据加载与预处理)
      - [Dataset](#dataset)
      - [torchvision](#torchvision)
      - [torch.utils.data.DataLoader](#torchutilsdatadataloader)
    - [torch.nn 神经网络](#torchnn-神经网络)
      - [神经网络的典型训练过程](#神经网络的典型训练过程)
        - [老师版](#老师版)
        - [自己版](#自己版)
      - [nn.functional & nn.Module](#nnfunctional--nnmodule)
      - [定义网络模型](#定义网络模型)
      - [网络中的可学习参数](#网络中的可学习参数)
      - [损失函数](#损失函数)
      - [torch.optim 优化器：更新权重](#torchoptim-优化器更新权重)
      - [训练网络](#训练网络)
      - [GPU加速：cuda](#gpu加速cuda)
      - [验证](#验证)
    - [示例：训练一个图像分类器：CIFAR-10](#示例训练一个图像分类器cifar-10)
  - [math 模块](#math-模块)
  - [python语法](#python语法)
  - [Q&A](#qa)
- [Homework1: Regression](#homework1-regression)
  - [流程及注意事项](#流程及注意事项)
  - [Q&A](#qa-1)
- [Homework2: Classification - Logistic Regrassion](#homework2-classification---logistic-regrassion)
  - [流程及注意事项](#流程及注意事项-1)
  - [我的遗漏考虑](#我的遗漏考虑)
  - [Q&A](#qa-2)
- [Homework2: Classification - Logistic Regrassion](#homework2-classification---logistic-regrassion-1)
  - [流程及注意事项](#流程及注意事项-2)
  - [Q&A](#qa-3)
- [Homework3](#homework3)
  - [Q&A](#qa-4)
- [Homework4](#homework4)
  - [gensim](#gensim)

<!-- /TOC -->

<br>
<br>
<br>


# 机器学习常用的包

## numpy

### 介绍
- NumPy（Numerical Python）是一个开源的 Python 科学计算库，NumPy 支持常见的数组和矩阵操作，并且可以处理任意维度的数组（Tensor）。三维以上的数组就叫做tensor对于同样的数值计算任务，使用 NumPy 比直接使用 Python 要简洁的多。
- NumPy 通常与 SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用。Matplotlib 是 Python 编程语言及其数值数学扩展包 NumPy 的可视化操作界面。

### ndarray的属性
- numpy中最基本的数据类型就是ndarray，描述**相同类型**的集合
- 注意type(a)和ndarray.dtype的区别

| 属性名称 |属性解释 |
| -- | -- |
| ndarray.shape |	数组维度-元组 |
| ndarray.ndim |	数组维数 |
| ndarray.size |	数组中的元素数量 |
| ndarray.dtype |	数组元素的类型 |

```
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

print(a.shape)               
print(a.ndim)
print(a.dtype)                   # datatype
print(type(a))                   # type of a
print(a.size)                    # a里有多少个元素
```

### ndarray的数据类型
|名称|	描述|
|--|--|--|
|np.bool|	用一个字节存储的布尔类型（True或False）|
|np.int8|	一个字节大小，-128 至 127|
|np.int16|	整数，-32768 至 32767|
|np.int32|	整数，$-2^{31}$ 至 $2^{32} -1$	|
|np.int64|	整数，$-2^{63}$ 至 $2^{63} - 1$	|
|np.uint8|	无符号整数，0 至 255|
|np.uint16	|无符号整数，0 至 65535|
|np.uint32|	无符号整数，0 至 $2^{32} - 1$	|
|np.uint64|	无符号整数，0 至 $2^{64} - 1$ |
|np.float16	|半精度浮点数：16位，正负号1位，指数5位，精度10位	|
|np.float32	|单精度浮点数：32位，正负号1位，指数8位，精度23位	|
|np.float64	|双精度浮点数：64位，正负号1位，指数11位，精度52位	|
|np.complex64	|复数，分别用两个32位浮点数表示实部和虚部	|
|np.complex128	|复数，分别用两个64位浮点数表示实部和虚部	|
|np.object_	|python对象	|
|np.string_	|字符串	|
|np.unicode_	|unicode类型	|

### 基本操作
#### 创建ndarray：
- np.zeros([4, 3])
- np.ones([4, 3])
- np.eye(4,3) or np.eye(4)
- np.empty([4, 3])
- 后面均可以指定dtype
- 从现有数组中构建：已经有一个列表，直接由列表生成
```
import numpy as np

x = [[1, 2, 3], [4, 5, 6]
y = np.array(x)                 # 从现有数组中构建     

a = np.array([[1, 2, 3], [4, 5, 6]], dtype = np.float64)       # 指定类型
b = np.array([[1, 2, 3], [4, 5, 6]])

print("数组a：\n%s\n数据类型：\n%s"%(a, a.dtype))                # 输出学习一下
print("数组b：\n%s\n数据类型：\n%s"%(b, b.dtype))

c = np.zeros([4, 3])                                            # zeros,全为1，维度要加[]，输出是float64
d = np.ones([4, 3])                                             # ones，全为0，维度要加[]，输出是float64
e = np.eye(4)                                                   # eye,没有s，对角线是1，其余是0，np.eye(4,4)可简化为4，不用写[]，输出是float64
f = np.eye(10,5)                                                # 函数里面直接写维度，不用写[]，输出是float64

print(c)
print(c.dtype)                                                  # 输出是float64
print(d)
print(d.dtype)                                                  # 输出是float64
print(e)
print(e.dtype)                                                  # 输出是float64
print(f)                                                        # 对角线是1，其余是0，不一定非要是方阵
print(f.dtype)                                                  # 输出是float64
```

#### range & np.arange & np.linspce
| | range(start, stop, step) | np.arange(start, stop, step, dtype) | np.linspace(start, stop, num, endpoint, retstep, dtype) |
| -- | -- | -- | -- |
| 模块 | Python自带的函数 | 属于numpy模块 | 属于numpy模块 |
| 取值范围 | [start, stop) | [start, stop) | [start, stop] (默认)  or  [start, stop) |
| 参数解释 | |dtype：输出的array数据类型。如果未指定dtype，且start、stop、step任一个为浮点型，都会生成一个浮点型序列。 | start：必填，不可缺省<br> stop：可包含可不包含，默认包含，根据endpoint来选择<br> num：指定均分的数量，默认为50 <br> endpoint：布尔值，默认为True。True表示包含stop，flase表示不包含stop<br> retstep：布尔值，可选，默认为False。如果为True，返回值和步长<br> dtype：输出数据类型，可选。如果不指定，则根据前面参数的数据类型|
| 参数默认值 |start = 0, step = 1,start、stop、step必须是整型|start = 0, step = 1 |start = 0, num = 50|
| 返回值 | 返回一个list对象，只能生成整型的序列 | 返回一个ndarray对象，可以生成整型、浮点型序列，如果未指定dtype，且start、stop、step任一个为浮点型，都会生成一个浮点型序列 |返回一个ndarray对象，可以生成整型、浮点型序列，如果未指定dtype，且start、stop、step任一个为浮点型，都会生成一个浮点型序列 |


```
import numpy as np

a = range(10,90)                                 # [start,stop)
b = np.arange(0,90,10)                           # [start,stop) 
c = np.linspace(0,90,10)                         # [start,stop] 

print(a)
print(b)
print(b.dtype)
print(c)
print(c.dtype)
```

#### 形状修改
```
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
a.astype(np.float64)
print(a)
print(a)
c = a.reshape([3,2])
print(a)
print(c)

```

#### 转置
- 三种方法：transpose方法、T属性以及swapaxes方法
- a.T
- 高维数组用transpose
- swapaxes：

#### 类型转换 
```
a = np.array([[[1, 2, 3], [4, 5, 6]], [[12, 3, 34], [5, 6, 7]]])
print(a.dtype)
b = a.astype(np.float32)                               # 不会修改a中元素的类型，而是另外返回一个类型修改后的ndarray
print(a.dtype)
print(b.dtype)
```



#### 数组去重
- unique是numpy模块下的函数，不是numpy.ndarray模块下的函数

```
x = np.array([[1,2,3],[2,3,4],[3,4,5]])

y = np.unique(x)              # 正确
y = x.unique()                # 错误
print("y:\n",y)
```

#### 数组运算
- 数组的算术运算是元素级别的操作，新的数组被创建并且被结果填充。
  
运算|函数
-- | -- 
a + b | np.add(a, b)
a - b | np.subtract(a, b)
a * b | np.multiply(a, b)<br>np.multiply(a, b, a)：表示将结果传入第三个参数<br>
a / b | np.divide(a, b)
a ** b | np.power(a, b)
a % b | np.remainder(a, b)

### 矩阵mat (尽量别用)
- 注意：numpy虽然现在还能用matrix，但未来不会再用了，因为matrix只能表示2维数组，不像tensor一样可以表示多维数组，所以尽量不要用matrix类型，现在用矩阵的地方尽量用ndarray来表示

#### mat和ndarray在处理乘法时的比较
| * | np.multiply() | np.dot() |
| -- | -- | -- |
|数组ndarray：对应元素相乘，矩阵mat：矩阵相乘|数组ndarray和矩阵mat均是对应位置相乘|数组ndarray和矩阵mat均是矩阵相乘|
- np.dot与np.matmul的区别: 1.二者都是矩阵乘法。2.np.matmul中禁止矩阵与标量的乘法。3.在矢量乘矢量的內积运算中，np.matmul与np.dot没有区别。4.np.matmul中，多维的矩阵，将前n-2维视为后2维的元素后，进行乘法运算。
```
import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[9,8,7],[6,5,4],[3,2,1]])
c = np.mat(a)
d = np.mat(b)


print("a*b:\n",a*b)                                 # 对应元素相乘
print("c*d:\n",c*d)                                 # 矩阵乘法

print("multiplyArray:\n",np.multiply(a,b))          # 对应元素相乘
print("multiplyMat:\n",np.multiply(c,d))            # 对应元素相乘

print("dotArray:\n",np.dot(a,b))                    # 矩阵乘法
print("dotMat:\n",np.dot(c,d))                      # 矩阵乘法
```

#### 矩阵的创建
- 矩阵只能是2维
- 可以使用 mat 方法将 2 维数组转化为矩阵
- 也可以使用 **Matlab** 的语法传入一个字符串来生成矩阵
```
import numpy as np
 
a = np.mat("1,2,3;4,5,6;7,8,9")         # 字符串最后不需要加分号

print(a)
```

#### 矩阵的逆
- A.I 表示 A 矩阵的逆矩阵
- A必须要可逆，否则会报错：numpy.linalg.LinAlgError: Singular matrix
- 矩阵才有逆矩阵，二维数组没有逆矩阵，会报错：AttributeError: 'numpy.ndarray' object has no attribute 'I'

#### 矩阵连乘
- 矩阵指数表示矩阵连乘，A ** 4


### 统计函数
- 可以指定维度，若无指定，则默认第0维

|方法|作用|
|--|--|
|a.sum(axis=None)|所有元素求和|
|a.prod(axis=None)|求积|
|a.min(axis=None)|最小值|
|a.max(axis=None)|最大值|
|a.argmin(axis=None)|最小值对应的索引|
|a.argmax(axis=None)|最大值对应的索引|
|a.ptp(axis=None)|最大值减最小值|
|a.mean(axis=None)|平均值|
|a.std(axis=None)|标准差|
|a.var(axis=None)|方差|

- a.sum()和sum(a)的区别
```
import numpy as np

a = np.array([[1,2,3],[3,4,5],[4,5,6]])
a.sum()                                         # 所有元素求和
sum(a)                                          # 对应元素求和
np.sum(a, axis = -1)                             # 指定沿着最后一维求和，写法一
a.sum(axis = -1)                                 # 指定沿着最后一维求和，写法二
```

- np.mean(x, axis=?) 和 ndarray.mean()
mean()函数功能：求取均值
经常操作的参数为axis，以m * n矩阵举例：
axis 不设置值，对 m*n 个数求均值，返回一个实数
axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
axis = 1 ：压缩列，对各行求均值，返回 m *1 矩阵



### 比较和逻辑函数
| 运算符 | 函数 |
| :---: | :---: | 
| == | equal | 
| != | not_equal | 
| > | greater| 
| >= | greater_equal | 
| < | less | 
| <= | less_equal | 
```
import numpy as np

a = np.array([[1,2,3],[3,4,5],[4,5,6]])

a > 3                                        # 判断每个元素是否>3
a[a > 3] = 10                                # 将>3的元素赋值为10
```

### all & any
- 使用 all() 来判断某个区间的元素是否全部大于 20
- 使用 any() 来判断某个区间是否存在大于 20 的元素
```
import numpy as np

a = np.array([[ 0, 1, 2, 3, 4, 5],
           [10,11,12,13,14,15],
           [20,21,22,23,24,25],
           [30,31,32,33,34,35]])

print(a[1:3, 1:3])                         # 注意1：3范围是前闭后开，因此取值是1、2
b = np.all(a[1:3, 1:3] > 20)               # 判断是否所有都是>20
c = np.any(a[1:3, 1:3] > 20)               # 判断是否有任何一个>20
print(b)
print(c)
```

### 其他操作
- np.round(x,n):round() 方法返回浮点数x的四舍五入值。x表示数值，n表示四舍五入到小数点哪一位。n=-1表示个位，n=0表示小数点第一位
- np.abs()  绝对值

### IO操作
#### 写入
- np.savetxt('路径+文件名',data) 可以将数组写入文件，默认使用科学计数法的形式保存。如果没有路径，则默认存在当前文件夹
```
import numpy as np

a = np.array([[ 0, 1, 2, 3, 4, 5],
           [10,11,12,13,14,15],
           [20,21,22,23,24,25],
           [30,31,32,33,34,35]])

np.savetxt("newout.text",a)
```

#### 读取
- open('文件名'),其余参数均有默认值
- open(name[], mode[], buffering[]),mode默认为只读
- np.loadtxt('文件名'),其余参数均有默认值
np.loadtxt(fname, dtype=, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)


#### numpy 其他操作
- np.concatenate((a,b,c),axis=0)一次完成多个数组的拼接 axis=0表示行连接，axis=1表示列的数组进行拼接
- np.dot() 点乘?

## Pandas
### 介绍
- https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html
- numpy主要是用来处理数组数据，pandas也是用来处理数组数据，但pandas更多的是用来做一些数据分析，特别是去读一些像是csv文件，做一些统计分析
- Pandas 是基于 NumPy 的一种工具,该工具是为了解决**数据分析**任务而创建的
- Pandas 纳入了大量库及一些标准的数据模型，提供了高效的操作大型数据集所需要的工具
- Pandas 提供了大量能使我们快速便捷地处理数据的函数与方法
- 是 Python 成为强大而高效的数据分析环境的重要因素之一


### 两种基本数据结构：Series & DataFrame
- Series：带索引的一维数组，可存储整数、浮点数、字符串、Python 对象等类型的数据。
- DataFrame：DataFrame 是最常用的 Pandas 对象。由多种类型的列构成的二维标签数据结构，类似于 Excel 、SQL 表，或 Series 对象构成的字典。默认情况下，如果不指定 index 参数和 columns，那么他们的值将用从 0 开始的数字替代。
```
import numpy as np
import pandas as pd

a = pd.Series([1,2,np.nan,6.8,'a'])                     
print(a)

dates = pd.date_range(start = '20200101', end = '20200115')                        # 日期要加上' '
b = pd.DataFrame(np.random.randn(15,4), index = dates, columns = list('ABCD'))
print(b)
```

### 写入csv文件
- DataFrame.to_csv('文件名')
```
import numpy as np
import pandas as pd

b = pd.DataFrame(np.random.randn(15,4))

b.to_csv('out.csv')                             # 正确
pd.to_csv('out.csv')                            # 错误
```

### 读取csv文件
- head 和 tail 方法可以分别查看最前面几行和最后面几行的数据（默认为 5）
- 返回的数据类型是DataFrame
```
import numpy as np
import pandas as pd

b = pd.DataFrame(np.random.randn(15,4))
b.to_csv('out.csv')                             
a = pd.read_csv('out.csv')                    # pd.read_csv()读取后存于a中
print(a.head())                               # a.head()读取前几行,默认读取前5行
a.tail(10)                                    # a.tail()读取后几行
```


### 对DataFrame的操作
- data.iloc[:, :] 注意：dataframe的表头不算在index里面，因此dataframe index=0就是excel表格中的第一行
- loc函数：通过行索引 "Index" 中的具体值来取行数据（如取"Index"为"A"的行）
- iloc函数：通过行号来取行数据（如取第二行的数据）
- data[data = 'NR' ] = 0
- data.to_numpy()
- 

## Matplotlib
### 介绍
- Matplotlib 是 Python 的一个绘图库。它包含了大量的工具，你可以使用这些工具创建各种图形，包括简单的散点图，正弦曲线，甚至是三维图形
  
### 画图
```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,2*np.pi,100)
plt.plot(x,np.sin(x),'r-^',x,np.cos(x),'g-*')
plt.show()
```

### 使用子图
```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,2*np.pi,100)
plt.subplot(2,1,1)                 # # （行，列，活跃区）
plt.plot(x,np.sin(x),'k')
plt.subplot(2,1,2)
plt.plot(x,np.cos(x),'y')

plt.show()
```


### 散点图
```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,2*np.pi,100)
plt.scatter(x,np.sin(x),np.random.rand(100)*40,'r')                     # 点的size和颜色均可以指定

plt.show()
```

### 直方图
- 使用 hist() 函数可以非常方便的创建直方图。第二个参数代表分段的个数。分段越多，图形上的数据条就越多。
```
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(500)
y = np.random.rand(500)
plt.subplot(2,1,1)
plt.hist(x,50)
plt.subplot(2,1,2)
plt.hist(y,50)

plt.show()
```

### 标题、标签、图例 & 图片保存
```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,np.pi*2,50)
plt.plot(x,np.sin(x),'r', label = 'sin(x)')
plt.plot(x,np.cos(x),'g', label = 'cos(x)')
plt.legend()                                    # 加上这一行就会把上面两行的label显示在图上
plt.xlabel('Rads')
plt.ylabel('Amplitude')
plt.title('picture')

plt.savefig('fig.jpg')                          # 图片保存

plt.show()
```


### Seaborn
- Seaborn 基于 matplotlib， 可以快速的绘制一些统计图表
- 看看老师视频？老师也没具体讲，自己研究吧


## Scikit-learn 
- Python 语言的机器学习工具
- Scikit-learn 包括大量常用的机器学习算法
- Scikit-learn 文档完善，容易上手

### sklearn 表格
<img src="http://imgbed.momodel.cn/q2nay75zew.png" width=800>

由图中，可以看到机器学习 `sklearn` 库的算法主要有四类：分类，回归，聚类，降维。其中：

+ 常用的回归：线性、决策树、`SVM`、`KNN` ；  
    集成回归：随机森林、`Adaboost`、`GradientBoosting`、`Bagging`、`ExtraTrees` 
+ 常用的分类：线性、决策树、`SVM`、`KNN`、朴素贝叶斯；  
    集成分类：随机森林、`Adaboost`、`GradientBoosting`、`Bagging`、`ExtraTrees` 
+ 常用聚类：`k` 均值（`K-means`）、层次聚类（`Hierarchical clustering`）、`DBSCAN` 
+ 常用降维：`LinearDiscriminantAnalysis`、`PCA`   　　

这个流程图代表：蓝色圆圈是判断条件，绿色方框是可以选择的算法，我们可以根据自己的数据特征和任务目标去找一条自己的操作路线。

### sklearn数据集
+ `sklearn.datasets.load_*()`
    + 获取小规模数据集，数据包含在 `datasets` 里
+ `sklearn.datasets.fetch_*(data_home=None)`
    + 获取大规模数据集，需要从网络上下载，函数的第一个参数是 `data_home`，表示数据集下载的目录,默认是 `/scikit_learn_data/`
    
`sklearn` 常见的数据集如下：

||数据集名称|调用方式|适用算法|数据规模|
|--|--|--|--|--|
|小数据集|波士顿房价|load_boston()|回归|506\*13|
|小数据集|鸢尾花数据集|load_iris()|分类|150\*4|
|小数据集|糖尿病数据集|	load_diabetes()|	回归	|442\*10|
|大数据集|手写数字数据集|	load_digits()|	分类|	5620\*64|
|大数据集|Olivetti脸部图像数据集|	fetch_olivetti_facecs|	降维|	400\*64\*64|
|大数据集|新闻分类数据集|	fetch_20newsgroups()|	分类|-|	 
|大数据集|带标签的人脸数据集|	fetch_lfw_people()|	分类、降维|-|	 
|大数据集|路透社新闻语料数据集|	fetch_rcv1()|	分类|	804414\*47236|
```
from sklearn.datasets import load_iris

iris = load_iris()
print("鸢尾花数据集的返回值：\n",iris.keys())
```

```
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target

n_samples, n_features = iris.data.shape

print(n_samples, n_features)
```

### 数据预处理
#### min-max标准化 MinMaxScaler
- 数据标准化和归一化是将数据映射到一个小的浮点数范围内，以便模型能快速收敛。
- 标准化有多种方式，常用的一种是min-max标准化（对象名为MinMaxScaler），该方法使数据落到[0,1]区间：$x^{'}=\frac{x-x_{min}}{x_{max} - x_{min}}$
- MinMaxScaler()：将数据归一到 [ 0，1 ]
- 方法一：先用fit :scaler = preprocessing.MinMaxScaler().fit(X), 得到scaler，scaler里面存着均值和方差；再用transform：scaler.transform(X)，这一步再用scaler中的均值和方差来转换X，使X标准化
- 方法二：fit_transform(X):不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
```
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data

sc = MinMaxScaler().fit(X)
# sc.fit(X)                # 用于计算数据的均值和方差
result1 = sc.transform(X)
result2 = MinMaxScaler().fit_transform(X)

print("放缩前：\n:", X[0])
print("放缩后result1：\n", result1[0])
print("放缩后result2：\n", result2[0])
```

#### Z-score标准化 StandardScaler
- 另一种是Z-score标准化（对象名为StandardScaler），该方法使数据满足标准正态分布：$x^{'}=\frac{x-\overline {X}}{S}$
```
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

result = StandardScaler().fit_transform(X)

print("放缩前：\n:", X[0])
print("放缩后result：\n", result[0])
```


#### 归一化
- 归一化（对象名为Normalizer，默认为L2归一化）：$x^{'}=\frac{x}{\sqrt{\sum_{j}^{m}x_{j}^2}}$
- 归一化和标准化的区别是什么？
```
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer

iris = load_iris()
X = iris.data

result = Normalizer().fit_transform(X)

print("放缩前：\n:", X[0])
print("放缩后result：\n", result[0])
```


#### 二值化
- 使用阈值过滤器将数据转化为布尔值，即为二值化。使用Binarizer对象实现数据的二值化：
- 大于threshold则为1，小于threshold则为0
```
from sklearn.datasets import load_iris
from sklearn.preprocessing import Binarizer

iris = load_iris()
X = iris.data

result = Binarizer(threshold = 3).fit_transform(X)

print("处理前：\n:", X[0])
print("处理后：\n", result[0])
```


#### 标签编码
- 使用 LabelEncoder 将不连续的数值或文本变量转化为有序的数值型变量
```
from sklearn.preprocessing import LabelEncoder

print(LabelEncoder().fit_transform(['apple', 'pear', 'orange', 'banana']))
```

#### 独热编码
- 有几类，数据就有几维
```
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

results = OneHotEncoder().fit_transform(y.reshape(-1,1)).toarray()   # 类别one-hot，共3类

print("处理前：", y)
print("处理后：", results[1])
```

### 数据集的划分
- 鸢尾花数据集共收集了三类鸢尾花，每一类鸢尾花收集了50条样本记录，共150条。数据集包括4个属性，分别为花萼的长、花萼的宽、花瓣的长和花瓣的宽
- 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
-  train_test_split 分割数据集，它默认对数据进行了洗牌
<br>
`sklearn.model_selection.train_test_split(x, y, test_size, random_state )`
   +  `x`：数据集的特征值
   +  `y`： 数据集的标签值
   +  `test_size`： 如果是浮点数，表示测试集样本占比；如果是整数，表示测试集样本的数量。
   +  `random_state`： 随机数种子,不同的种子会造成不同的随机采样结果。相同的种子采样结果相同。
   +  `return` 训练集的特征值 `x_train` 测试集的特征值 `x_test` 训练集的目标值 `y_train` 测试集的目标值 `y_test`。


```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.3,random_state = 22)

print("X_train:",X_train.shape)
print("X_test:",X_test.shape)
print("y_train:",y_train.shape)
print("y_train:",y_train.shape)

print("X_train:",X_train[:2])
print("X_test:",X_test[:2])
print("y_train:",y_train)
print("y_train:",y_train)
```

### 定义模型
#### 估计器（`Estimator`）
估计器，很多时候可以直接理解成分类器，主要包含两个函数：

+ `fit()`：训练算法，设置内部参数。接收训练集和类别两个参数。
+ `predict()`：预测测试集类别，参数为测试集。

大多数 `scikit-learn` 估计器接收和输出的数据格式均为 `NumPy`数组或类似格式。

<br>

#### 转换器（`Transformer`）  
转换器用于数据预处理和数据转换，主要是三个方法：

+ `fit()`：训练算法，设置内部参数。
+ `transform()`：数据转换。
+ `fit_transform()`：合并 `fit` 和 `transform` 两个方法。

<br>

在 `scikit-learn` 中，所有模型都有同样的接口供调用。监督学习模型都具有以下的方法：
+ `fit`：对数据进行拟合。
+ `set_params`：设定模型参数。
+ `get_params`：返回模型参数。
+ `predict`：在指定的数据集上预测。
+ `score`：返回预测器的得分。

鸢尾花数据集是一个分类任务，故以决策树模型为例，采用默认参数拟合模型，并对验证集预测。

```
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size = 0.3,random_state = 22)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
model.predict(X_test)
acc = model.score(X_test,y_test)

print(acc)
```

### 模型评估
#### 交叉验证
- 评估模型的常用方法为 `K` 折交叉验证，它将数据集划分为 `K` 个大小相近的子集（`K` 通常取 `10`），每次选择其中(`K-1`)个子集的并集做为训练集，余下的做为测试集，总共得到 `K` 组训练集&测试集，最终返回这 `K` 次测试结果的得分，取其**均值**可作为选定最终模型的指标。
```
# 交叉验证
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, scoring=None, cv=10)
```

- 注意：由于之前采用了 train_test_split 分割数据集，它默认对数据进行了洗牌，所以这里可以直接使用 cv=10 来进行 10 折交叉验证（cross_val_score 不会对数据进行洗牌）。如果之前未对数据进行洗牌，则要搭配使用 KFold 模块：
```
from sklearn.model_selection import KFold
n_folds = 10
kf = KFold(n_folds, shuffle=True).get_n_splits(X)
cross_val_score(model, X, y, scoring=None, cv = kf)
```


### 保存和加载模型：joblib
- 训练模型后可将模型保存，以免下次重复训练。保存与加载模型使用 sklearn 的 joblib
- 新版本的sklearn已经移除joblib 所以直接import joblib就行
```
#from sklearn.externals import joblib               # 错误
import joblib                                      # 新版本的sklearn已经移除joblib 所以直接import joblib就行
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

joblib.dump(model,'MyModel.pkl')                   # 保存模型
model = joblib.load('MyModel.pkl')                 # 加载模型
print(model)
```

## os 模块
- os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
- os.path.join(path1, path2) 路径拼接合并

## cv2 模块
- cv2.imread()返回np.array类型
- cv2.resize(InputArray src, OutputArray dst, Size, fx, fy, interpolation)

## PIL Image对象
- PIL(Python Image Library)是python的第三方图像处理库，但是由于其强大的功能与众多的使用人数，几乎已经被认为是python官方图像处理库了
- Image类是PIL中的核心类
```
from PIL import Image             #调用库，包含图像类
    im = Image.open("3d.jpg")     #文件存在的路径，如果没有路径就是当前目录下文件
    im.show()
```
### Image.open()和cv2.imread()的区别
- Image.open(）得到的img数据类型呢是Image对象，不是普通的数组。cv2.imread()得到的img数据类型是np.array()类型。
- 具体可以看https://blog.csdn.net/weixin_42213622/article/details/109110140


## 机器学习完整例子示范
### 步骤
- step 1. 获取及加载数据集
- step 2. 数据预处理
- step 3. 划分数据集
- step 4. 定义模型
- step 5. 模型训练
- step 6. 模型预测
- step 7. 模型评估：测试集算准确率或交叉验证

### 解释说明
- LogisticRegression(penalty='l2',solver='newton-cg',multi_class='multinomial')，因为线性回归容易过拟合，所以加上一个正则项penalty='l2'，solver是一个优化器，multi_class='multinomial'，设定是一个多分类
```
from sklearn.datasets import load_iris
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import joblib

# 加载数据集
iris = load_iris()

X = iris.data
y = iris.target

# 数据集标准化
transfer = StandardScaler()
transfer.fit_transform(X)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# 模型构建
estimator = LogisticRegression(penalty='l2',solver='newton-cg',multi_class='multinomial')

# 训练模型
estimator.fit(X_train,y_train)

# 模型评估
y_predict = estimator.predict(X_test)
acc = metrics.accuracy_score(y_predict,y_test)
print("模型准确率为：%.1f%%"%(acc*100))

# 交叉验证
kfscore = cross_val_score(estimator, X, y, scoring = None, cv = 10)
print(kfscore)

# 模型保存
joblib.dump(estimator,'estimator.pkl')
```

## pytorch

### 介绍
- **基于python**的科学计算包，主要服务于以下场景：1. 使用**GPU**的强大计算能力 2. 提供最大的灵活性和高速的深度学习研究平台
- Torch & Numpy：Torch是一个与Numpy类似的Tensor（张量）操作库，与numpy不同的是**Torch对GPU支持的很好**
- Tensors & ndarray: Tensors和Numpy中的ndarray类似，但是在pytorch中tensors可以使用GPU进行计算

### 历史发展
- lua语言：一门比python还简单的语言，简单高效，但过于小众
- torch：使用了lua作为接口
- pytorch：不是简单的封装Lua Torch提供python接口，而是对tensor之上的所有模块进行了重构

### Tensorflow & Pytorch & keras的比较
- tensorflow(google) & pytorch(facebook) & keras的比较：tensorflow比较复杂、市场占有率也比较低，keras太简单，封装得太好，未来如果要去深入的建一些模型不太方便。pytorch的市场占有率目前在50%以上

### 查看pytorch版本
```
import torch as t
print(t.__version__)
```

### 官方文档/官方手册
- https://pytorch.org/docs/master/torch.html

### Tensor 张量

#### Tensor(张量)创建
- 注意函数的参数的形式，与numpy似乎有些区别
- ones & a.new_ones(5,3,dtype=t.float64): new_ones返回一个指定size全为1的tensor，并且默认和a有相同的t.dtype和t.device
- t.rand_like(x, dtype=t.float):返回一个tensor，与x相同size，数值是均匀分布在[0, 1]的随机数，默认其他属性都和x一样例如device、layout、dtype等
- 
```
import torch as t
import numpy as np
a = t.empty(5, 3)                         # 创建一个5*3的矩阵，但是并未初始化

b = t.zeros(5, 3, dtype=t.long)           # 创建一个0填充的矩阵，数据类型为long

c = t.tensor([1, 2, 3])                   # 利用list来生成一个tensor

d1 = t.ones(5, 3)                         # torch:全为1,可以加[]，也可以不加[]
d2 = np.ones([5, 3])                      # torch:全为1，一定要加[]
d3 = d1.new_ones(5, 3, dtype=t.float64)    # 注意是d1.new_ones

print("a:\n", a)
print("b:\n", b)
print("c:\n", c)
print("d1:\n", d1)
print("d2:\n", d2)
```

#### 获取tensor的size
- 注意tensor的size和ndarray的size不一样
- t.size()的返回值是tuple，所以它支持tuple类型的所有操作
- 
```
import torch as t
import numpy as np

x = t.tensor([[1, 2, 3], [2, 3, 4]])
y = np.array([[1, 2, 3], [2, 3, 4]])

print(x.size())                          # size()是函数，返回tensor的大小
print(y.size)                            # size是属性，返回ndarray中的元素数量
```


#### 运算：加减乘除、矩阵乘法、矩阵求逆
- 注意：任何以_为结尾的操作都会用结果替换原变量，例如: x.copy_(), x.t_()

运算|函数
-- | -- 
a + b | t.add(a, b) or t.add(a, b, out)
a - b | t.sub(a, b, alpha)     # output = a - b * alpha
a * b | t.mul(a, b)
a / b | t.div(a, b)
a ** b | t.pow(base, exp)
a % b | t.remainder(a, b)
矩阵乘法 | t.mm(a, b)
矩阵求逆 | a = t.inverse(b)
平均值 | a = b.mean()     # 返回所有元素的平均值，tensor中只有一个元素


#### 改变tensor的维度和大小
- torch.view()
- torch.view与numpy.reshape类似
- flatten：numpy.ndarray.flatten，返回一个一维数组。flatten只能适用于numpy对象，即array或者mat，普通的list列表不适用。a.flatten()：a是个多维数组，a.flatten()就是把a降到一维，默认是按行的方向降


#### 获得数值
- 只有一个元素的张量，可以使用.item()获得python数据类型的数值
```
import torch as t

x = t.randn(1)
print(x)

y = x.item()
print(y)
```

#### ndarray和tensor的互相转换
- tensor和ndarray的相互转换很轻松
- tensor和ndarray共享底层内存地址，修改一个会导致另一个的变化
- 改变tensor的值也会改变ndarray的值
```
import torch as t
import numpy as np

x = t.ones(5, 3)
print(x)

y = x.numpy()              # tensor转化成ndarray，注意函数名称是numpy，不是ndarray
print(y)

a = np.ones([5, 3])
print(a)

b = t.from_numpy(a)        # ndarray转换成tensor，注意函数名称是from_numpy，不是from_ndarray
print(b)

x = x + 1                  # 我发现这个却不改变ndarray的值，可能内存有点不一样？
x.add_(1)                  # 修改x也会改变ndarray的值

print(x)
print(y)
```


#### CUDA 张量
- 所有的tensor类型默认都是基于CPU使用，x.to(device)方法可以将Tensor移动到任何设备中

```
import torch as t

x = t.tensor([[1, 2, 3], [4, 5, 6]])

if t.cuda.is_available():
    device = t.device("cuda:0")
    y = t.ones_like(x, device=device)               # 直接从GPU创建tensor
    x = x.to(device)                                # 将tensor从CPU转移至GPU
    z = x + y
    print(z)
    z = z.to("cpu", t.double)
    print(z)
```

### Autograd 自动求导机制
- 深度学习算法本质上是通过反向传播求导数，Pytorch的Autograd模块实现了此功能。在Tensor上的所有操作，Autograd都能为它们自动提供微分，避免手动计算导数的复杂过程
- Tensor和Function连成计算图，它表示和存储了完整的计算历史
- Pytorch中所有神经网络的核心是autograd包，autograd包为tensor上的所有操作提供了自动求导
- torch.Tensor中若设置require_grad为True，那么将会追踪所有对于该张量的操作
- 当完成计算后通过调用.backward()，将自动计算所有的梯度，这个tensor的所有梯度将会自动积累到.grad属性
- 如果tensor是一个标量，则不需要为backward()指定任何参数，否则，需要制定一个gradient参数来匹配张量的形状
- 要阻止张量跟踪历史记录，可以调用：1. .detach()方法 2. 将代码块包装在with_torch.no_grad()中，在评估模型时特别有用，因为模型可能具有requires_grad = True的可训练参数，但是我们不需要梯度计算


#### autograd.Variable
- pytorch0.4更新后Tensor和Variable合并:torch.Tensor 和torch.autograd.Variable现在是同一个类。torch.Tensor 能够像之前的Variable一样追踪历史和反向传播。Variable仍能够正常工作，但是返回的是Tensor。所以在0.4的代码中，不需要使用Variable了。
- autograd.Variable是Autograd中的核心类，它简单封装了tensor，并支持几乎所有tensor的操作。Tensor在被封装为Variable之后，可以调动它的.backward实现反向传播，自动计算所有梯度
- forward函数的输入和输出都是Variable，只有Variable才具有自动求导功能，Tensor是没有的，所以在输入时，需要把Tensor封装成Variable
- Variable的数据结构，autograd.Variable中包含了data、grad、grad_fn。
- grad也是个Variable，而不是tensor，它和data的形状一样
- Variable和tensor具有近乎一致的接口，在实际使用中可以无缝切换


##### Variable创建
```
import torch as t

x = t.autograd.Variable(t.ones(3, 5), requires_grad=True)

print(x)
```

#### requires_grad = True
- torch.Tensor中若设置require_grad为True，那么将会追踪所有对于该张量的操作

```
import torch as t

a = t.ones(5, 3, requires_grad=True)

print(a)
```

#### 梯度 反向传播
- 注意，grad在反向传播的过程中是累加的，这意味着每次运行反向传播，梯度都会累加之前的梯度，所以反向传播前需要把梯度清零
- grad也是个Variable，而不是tensor，它和data的形状一样。
- 
```
import torch as t

x = t.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

x.grad.data.zero_()    # 梯度清零，grad也是个Variable，而不是tensor，它和data的形状一样
out.backward()         # 反向传播，因为out是一个scalar，out.backward()等于out.backward(torch.tensor(1))

print(x)
print(y)               # y已经被计算出来了，所以，grad_fn已经被自动生成了
print(z)
print(out)
print(x.grad)          # 求出来四个元素均是4.5
```


#### 另一个autograd的例子
- L2范数，欧式距离 torch.sqrt(torch.sum(torch.pow(y, 2)))
- 如果out是一个标量，则不需要为out.backward()指定任何参数，否则，需要制定一个gradient参数来匹配张量的形状
- 对标量输出它才会计算梯度，而求一个矩阵对另一矩阵的导数束手无策。
- 不是很明白这个gradient参数的设置？
```
import torch as t

x = t.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:               # L2范数
    y = y * 2

# print(x)
# print(y)

# y.backward()                   # 错误，y不是一个标量，因此要制定一个gradient参数来匹配张量的形状
# print(x.grad)

# gradients = t.tensor([0.1, 1.0, 0.0001], dtype=t.float)
# y.backward(gradients)                                      # 正确

# print(x.grad)

gradients = t.tensor([1, 1, 1], dtype=t.float)
y.backward(gradients) 

print(x.grad)
```


#### with torch.no_grad()
- 如果requires_grad=True但是你又不希望进行autograd的计算，那么可以将变量包裹在with torch.no_grad()中
```
import torch as t

x = t.randn(3, requires_grad=True)

print(x.requires_grad)
print((x**2).requires_grad)

with t.no_grad():
    print(x.requires_grad)
    print((x**2).requires_grad)
```


### 数据加载与预处理
- Pytorch 读取数据虽然特别灵活，但是还是具有特定的流程的，它的操作顺序为：
  - 创建一个 Dataset 对象，该对象如果现有的 Dataset 不能够满足需求，我们也可以自定义 Dataset`
  - 创建一个 DataLoader 对象
  - 不停的 循环 这个 DataLoader 对象
- 一般情况下，处理图像、文本、音频和视频数据时，可以使用标准的python包来加载数据到一个numpy数组中，然后把这个数组转换成torch.*Tensor
- 图像可以使用Pillow, Opencv
- 音频可以使用scipy，librosa
- 文本可以使用原始python和Cython来加载，或是使用NLTK和SpaCy处理
- 特别地，对于图像任务,可使用torchvision，它包含了处理一些基本图像数据集地方法。这些数据集包括Imagenet,CIFAR10,MNIST等。除了数据加载意以外，torchvision还包含了数据转换器。torchvision.datasets 和 torch.utils.data.DataLoader


#### Dataset
- Dataset对象是一个数据集，可以按下标访问，返回形如(data,label)的数据
- 创建一个 Dataset 对象，该对象如果现有的 Dataset 不能够满足需求，我们也可以自定义 Dataset，通过继承 torch.utils.data.Dataset。在继承的时候，需要 override 三个方法。
    - `__init__`： 用来初始化数据集，定义一个类必须写构造函数。为什么自定义Dataset的init函数不需要继承父类Dataset的__init__()函数？
    - `__getitem__`：给定索引值，返回该索引值对应的数据；它是python built-in方法，其主要作用是能让该类可以像list一样通过索引值对数据进行访问。返回一条数据或一个样本。`obj[index]`等价于`obj.__getitem__(index)`。注意，getitem应该分两种情况，如果是训练数据或是验证数据，就返回data,label。如果是测试数据，就返回data。
    - `__len__`：用于len(Dataset)时能够返回大小，返回样本的数量，`len(obj)`等价于`obj.__len__()`
- dataset中返回的数据类型有规定吗？array？tensor？图像数据是tensor，transform中读取数据后，转换成tensor。至于label，应该是int吧？
```
class MyDataset(torch.utils.data.Dataset):      # 需要继承dataset
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0
```

```
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import torch as t
class MyDataset(Dataset):
    def __init__(self, root):
        # 将所有图片的绝对路径存在self.imgs_path列表中
        # 这里不实际加载图片，只是指定路径
        # 当调用__getitem__时才会真正读照片
        # 不需要继承dataset的构造函数？
        imgs = os.listdir(root)
        # 下面的代码也可精简为 self.imgs_path = [os.path.join(root, img) for img in imgs]
        self.imgs_path = []
        for img in imgs:
            self.imgs_path.append(os.path.join(root, img))
    
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        pil_image = Image.open(img_path)
        array_image = np.asarray(pil_image)
        tensor_image = t.from_numpy(array_image)
        label = xxxxxxxxxx
        return tensor_image,label

    def __len__(self):
        return len(self.imgs_path)
```
```
class MyDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = data.shape[0]
        
    def __getitem__(self, mask):
        label = self.label[mask]
        data = self.data[mask]
        return label, data

    def __len__(self):
        return self.length
```


#### torchvision
- torchvision：自定义的Dataset后，有时候返回的数据可能还是不适合实际使用。例如，返回样本形状大小不一，这对需要取batch训练的神经网络来说很不友好。例如，返回的样本数值较大，未归一化至[-1,1]。针对上述问题，pytorch提供了torchvision。torchvision是一个视觉工具包，提供了很多视觉图像处理的工具，其中torchvision中transforms模块提供了对PIL Image对象和Tensor对象的常用操作
- 对PIL Image对象的常见操作如下：
  - Resize(transforms.Resize):
  - CenterCrop、RandomCrop、RandomSizeCrop：裁剪图片。CenterCrop中心裁剪、RandomCrop随机裁剪、RandomSizeCrop先将给定的PIL Image随机切，再resize成指定的大小
  - Pad：填充
  - ToTensor：将PIL Image对象转成tensor，会自动将[0,255],归一化至[0,1]
- 对Tensro的常见操作如下：
  - Normalize：标准化，即减均值，除以标准差。ToTensor已经归一化至[0,1],T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),可以标准化至[-1,1]
  - ToPILImage：将Tensor转换成PIL Image对象
- 如果要对图片进行多个操作，可通过Compose将这些操作拼接起来，类似于nn.Sequential。
```
from torchvision import transforms as T
transform = T.Compose([
  T.Resize(224)                                       # 保持长宽比不变，最短边为224像素
  T.CenterCrop(224)                                   # 从图片中间切出224*224的图片
  T.ToTensor()                                        # 将图片从图片（Image）转成Tensor，并且归一化至[0,1]
  T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])     # 标准化至[-1,1]
])
```

#### torch.utils.data.DataLoader
- Dataset只负责数据的抽象，一次调用__getitem__只返回一个样本。在训练神经网络时，是对一个batch进行操作，同时还需要对数据进行shuffle和并行加速等。对此，Pytorch提供了DataLoader帮助我们实现这些功能
- 实现功能：对一个batch的数据进行操作，对数据进行shuffle，并行加速
- DataLoader是一个可迭代的对象，它将dataset返回的每一条数据样本拼接成一个batch，并提供多线程加速优化和数据打乱等操作。当程序对dataset的所有数据遍历完一遍之后，对DataLoader也完成了一次迭代
- DataLoader 是 torch 给你用来包装你的数据的工具，所以你要将( numpy array 或其他) 数据形式装换成 Tensor, 然后再放进这个包装器中。 使用 DataLoader 帮助我们对数据进行有效地迭代处理。
- DataLoader的函数定义如下：
```
torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
```
- 常用参数解释：
  - dataset (Dataset): 是一个 DataSet 对象，表示需要加载的数据集
  - batch_size (int, optional): 每一个 batch 加载多少组样本，即指定 batch_size ，默认是 1
  - shuffle (bool, optional): 布尔值 True 或者是 False ，表示每一个 epoch 之后是否对样本进行随机打乱，默认是 False
  - sampler (Sampler, optional): 自定义从数据集中抽取样本的策略，如果指定这个参数，那么 shuffle 必须为 False
  - batch_sampler (Sampler, optional): 与 sampler 类似，但是一次只返回一个 batch 的 indices（索引），需要注意的是，一旦指定了这个参数，那么 batch_size,shuffle,sampler,drop_last 就不能再制定了（互斥）
  - num_workers (int, optional): 这个参数决定了有几个进程来处理 data loading 。0 意味着所有的数据都会被 load 进主进程，默认为0
  - collate_fn (callable, optional): 将一个 list 的 sample 组成一个 mini-batch 的函数（这个还不是很懂）
  - pin_memory (bool, optional): 如果设置为True，那么 data loader 将会在返回它们之前，将 tensors 拷贝到 CUDA 中的固定内存（CUDA pinned memory）中
  - drop_last (bool, optional): 如果设置为 True：这个是对最后的未完成的 batch 来说的，比如 batch_size 设置为 64，而一个 epoch只有 100 个样本，那么训练的时候后面的 36 个就被扔掉了，如果为 False（默认），那么会继续正常执行，只是最后的 batch_size 会小一点。
  - timeout (numeric, optional): 如果是正数，表明等待从 worker 进程中收集一个 batch 等待的时间，若超出设定的时间还没有收集到，那就不收集这个内容。这个 numeric 应总是大于等于0，默认为0。

- DataLoader的函数定义如下：（怎么版本还不一样呢？哪个是最新版本，研究下）
```
DataLoader(dataset, batch_size=1, shuffle=False, sample=None, num_workers=0, collate_fn=default_collate, pin_momory=Flase, drop_last=False)
```
- dataset: 加载的数据集（Dataset对象）
- batch_size: batch size 批大小
- shuffle: 
- sample：样本抽样
- num_workers: 使用多进程加载的进程数，0代表不使用多进程
- collate_fn: 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
- pin_memory: 是否将数据保存在pinmemory去，pin memory中的数据转到GPU会快一些
- drop_last: dataset中的数据个数可能不是batchsize的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
```
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=3, shuffle=True,num_workers=0, drop_last=False)
dataiter = iter(dataloader)
imgs,labels = next(dataiter)
```
- iter() 函数用来生成迭代器。`iter(object[, sentinel])`。object: 支持迭代的集合对象
- next() 返回迭代器的下一个项目。next() 函数要和生成迭代器的 iter() 函数一起使用。
- dataloader是一个可迭代的对象，我们可以像使用迭代器一样使用它，例如：
```
for batch_datas, batch_labels in dataloader:
    train()
```
或
```
dataiter = iter(dataloader)
batch_datas, batch_labels = next(dataiter)
```
- DataLoader怎么版本还不一样呢？确实，目前上面的两个都不是最新版本，我查了官方文档，又增加了新的参数。目前我的水平，了解里面的几个重要参数就行了，后面的再慢慢学习


### torch.nn 神经网络
- torch.nn的核心数据结构是Module，它是一个抽象的概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络层
- nn:专门为神经网络设计的接口，提供了很多有用的功能（神经网络层，损失函数，优化器等）
- nn构建于Autograd之上，可用来定义和运行神经网络
- nn.Module是nn中最重要的类，可以把它看作一个网络的封装，包含网络各层定义及forward方法，调用forward(input)方法，可返回前向传播的结果
- 使用torch.nn包来构建神经网络，nn包依赖autograd包来定义模型并求导。
- 一个nn.Module包含各个层和一个forward(input)方法，该方法返回output
- nn.Module子类的函数必须在构造函数中执行父类的构造函数
- torch.nn只支持mini-batches，不支持一次只输入一个样本，即一次必须是一个btach。如果只想输入一个样本，则用input.unsqueeze(0)将batch_size设为1。例如，nn.Conv2d的输入必须是4维的，形如nSamples*nChannels*Height*Width，可将nSamles设置为1，即 1*nChannels*Height*Width
- 在实际使用中，最常见的做法是继承nn.Module,撰写自己的网络层
- 自定义层必须继承nn.Module，并且包含__init__构造函数和forward函数
- forward函数需要return吗？需要。需要return output。
- forward函数好像只有定义，后面训练没有具体写到？predict_y = model(train_x)其实就是调用了model中的forward方法
- __init__构造函数中，必须调用nn.Module的构造函数，即`super(MyNet, self).__init__()` 或 `nn.Module.__init__(self)`。子类可以在继承父类方法的同时，对方法进行重构。这样一来，子类的方法既包含父类方法的特性，同时也包含子类自己的特性。在单类继承中，super()函数用于指向要继承的父类，且不需要显式的写出父类名称。Python3.x 和 Python2.x 的一个区别是: Python 3 可以使用直接使用 super().xxx 代替 super(MyNet, self).xxx。
```
nn.Module.__init__(self)
super(MyNet, self).__init__()       # python2写法
super().__init__()                  # python3写法
```
 - forward函数实现前向传播
 - 无须写反向传播函数，因其前向传播都是对varible进行操作，nn.Module能够利用autograd自动实现反向传播
 - Conv2d:Conv2d的输入必须是四维的，(batchsize,channels,height,weight)
 - 注意，conv2d中的padding=1，代表图片周围加一层，也就是行列左右各增加1，也就是行列均增加2
 - 注意，transforms.Compose([xx,xx,xx]), nn.Squential(xx, xx, xx),一个要加中括号，一个不用
```
torch.nn.Conv2d(in_channels, out_channels, kernel_size(int or tuple), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
```
 - MaxPool2d:
```
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```
 - BatchNorm2d:批规范化层，分为1D,2D,3D。
```
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```
 - dropout2d:用来防止过拟合，同样分为1D,2D,3D。
 - nn.Linear:Y=X*AT(转置)+b, 假设m = nn.Linear(20, 30), 如果将x维度=(128,20),得到的y维度=(128，30)
```
torch.nn.Linear(in_features, out_features, bias=True)
```
```
class MyNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.cnn = nn.Sequential(
            nn.Conv2d(3,64,3,1),      
            nn.MaxPool2d(2,2,0),
        )
        self.fc = nn.Sequential(
            nn.Linear(),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
```

#### 神经网络的典型训练过程
##### 老师版
- Step1: 定义网络模型：定义包含一些可学习的参数（或者叫权重）神经网络模型
- Step2: 加载数据集和数据预处理：访问数据集，数据集一般都是一个batch一个batch去访问的
- Step3: 输入数据到神经网络
- Step4: 定义损失函数计算损失（输出结果和正确值的差值大小），比如分类用CrossEntropy
- Step5: 反向传播获得梯度
- Step6: 用获得的梯度更新网络的参数：weight = weight - learning_rate * gradient

##### 自己版
- Step1：定义网络模型
- Step2：加载数据集和数据预处理
- Step3：定义损失函数和优化器
- Step4：训练模型
- Step5：评估模型

#### nn.functional & nn.Module
- nn中的大多数layer在functional中都有一个与之相对应的函数。
- nn.Module和nn.functional的主要区别在于，用nn.Moduel实现的layers是一个特殊的类，都是由class Layer(nn.Module)定义，会自动提取可学习的参数。而nn.functional中的函数更像是纯函数，由def function(input)定义。
- 什么时候用nn.Module，什么时候用nn.functional?如果模型有可学习的参数，最好用nn.Module,否则既可以用nn.Module也可以用nn.functional,二者在性能上没有太大差异。由于激活函数、池化等层没有可学习参数，可以使用对应的functional函数代替，而卷积、全连接等具有可学习参数的网络建议使用nn.Module
- 虽然dropout操作也没有可学习的参数，但建议还是使用nn.Dropout而不是nn.functional.dropout,因为dropout在训练和测试两个阶段的行为有所差别，使用nn.Module对象能够通过model.eval操作加以区分
- 不具备可学习参数的层（激活层、池化层等），将它们用函数代替，这样可以不放置在构造函数__init__中。
- 激活函数到底应该写在init里面，还是写在forwad函数里面？不具备可学习参数的层，如激活层，池化层，将它们用函数代替，即用nn.functional中的方法，这样可以不用9放置在构造函数__init__中。如果是nn.Module,就要放在__init__构造函数里面。
```
from torch.nn import functional as F
class MyNet(nn.Module):
def __init__(self):
    super(MyNet, self).__init__()
    self.conv1 = nn.conv2d(3, 64, 3)
    self.conv2 = nn.conv2d(64, 16, 3)
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

def forward(self, x):
    x = F.pool(F.relu(self.conv1(x)), 2)
    x = F.pool(F.relu(self.conv2(x)), 2)
    x = x.view(-1,16*5*5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
```


#### 定义网络模型
- nn.Module子类的函数必须在构造函数中执行父类的构造函数
- Python3.x 和 Python2.x 的一个区别是: Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx :
- 如果要创建一个对象，构造函数会自动被调用起来
- 模型中必须要定义forward函数，backward函数（用来计算梯度）会被autograd机制自动创建
- 现在，如果在反向过程中跟随loss,使用他的grad.fn属性，将看到如下所示的计算图：input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d -> view -> linear -> relu -> linear -> relu -> linear -> MSELoss -> loss
- 所以，当我们调用loss.backward()时，整张计算图都会根据loss进行微分，而且图中所有设置为requires_grad=True的张量将会拥有一个随着梯度累积的.grad张量
- forward()函数中，input首先经过卷积层，此时的输出x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值
- forward函数的输入和输出都是Variable，只有Variable才具有自动求导功能，Tensor是没有的，所以在输入时，需要把Tensor封装成Variable
- 注意，forward方法中，卷积层的输出需要flatten，否则维度不对了。而且，cnn的输出维度是（batch_size,channels,w,h)
```
import torch.nn as nn
import torch.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()           # python2写法
        # super().__init__()                  # python3写法
        # nn.Module.__init__(self)            # 等价写法
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        # x四维，(batchsize, channels, x, y)，因此一共是batchsize*channels*x*y
        x.view(x.size()[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                  # 最后一个全连接层不需要激活函数？
        return x                         # 需要return？


net = Net()
print(net)
```


#### 网络中的可学习参数
- 把网络中具有可学习参数的层放在构造函数__init__()中，如果某一层，如激活函数ReLU，不具有可学习参数，则既可以放在构造函数中，也可以不放
- 网络的可学习参数通过net.parameters()返回，net.named_parameters()可同时返回可学习的参数及名称
- net.parameters()返回可被学习的参数（权重）列表和值
- 这部分的理解和使用还比较不会？
```
parameters = list(net.parameters())
print(len(parameters))

for name, parameters in net.named_parameters():
    print(name, ":", parameters.size())
```


#### 损失函数
- Pytorch将损失函数实现为nn.Module的子类
- 一个损失函数接收一对(output,target)作为输入，计算一个值来估计网络输出和目标值相差多少
- nn包中有很多不同的损失函数。nn.MSEloss是一个比较简单的损失函数，它计算输出和目标之间的均方误差。分类问题做最小二乘法的收敛性会很差，所以一般分类问题很少用最小二乘法来做
```
import torch.nn as nn

criterion = nn.MSEloss()
loss = criterion(output, target)
```
- 损失函数：在深度学习中要用到各种各样的损失函数，这些损失函数可以看作是一种特殊的layer，Pytorch也将这些损失函数定义为nn.Module的子类。然而在实际使用中，通常将这些损失函数专门提取出来，作为独立的一部分。
- 注意，criterion(predict_y, label)，注意参数的顺序，不可以反过来



#### torch.optim 优化器：更新权重 
- 优化器：在反向传播计算完所有参数的梯度后，还需要使用优化方法更新网络的权重和参数。Pytorch将深度学习常用的优化方法全部封装在torch.optim中，其设计十分灵活，能够很方便的扩展成自定义的优化方法。torch.optim中实现了深度学习中绝对多数的优化方法，例如RMSProp、Adam、SGD等。所有的优化方法，包括自定义和非自定义的优化方法，都是继承基类(父类）optim.Optimizer,并实现了自己的优化步骤。
- 注意，grad在反向传播的过程中是累加的，这意味着每次运行反向传播，梯度都会累加之前的梯度，所以反向传播前需要把梯度清零
```
import torch.optim as optim
# 新建一个参数，指定要调整的参数和学习率
optimizer = optim.SGD(MyNet.parameters(), lr=0.01)

```
- MyNet.parameters()：返回网络的可学习参数
- MyNet.named_parameters()：同时返回可学习的参数及名称
- optimizer.step()：更新参数
- Pytorch将常用的深度学习优化方法全部封装在torch.optim中，其设计十分灵活，能够很方便的扩展成自定义的优化方法。所有的优化方法都是继承基类optim.Optimizer,并实现了自己的优化步骤
- 权重更新规则是随机梯度下降（SGD）：weight = weight - learning_rate * gradient
- 可以使用简单的python代码实现这个规则：
```
learn_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```
- 但是当神经网络想要使用不同的更新规则时，比如SGD\Nesterov-SGD、Adam、RMSPROP等，pytorch中构建了一个包torch.optim实现了所有的这些规则，使用它们非常简单
```
import torch.optim as optim

# 新建一个优化器，指定要调整的参数和学习率
optimizer = optim.SGD(net.parameters(),lr=0.01)

optimizer.zero_grad()                 # 与net.zero_grad() 效果一样

output = net(input)
loss = criterion(output, target)
loss.backward()

optimizer.step()
```


#### 训练网络
- 多进程需要在main函数中运行，因此当num_workers设定大于1时，需要在训练时加上if __name__=='__main__':
- enumerate()函数：enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。返回 enumerate(枚举) 对象。enumerate(sequence, [start=0]) ，第二个参数表示开始的索引
- model.train()和model.eval()如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
```
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 输入数据
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        # 梯度清零
        optimizer.zero_grad()
        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # 更新参数
        optimizer.step()
        # 打印log信息
        running_loss += loss.data.item()
        if i % 2000 == 1999:
            print('[%d %5d] loss:%.3f' % (epoch+1, i + 1, running_loss / 2000))
            running_loss = 0.0
print("Finish Training!")
```

#### GPU加速：cuda
- 在Pytorch中，Tensor、nn.Module、Variable(包括Parameter)均分别为CPU和GPU两个版本
- Tensor、nn.Module、Variable(包括Parameter)都带有.cuda方法，调用.cuda方法即可将其转为对应的GPU对象
- 注意，tensor.cuda和variable.cuda都会返回一个新对象，这个新对象的数据已经转移至GPU，而原来的tensor、variable的数据还在原来的CPU设备上。
- module.cuda会将所有数据迁移至GPU，并返回自己。所以module.cuda和module = modlue.cuda的效果相同
- variable和nn.Module在GPU和CPU之间的转换，本质上还是利用了Tensor在CPU和GPU的转换。
- nn.Module的cuda方法是将nn.Module下的所有parameter（包括子Module的parameter）都转移至GPU。
- 为什么将数据转移至GPU的方法叫做.cuda而不是.gpu呢？因为GPU的编程接口采用CUDA，而目前并不是所有的GPU都支持cuda，只有部分NVIDIA的GPU才支持。Pytorch未来可能会支持AMD的GPU，而AMD的GPU编程接口采用OpenCL，因此Pytorch还预留着.cl方法，用于支持AMD的GPU
```
new_tensor = tensor.cuda(0)    # 返回了new_tensor，保存在第0块GPU上，tensor还是在GPU上。
new_tensor = tensor.cuda()     # 如果不写参数，则默认使用第0块。
```
- 大部分的损失函数也属于nn.Module，但在使用GPU时，很多时候我们都忘记使用它的.cuda方法，在大多数情况下不会报错，因为损失函数本身没有可以学习的参数。但在某些情况下会出现问题，为了保险期间同时也为了代码更规范，应记得调用criterion.cuda
```
criterion = nn.CrossEntropyLoss()
criterion.cuda()
loss = criterion(predict_y, label)
```

#### 验证
- 验证相对来说比较简单，但要注意需将模型置于验证模式model.eval(), 验证完成后还需要将其置灰训练模式model.train()。这两句代码会影响BatchNorm和Dropout等层的运行模式。



### 示例：训练一个图像分类器：CIFAR-10
- Step1: 使用torchvision加载和归一化CIFAR-10训练集和测试集
- Step2: 定义一个卷积神经网络
- Step3: 定义损失函数
- Step4: 在训练集上训练网络
- Step5: 在测试集上测试网络

```
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

# 数据加载和归一化
transform = transforms.Compose([
    transforms.ToTensor(),                                                # 把PIL Image对象转换为Tensor对象
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = tv.datasets.CIFAR10(root='./dataset/', train=True,
                               transform=transform, download=True)
testset = tv.datasets.CIFAR10(root='./dataset/', train=False,
                              transform=transform, download=True)

trainloader = DataLoader(trainset, batch_size=6, shuffle=True)
testloader = DataLoader(testset, batch_size=6, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.8)             # momentum 动量因子


# 训练网络
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):                      # 这里是一个batch一个batch吗
        # 输入数据
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)         # input是4维(batchsize,channels,x,y)
        # 梯度清零
        optimizer.zero_grad()
        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # 更新参数
        optimizer.step()
        # 打印log信息
        running_loss += loss.data.item()
        if i % 2000 == 1999:
            print('[%d %5d] loss:%.3f' % (epoch+1, i + 1, running_loss / 2000))
            running_loss = 0.0
print("Finish Training!")

```



## math 模块
- math.floor(x) 返回数字x的下舍整数，即向下取整
- math.round(x) 四舍五入
- math.ceil(x) 向上取整



## python语法
- with open() as f:有一些任务，可能事先需要设置，事后做清理工作。如果不用with语句,一是可能忘记关闭文件句柄；二是文件读取数据发生异常，没有进行任何处理。这时候就是with一展身手的时候了。除了有更优雅的语法，with还可以很好的处理上下文环境产生的异常。
- a.strip():Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。返回移除字符串头尾指定的字符生成的新字符串。
- a.split(): Python split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串。返回分割后的字符串列表。
- next(f): 跳过一行，返回值也是跳过一行的f，f本身也是跳过一行了
- f.readline()：无参数返回表头，有参数，参数的大小表示字节数
- format用法:相对基本格式化输出采用‘%’的方法，format()功能更强大，该函数把字符串当成一个模板，通过传入的参数进行格式化，并且使用大括号‘{}’作为特殊字符代替‘%’.1、基本用法:（1）不带编号，即“{}”,（2）带数字编号，可调换顺序，即“{1}”、“{2}”,（3）带关键字，即“{a}”、“{tom}”
- list.append(element), 列表中增加一个元素，例如 fruit.append('apple')
- zip():zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。


## Q&A
1. 如何处理csv文件？

2. list和numpy.ndarray的区别？
3. pip? pip 是 Python 包管理工具，该工具提供了对Python 包的查找、下载、安装、卸载的功能
4. %matplotlib inline：使用jupyter notebook 或者 jupyter qtconsole的时候，才会用到%matplotlib
5. np.random.randn() & np.random.rand() ,括号中的参数表示大小，一个参数np.random.randn(15)表示15个元素，两个参数np.random.randn(15,4)表示15*4的二维数组
| | np.random.randn() | np.random.rand() |
|--|--|--|
|特点|标准正态分布|均匀分布于[0, 1)中|
6. from xxx import yyy  ,x y都是代表函数还是类？
7. result = StandardScaler().fit_transform(X)   函数返回对象的.fit_transform(X)
8. 归一化和标准化的区别是什么？
9. random_state:random_state：是随机数的种子。随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。参数test_size：如果是浮点数，在0-1之间，表示test set的样本占比；如果是整数的话就表示test set样本数量。test_size只是确定training set于test set的各自所占比例或者数量，并没有确定数据的划分规则。比如我们有数据集[1,2,3,4,5,6,7,8,9],我们确定test_size=3,那问题是我们应该取哪三个数作为test set呢，这时候就应该使用random_state来确定我们的划分规则，假设我们取random_state=1，它按一定的规则去取出我们的数据，当我们random_state=2时，它又换成另一种规则去取我们的数据，random_state的取值范围为0-2^32。当random_state=None或0时，可以理解为随机分配一个整数给random_state，这样就导致每次运行的结果都可能不同。其他函数中的random_state参数功能是类似的。
10. pkl和pth:目前暂时不需要知道他们的区别，当作是差不多的东西就行了
11. 数据集中的关键字frame是什么？
12. metrics.accuracy_score和estimator.score()有什么区别：有三种不同的方法来评估一个模型的预测质量：
estimator的score方法：sklearn中的estimator都具有一个score方法，它提供了一个缺省的评估法则来解决问题。分类算法必须要继承ClassifierMixin类， 回归算法必须要继承RegressionMixin类，里面都有一个score
()方法。score(self, X, y_true)函数会在内部调用predict函数获得预测响应y_predict，然后与传入的真是响应进行比较，计算得分。使用estimator的score函数来评估模型的属性，默认情况下，分类器对应于准确率：sklearn.metrics.accuracy_score， 回归器对应于均方差： sklearn.metrics.r2_score。
Scoring参数：使用cross-validation的模型评估工具，依赖于内部的scoring策略。指定在进行网格搜索或者计算交叉验证得分的时候，使用什么标准度量'estimator'的预测性能，默认是None，就是使用estimator自己的score方法来计算得分。我们可以指定别的性能度量标准，它必须是一个可调用对象，sklearn.metrics不仅为我们提供了一系列预定义的可调用对象，而且还支持自定义评估标准
Metric函数：metrics模块实现了一些函数，用来评估预测误差。
13. score里面的参数也不一样
14. Uber的pyro做一些统计模型很不错，老师推荐大家去学一下
15. cuda
16. 为什么python有时候不显示函数的参数定义？因为python是脚本语言，就是有些库是你运行起来后python才去找的，所以你在写代码的时候，他不一定知道你的库文件在哪，而C/C++你写的时候就要指定你的.h文件在哪里，他就会去找，找到就有提示找不到就会报错
17. 测试集标准化是用训练集的均值和方差？hw2里面是这样的？?
18. 为什么使用crossentropy？什么时候使用crossentropy？
19. 为什么weight初始化为0，那不是没有了么？逻辑回归中可以初始化为0，why？回归中好像也可以，作业范例是初始化为0。神经网络中不行。好像是因为隐藏层。具体原因还没很明白，之后研究一下。


# Homework1: Regression

## 流程及注意事项

1. 读取原始数据
- encoding = 'big5'
- 注意表格是否有表头，默认有表头，header=None表示表格无表头

2. 处理原始数据：NR->0, 提取表格中的数据，即去除非必要的行或列
- 需要把dataframe转换成numpy

3. 准备模型需要的x和y
- 指定ndarray的dtype，虽然empty的dtype默认是float，但指定一下说明考虑全面

4. 数据预处理：对x和y进行标准化
- 标准化的目的是什么？标准化是对哪一个区域的数据进行标准化？是对同一feature的所有样本数据标准化，还是对一个样本的所有feature标准化？应该是对一个样本的所有feature标准化，这样目的是能够使得一个样本的所有feature在同一标准下，不会被数值很大的feature影响
- mean和std是所有样本用同一个，还是不同的样本计算不同的mean和std？
- 训练数据的标准化和测试数据的标准化一样吗？还是测试数据应该使用训练数据的mean和std？哪个更加准确？

5. 建立模型

6. 定义损失函数

7. 找出最小化损失函数的解：梯度下降
- w和b可以初始化为0吗？可以，因为第一个样本丢进模型虽然预测结果为0，但是由损失函数算出来的梯度取决于x，因此梯度算出来不为0，因此从第二个样本开始就预测结果就不为0了。但是神经网络不行，因为神经网络具有隐藏层，为什么？等后面做神经网络的作业再研究
- b是怎么处理的？不仅对w求梯度，也对b求梯度，但注意对w和对b求微分，w后面*x，b后面*1
- x_train_data本来维度是[12*20*15, 18*9], 因为b的原因，应该再加一维1（因为b的系数是1），也可以另外重新定义，不在x_train_data上直接拓展
- adagrad中的gradient的平方和，是w和b一起的gradient的平方和，还是w和b分开算？每个梯度都有自己的adagrad，本题中有18*9+1个梯度，因此也有18*9+1个adagrad
- 可以用adagrad
- eps, 防止除以0
- 注意是y' - y,若写成y-y'(y'=wx+b即预测值，y是真实值),那么取微分后应该还要*-1，因为是对w和b求微分
- 保存最后的w和b的系数
```
np.save('./model/w_b.npy', w)
w = np.load('./model/w_b.npy')
```


1. 测试
- 注意表格是否有表头

9. 保存结果
- python 使用 with open() as 读写文件 https://blog.csdn.net/xrinosvip/article/details/82019844
- newline参数的意思？好绕。。
```
import csv
with open('mysubmit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), predict_test_data[i]]
        csv_writer.writerow(row)
```

## Q&A
1. 标准化后，整体的loss反而增加了？为什么？
1. 如何使用验证集？


# Homework2: Classification - Logistic Regrassion

## 流程及注意事项

1. 读取数据
- 我用的homework1读取数据的方法（pandas），但是第二个作业用的是另一种读取csv文件的方法（with open as)，也可学习参考
- 记得从dataframe转换成numpy
- 数据类型转换成float

2. 数据处理
- 标准化：对每一列，即每一个feature进行标准化，(x-mean)/(std+eps)

3. 训练模型
- shuffle:训练、验证、测试都要shuffle吗？训练要吧,验证和测试应该不用。为什么要shuffle？原始的数据，在样本均衡的情况下可能是按照某种顺序进行排列，如前半部分为某一类别的数据，后半部分为另一类别的数据。但经过打乱之后数据的排列就会拥有一定的随机性，在顺序读取的时候下一次得到的样本为任何一类型的数据的可能性相同。如何shuffle？作业范例中采用这样的方式，不是完全明白，可以再理解理解。shuffle作业范例中是在训练中shuffle的，我是在数据处理时shuffle的，现在想想应该是要在每一轮的训练中，这样每次训练新一轮重新shuffle一遍更好。
```
def _shuffle(X, Y):
    randomsize = np.arange(X.shape[0])
    np.random.shuffle(randomsize)
    return (X[randomsize], Y[randomsize])             # 这里为什么要加小括号？
```
- mini-batch：每一轮都会过一遍全部的训练数据，在每一轮中，一个batch一个batch进行梯度下降。为什么要用batch？当数据集比较大的时候，一次性将所有样本输入去计算一次cost存储会吃不消，因此会采用一次输入一定量的样本来进行训练。mini-batch属于随机梯度下降和批量梯度下降的折中。
- b的处理：计算b的梯度是，记得要乘上全是1的矩阵，因为每个样本对应的b的系数都是1
- lr的处理：使用adagrad来调节lr，作业范例中使用的是step来调节。
- 验证集：验证集可以计算acc
- acc：acc如何计算？我用的方法和作业范例使用的方法不一样。作业范例的方法如下。既可以计算训练集的acc，也可以计算验证集的acc。每一轮遍历完一遍所有数据就输出一次acc。
```
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc
```
- 为什么会溢出？np.log(x)和np.exp(z)需要考虑溢出的状况。如何防止溢出？看以下作业范例，使用np.clip(x, min, max)
```
def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))
```

4. 画图
- 要先保存再show，否则保存的图片是空白的。为什么？在 plt.show() 后调用了 plt.savefig() ，在 plt.show() 后实际上已经创建了一个新的空白的图片（坐标轴），这时候你再 plt.savefig() 就会保存这个新生成的空白图片。

5. 测试并保存结果
- 将算出的test_y转换成dataframe，然后再存储成csv

## 我的遗漏考虑
1. 标准化
2. shuffle
3. mini-batch
4. 尽量使用adagrad
5. 结果画图呈现

## Q&A
1. 给的训练数据和测试数据都是经过处理的，但是如何处理这些数据我还不会。
- 为什么要balance positve 和 negative?
- 如何处理连续数据？
- 如何处理离散数据？
- 哪些attribute是unnecessary的attribute？
2. 梯度的求和到底是哪些项的和？懵逼


# Homework2: Classification - Logistic Regrassion

## 流程及注意事项

1. 读取数据

2. 处理数据：标准化
- 概率生成模型不需要shuffle

3. 划分数据：划分成两个类别
```
train_0 = np.array([x for x, y in zip(train_x,raw_train_y) if y == 0])
train_1 = np.array([x for x, y in zip(train_x,raw_train_y) if y == 1])
```

4. 计算mean0、mean1、conv0、conv1、shared_conv
- 如何求每一类别的平均数？data是x*510,其中510是features，因此mean的维度是1*510，conv的维度是510*510
- shared_conv 按照两个类别的比例计算   shared_conv = class0比例*conv0 + class1比例*conv1
- 如何计算协方差矩阵？numpy有自带的协方差矩阵计算，但是我试了下，样本太大，计算不了，内存不够，没办法计算这么大的矩阵。作业范例中是自己算的，不是调用numpy函数，但我还不太明白是怎么算的
- 通过训练集可以求出mean0、mean1、conv0、conv1、shared_conv，就可以直接求出w和b，不需要训练了


1. 计算w和b，利用mean0、mean1、shared_conv
- 计算w和b会用到shared_conv的逆矩阵。求逆矩阵可以使用np.linalg.inv(a),但是协方差矩阵shared_conv不一定可以求逆矩阵，若不能求逆矩阵，就求伪逆。shared_conv很可能是不可逆矩阵（奇异矩阵），因此要用SVD。Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.Via SVD decomposition, one can get matrix inverse efficiently and accurately.
- 奇异值分解SVD：np.linalg.svd(a, full_matrices=True, compute_uv=True) 。a : 是一个形如(M,N)矩阵。full_matrices：的取值是为0或者1，默认值为1，这时u的大小为(M,M)，v的大小为(N,N) 。否则u的大小为(M,K)，v的大小为(K,N) ，K=min(M,N)compute_uv：取值是为0或者1，默认值为1，表示计算u,s,v。为0的时候只计算s。
- singular matrix是奇异矩阵的意思。奇异矩阵是不可逆矩阵。 设A为n阶方阵，若存在另一n阶方阵B，使得AB=BA=I，则称A为非奇异矩阵，若不存在，则为奇异矩阵。 当exogenous variable 中虚拟变量过多，可能产生singular matrix或near singular matrix，表示这些变量间存在较大相关性。

6. 测试集
- 直接用求出的w和b计算predic_y即可得到结果

## Q&A

2. 概率生成模型不需要验证集？为什么？因為 generative model 有可解析的最佳解，因此不必使用到验证集
3. np.dot()不能连续三个矩阵相乘？不行的，2个2个写
4. 如何计算协方差矩阵？numpy有自带的协方差矩阵计算，但是我试了下，样本太大，计算不了，内存不够，没办法计算这么大的矩阵。作业范例中是自己算的，不是调用numpy函数，但我还不太明白是怎么算的


# Homework3

1. 自定义Dataset


2. 处理数据


3. 加载数据Dataloader


4. 定义神经网络


5. 定义损失函数


6. 定义优化器


7. 训练数据


## Q&A
1. 为什么自定义Dataset的init函数不需要继承父类Dataset的__init__()函数？
2. 梯度清零。为什么要梯度要设计成累加的呢？这样才多了个梯度清零的动作。没弄懂里面的数学算式。
5. 最后一层神经元还要再加ReLU?还是加Softmax激活函数？我看作业范例里面最后一层全连接层并没有加激活函数？
7.  Dropout/BatchNorm


# Homework4

## gensim

- 一款开源的python第三方工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。主要用于主题建模和文档相似性处理，它支持包括TF-IDF，LSA，LDA，和word2vec在内的多种主题模型算法。Gensim在诸如获取单词的词向量等任务中非常有用。
- 使用Gensim训练Word2vec的训练步骤：
1. 将语料库预处理：一行一个文档或句子，将文档或句子分词（以空格分割，英文可以不用分词，英文单词之间已经由空格分割，中文语料需要使用分词工具进行分词，常见的分词工具有StandNLP、ICTCLAS、Ansj、FudanNLP、HanLP、结巴分词等）；
2. 将原始的训练语料转化成一个sentence的迭代器，每一次迭代返回的sentence是一个word（utf8格式）的列表。可以使用Gensim中word2vec.py中的LineSentence()方法实现；
3. 将上面处理的结果输入Gensim内建的word2vec对象进行训练即可：
```
from gensim.models import word2vec

word2vec.Word2Vec(sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(), max_final_vocab=None)
```
- sentences：可以是一个list，对于大语料集，建议使用BrownCorpus,Text8Corpus或lineSentence构建。
- size：是指词向量的维度，默认为100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
- window：窗口大小，即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为c。window越大，则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间。个人理解应该是某一个中心词可能与前后多个词相关，也有的词在一句话中可能只与少量词相关（如短文本可能只与其紧邻词相关）。
- min_count: 需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。可以对字典做截断， 词频少于min_count次数的单词会被丢弃掉。
- negative：即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。
- cbow_mean: 仅用于CBOW在做投影的时候，为0，则算法中的为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示，默认值也是1，不推荐修改默认值。
- iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
- alpha: 是初始的学习速率，在训练过程中会线性地递减到min_alpha。在随机梯度下降法中迭代的初始步长。算法原理篇中标记为η，默认是0.025。
- min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha, min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。
- max_vocab_size: 设置词向量构建期间的RAM限制，设置成None则没有限制。
- sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。
- seed：用于随机数发生器。与初始化词向量有关。
- workers：用于控制训练的并行数。


