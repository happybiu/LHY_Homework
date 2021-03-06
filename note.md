<!-- TOC -->

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
  - [pad填充](#pad填充)
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
  - [标题、标签、图例 & 图片保存](#标题标签图例--图片保存)
  - [散点图](#散点图)
  - [直方图](#直方图)
  - [Seaborn](#seaborn)
- [Scikit-learn](#scikit-learn)
  - [sklearn 表格](#sklearn-表格)
  - [sklearn数据集](#sklearn数据集)
- [鸢尾花例子](#鸢尾花例子)
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
  - [保存模型和加载模型：joblib](#保存模型和加载模型joblib)
- [os模块](#os模块)
- [cv2模块](#cv2模块)
- [PIL: Image对象](#pil-image对象)
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
    - [类型转换](#类型转换-1)
    - [tensor 拼接](#tensor-拼接)
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
    - [torch.nn.init](#torchnninit)
    - [nn.functional & nn.Module](#nnfunctional--nnmodule)
    - [定义网络模型](#定义网络模型)
    - [网络中的可学习参数](#网络中的可学习参数)
  - [损失函数](#损失函数)
  - [torch.optim 优化器：更新权重](#torchoptim-优化器更新权重)
  - [训练网络](#训练网络)
  - [GPU加速：cuda](#gpu加速cuda)
  - [验证](#验证)
- [范例](#范例)
  - [神经网络的典型训练过程](#神经网络的典型训练过程)
    - [老师版](#老师版)
    - [自己版](#自己版)
  - [示例：训练一个图像分类器：CIFAR-10](#示例训练一个图像分类器cifar-10)
- [math 模块](#math-模块)
- [python语法](#python语法)
- [Q&A](#qa)
- [python 工程目录组织/项目完整开发流程](#python-工程目录组织项目完整开发流程)
  - [网上找的](#网上找的)
  - [pytorch书上的猫狗实战](#pytorch书上的猫狗实战)
  - [补充说明](#补充说明)
- [环境变量/工作目录/当前路径](#环境变量工作目录当前路径)
- [*args和**kwargs](#args和kwargs)
  - [*args](#args)
  - [**kwargs](#kwargs)
  - [Q&A](#qa-1)
- [Homework1: Regression](#homework1-regression)
  - [流程及注意事项](#流程及注意事项)
  - [Q&A](#qa-2)
- [Homework2: Classification - Logistic Regrassion](#homework2-classification---logistic-regrassion)
  - [流程及注意事项](#流程及注意事项-1)
  - [我的遗漏考虑](#我的遗漏考虑)
  - [Q&A](#qa-3)
- [Homework2: Classification - Logistic Regrassion](#homework2-classification---logistic-regrassion-1)
  - [流程及注意事项](#流程及注意事项-2)
  - [Q&A](#qa-4)
- [Homework3: CNN](#homework3-cnn)
  - [Q&A](#qa-5)
- [Homework4: RNN](#homework4-rnn)
  - [模型](#模型)
    - [gensim.models.word2vec](#gensimmodelsword2vec)
    - [nn.Embedding():](#nnembedding)
    - [nn.LSTM():](#nnlstm)
  - [流程及注意事项](#流程及注意事项-3)
  - [Q&A](#qa-6)
- [Homework5:](#homework5)
  - [task 1](#task-1)
  - [task 2](#task-2)
  - [task 3](#task-3)
  - [Q&A](#qa-7)
  - [注意事项](#注意事项)
- [Homework6: Attack and Defense](#homework6-attack-and-defense)
  - [Q&A](#qa-8)
- [homework7: Network Compression](#homework7-network-compression)
  - [Architecture Design](#architecture-design)
    - [Q&A](#qa-9)
  - [Knowledge Distillation](#knowledge-distillation)
  - [Network Pruning](#network-pruning)
  - [Q&A](#qa-10)
  - [Weight Quantization](#weight-quantization)
- [homwork8:seq2seq](#homwork8seq2seq)
  - [json 模块](#json-模块)
  - [re模块](#re模块)
  - [seq2seq模型](#seq2seq模型)
  - [BLEU score](#bleu-score)
  - [RNN输入维度和输出维度](#rnn输入维度和输出维度)
  - [要点](#要点)
- [Q&A](#qa-11)
- [Homework9:Unsupervised Learning](#homework9unsupervised-learning)
  - [要点](#要点-1)
  - [Q&A](#qa-12)
- [Homework10：Anomaly Detection](#homework10anomaly-detection)
  - [KNN（K Nearest Neighbour)](#knnk-nearest-neighbour)
    - [要点](#要点-2)
  - [PCA](#pca)
  - [Auto-Encoder](#auto-encoder)
  - [Q&A](#qa-13)

<!-- /TOC -->

<br>
<br>
<br>

# numpy

## 介绍
- NumPy（Numerical Python）是一个开源的 Python 科学计算库，NumPy 支持常见的数组和矩阵操作，并且可以处理任意维度的数组（Tensor）。三维以上的数组就叫做tensor对于同样的数值计算任务，使用 NumPy 比直接使用 Python 要简洁的多。
- NumPy 通常与 SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用。Matplotlib 是 Python 编程语言及其数值数学扩展包 NumPy 的可视化操作界面。

## ndarray的属性
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

## ndarray的数据类型
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

## 基本操作
### 创建ndarray：
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

### range & np.arange & np.linspce
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

### 形状修改
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

### 转置
- 三种方法：transpose方法、T属性以及swapaxes方法
- a.T: 适用于二维数组
- 高维数组用transpose
```
image_list = np.transpose(image_list, (0, 3, 1, 2))
image_list = image_list.transpose((0, 3, 1, 2))
```
- swapaxes：

### 类型转换 
```
a = np.array([[[1, 2, 3], [4, 5, 6]], [[12, 3, 34], [5, 6, 7]]])
print(a.dtype)
b = a.astype(np.float32)                               # 不会修改a中元素的类型，而是另外返回一个类型修改后的ndarray
print(a.dtype)
print(b.dtype)
```



### 数组去重
- unique是numpy模块下的函数，不是numpy.ndarray模块下的函数

```
x = np.array([[1,2,3],[2,3,4],[3,4,5]])

y = np.unique(x)              # 正确
y = x.unique()                # 错误
print("y:\n",y)
```

## pad填充
- np.pad(array, pad_width, mode, **kwargs)方法返回：填充后的数组。array：表示需要填充的数组；pad_width：表示每个轴（axis）边缘需要填充的数值数目。mode：表示填充的方式（取值：str字符串或用户提供的函数）mode中的填充方式：
   - constant: 表示连续填充相同的值，每个轴可以分别指定填充值，constant_values=（x, y）时前面用x填充，后面用y填充，缺省值填充0
   - edge:表示用边缘值填充
   - linear_ramp: 表示用边缘递减的方式填充
   - maximum: 表示最大值填充
   - mean: 表示均值填充
   - median: 表示中位数填
   - minimum: 表示最小值填充
   - reflect: 表示对称填充
   - symmetric: 表示对称填充
   - wrap: 表示用原数组后面的值填充前面，前面的值填充后面


### 数组运算
- 数组的算术运算是元素级别的操作，新的数组被创建并且被结果填充。
  
运算|函数
-- | -- 
a + b | np.add(a, b)
a - b | np.subtract(a, b)
a * b | np.multiply(a, b)<br>np.multiply(a, b, a)：表示将结果传入第三个参数<br>
a / b | np.divide(a, b)
a ** b | np.power(a, b)
a % b | np.remainder(a, b)

## 矩阵mat (尽量别用)
- 注意：numpy虽然现在还能用matrix，但未来不会再用了，因为matrix只能表示2维数组，不像tensor一样可以表示多维数组，所以尽量不要用matrix类型，现在用矩阵的地方尽量用ndarray来表示

### mat和ndarray在处理乘法时的比较
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

### 矩阵的创建
- 矩阵只能是2维
- 可以使用 mat 方法将 2 维数组转化为矩阵
- 也可以使用 **Matlab** 的语法传入一个字符串来生成矩阵
```
import numpy as np
 
a = np.mat("1,2,3;4,5,6;7,8,9")         # 字符串最后不需要加分号

print(a)
```

### 矩阵的逆
- A.I 表示 A 矩阵的逆矩阵
- A必须要可逆，否则会报错：numpy.linalg.LinAlgError: Singular matrix
- 矩阵才有逆矩阵，二维数组没有逆矩阵，会报错：AttributeError: 'numpy.ndarray' object has no attribute 'I'

### 矩阵连乘
- 矩阵指数表示矩阵连乘，A ** 4


## 统计函数
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



## 比较和逻辑函数
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

## all & any
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

## 其他操作
- np.round(x,n):round() 方法返回浮点数x的四舍五入值。x表示数值，n表示四舍五入到小数点哪一位。n=-1表示个位，n=0表示小数点第一位
- np.abs()  绝对值

## IO操作
### 写入
- np.savetxt('路径+文件名',data) 可以将数组写入文件，默认使用科学计数法的形式保存。如果没有路径，则默认存在当前文件夹
```
import numpy as np

a = np.array([[ 0, 1, 2, 3, 4, 5],
           [10,11,12,13,14,15],
           [20,21,22,23,24,25],
           [30,31,32,33,34,35]])

np.savetxt("newout.text",a)
```

### 读取
- open('文件名'),其余参数均有默认值
- open(name[], mode[], buffering[]),mode默认为只读
- np.loadtxt('文件名'),其余参数均有默认值
np.loadtxt(fname, dtype=, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
- np.load('./xxx.npy'):

### numpy 其他操作
- np.concatenate((a,b,c),axis=0)一次完成多个数组的拼接 axis=0表示行连接，axis=1表示列的数组进行拼接
- np.dot() 点乘?

# Pandas
## 介绍
- https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html
- numpy主要是用来处理数组数据，pandas也是用来处理数组数据，但pandas更多的是用来做一些数据分析，特别是去读一些像是csv文件，做一些统计分析
- Pandas 是基于 NumPy 的一种工具,该工具是为了解决**数据分析**任务而创建的
- Pandas 纳入了大量库及一些标准的数据模型，提供了高效的操作大型数据集所需要的工具
- Pandas 提供了大量能使我们快速便捷地处理数据的函数与方法
- 是 Python 成为强大而高效的数据分析环境的重要因素之一


## 两种基本数据结构：Series & DataFrame
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

## 写入csv文件
- DataFrame.to_csv('文件名')
```
import numpy as np
import pandas as pd

b = pd.DataFrame(np.random.randn(15,4))

b.to_csv('out.csv')                             # 正确
pd.to_csv('out.csv')                            # 错误
```

## 读取csv文件
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


## 对DataFrame的操作
- data[data = 'NR' ] = 0
- data.to_numpy()
- loc & iloc的区别：建议当用行索引的时候, 尽量用 iloc 来进行索引; 而用标签索引的时候用 loc 
  - iloc: 通过行号来取行数据（如取第二行的数据）
  - data.iloc[:, :] 注意：dataframe的表头不算在index里面，因此dataframe index=0就是excel表格中的表头下面的第一行
```
data.iloc[2]             # 取第2行全部（不算表头的第二行，因为dataframe的index不是表头，而是表头下的第一行）
data.iloc[1:4]           # 取第1-3行全部（左闭右开）
data.iloc[2, 2]          # 取第2行第2列的内容
data.iloc[2:3, 0:]       # 用切片来指定行列
data.iloc[2,'c']         # 错误，iloc不能直接取字段，loc可以
```
  - loc: 通过行索引 "Index" 中的具体值来取行数据（如取"Index"为"A"的行）
```
data.loc[2]        # 同iloc
data.loc[1:4]      # 与iloc不同，左闭右闭
data.loc[2,'c']    # 可以直接取字段
data.loc[2,3]      # 错误，不能取指定行指定列
```

# Matplotlib
## 介绍
- Matplotlib 是 Python 的一个绘图库。它包含了大量的工具，你可以使用这些工具创建各种图形，包括简单的散点图，正弦曲线，甚至是三维图形
  
## 画图
```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,2*np.pi,100)
plt.plot(x,np.sin(x),'r-^',x,np.cos(x),'g-*')
plt.show()
```

## 使用子图
```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,2*np.pi,100)
plt.subplot(2,1,1)                 # （行，列，活跃区）
plt.plot(x,np.sin(x),'k')
plt.subplot(2,1,2)
plt.plot(x,np.cos(x),'y')

plt.show()
```
- plt.subplots()和plt.subplot的区别
- plt.subplots()是一个函数，返回一个包含figure和axes（轴）对象的元组。因此，使用fig,ax = plt.subplots()将元组分解为fig和ax两个变量。axes[0]便是第一个子图，axes[1]是第二个。
```
# 函数
subplots: (nrows: int = ..., ncols: int = ..., sharex: bool | Literal['none', 'all', 'row', 'col'] = ..., sharey: bool | Literal['none', 'all', 'row', 'col'] = ..., squeeze: bool = ..., subplot_kw: Dict | None = ..., gridspec_kw: Dict | None = ..., **fig_kw: Any) -> Tuple[Figure, Axes]

# 示例
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
```
- 我看参数里并没有figsize这个参数？？
```

```

5. subplots 和subplot的用法？横轴纵轴以及title的标注等？figsize？两者都可以实现画子图功能，只不过subplots帮我们把画板规划好了，返回一个坐标数组对象，而subplot每次只能返回一个坐标对象，subplots可以直接指定画板的大小
- subplot:plt.subplot('行','列','编号/活跃区'), 编号是一行一行数的
```
plt.subplot(2, 2, 1)          # 两行两列，这是第一个图 
plt.plot(x, y, 'b--')         # x和y,指定线条颜色b：blue，指定线条形状--
plt.ylabel('y1')
plt.subplot(2, 2, 2)          # 两行两列,这是第二个图
plt.plot(x, y,'r--')
plt.ylabel('y2')
plt.subplot(2, 2, 3)          # 两行两列,这是第三个图
plt.plot(x, y,'m--')
plt.subplot(2, 2, 4)          # 两行两列,这是第四个图
plt.plot(x, y,'k--')

plt.show()
```
- subplots:fig 变量可以让我们可以修改 figure 层级（figure-level）的属性或者将 figure 保存成图片，例如：fig.savefig('thefilename.png')。ax 变量中保存着所有子图的可操作 axe 对象。
```
fig, ax=plt.subplots(2,2)

ax[0][0].plot(t,s,'r*')
ax[0][1].plot(t*2,s,'b--')

plt.show()
```

## 标题、标签、图例 & 图片保存
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

## 散点图
```
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,2*np.pi,100)
plt.scatter(x,np.sin(x),np.random.rand(100)*40,'r')                     # 点的size和颜色均可以指定

plt.show()
```

## 直方图
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



## Seaborn
- Seaborn 基于 matplotlib， 可以快速的绘制一些统计图表
- 看看老师视频？老师也没具体讲，自己研究吧


# Scikit-learn 
- Python 语言的机器学习工具
- Scikit-learn 包括大量常用的机器学习算法
- Scikit-learn 文档完善，容易上手

## sklearn 表格
<img src="http://imgbed.momodel.cn/q2nay75zew.png" width=800>

由图中，可以看到机器学习 `sklearn` 库的算法主要有四类：分类，回归，聚类，降维。其中：

+ 常用的回归：线性、决策树、`SVM`、`KNN` ；  
    集成回归：随机森林、`Adaboost`、`GradientBoosting`、`Bagging`、`ExtraTrees` 
+ 常用的分类：线性、决策树、`SVM`、`KNN`、朴素贝叶斯；  
    集成分类：随机森林、`Adaboost`、`GradientBoosting`、`Bagging`、`ExtraTrees` 
+ 常用聚类：`k` 均值（`K-means`）、层次聚类（`Hierarchical clustering`）、`DBSCAN` 
+ 常用降维：`LinearDiscriminantAnalysis`、`PCA`   　　

这个流程图代表：蓝色圆圈是判断条件，绿色方框是可以选择的算法，我们可以根据自己的数据特征和任务目标去找一条自己的操作路线。

## sklearn数据集
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

# 鸢尾花例子

## 数据预处理

### min-max标准化 MinMaxScaler
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

### Z-score标准化 StandardScaler
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


### 归一化
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


### 二值化
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


### 标签编码
- 使用 LabelEncoder 将不连续的数值或文本变量转化为有序的数值型变量
```
from sklearn.preprocessing import LabelEncoder

print(LabelEncoder().fit_transform(['apple', 'pear', 'orange', 'banana']))
```

### 独热编码
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

## 数据集的划分
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

## 定义模型
### 估计器（`Estimator`）
估计器，很多时候可以直接理解成分类器，主要包含两个函数：

+ `fit()`：训练算法，设置内部参数。接收训练集和类别两个参数。
+ `predict()`：预测测试集类别，参数为测试集。

大多数 `scikit-learn` 估计器接收和输出的数据格式均为 `NumPy`数组或类似格式。

<br>

### 转换器（`Transformer`）  
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

## 模型评估
### 交叉验证
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


## 保存模型和加载模型：joblib
- Joblib是一组在Python中提供轻量级管道的工具。特别是:1. 函数的透明磁盘缓存和延迟重新计算(记忆模式)。2. 简单并行计算。
- Joblib经过了优化，特别是在处理大型数据时速度更快、更健壮，并且对numpy数组进行了特定的优化。
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
- 模型保存 .model & .pth & .pkl  % .pt的区别
```
torch.save(model, './xxx.model')
```
- 我们经常会看到后缀名为.pt, .pth, .pkl的pytorch模型文件，这几种模型文件在格式上有什么区别吗？其实它们并不是在格式上有区别，只是后缀不同而已（仅此而已），在用torch.save()函数保存模型文件时，各人有不同的喜好，有些人喜欢用.pt后缀，有些人喜欢用.pth或.pkl.用相同的torch.save()语句保存出来的模型文件没有什么不同。在pytorch官方的文档/代码里，有用.pt的，也有用.pth的。一般惯例是使用.pth,但是官方文档里貌似.pt更多，而且官方也不是很在意固定用一种。
```
torch.save(model.state_dict(), mymodel.pth)      #只保存模型权重参数，不保存模型结构
torch.save(model, mymodel.pth)                   #保存整个model的状态
```
- torch.save
- torch.load:用来加载模型。torch.load() 使用 Python 的 解压工具（unpickling）来反序列化 pickled object 到对应存储设备上。首先在 CPU 上对压缩对象进行反序列化并且移动到它们保存的存储设备上，如果失败了（如：由于系统中没有相应的存储设备），就会抛出一个异常。用户可以通过 register_package 进行扩展，使用自己定义的标记和反序列化方法
```
# 函数
torch.load(f, map_location=None, pickle_module=<module 'pickle' from '...'>)

# 使用
torch.load('tensors.pt', map_location=torch.device('cpu'))
```

- 保存模型和加载模型 整理一下
```
import joblib
import torch

# 保存模型
def my_save_model(model):
    # joblib
    joblib.dump(model, "./ckpt/mymodel1.pkl")
    # torch
    torch.save(model.state_dict(), "./mymodel2.pkl")
    torch.save(model, "./ckpt/mymodel3.pkl")


# 加载模型
def my_load_model(model):
    # joblib
    joblib.load("./ckpt/mymodel.pkl")
    # load_state_dic
    model.load_state_dict(torch.load("./ckpt/mymodel2.pkl"))
    # torch
    torch.load("./ckpt/mymodel3.pkl")
```


# os模块
- os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
- os.path.join(path1, path2) 路径拼接合并

# cv2模块
- cv2.imread()返回np.array类型
- cv2.resize(InputArray src, OutputArray dst, Size, fx, fy, interpolation)

# PIL: Image对象
- PIL(Python Image Library)是python的第三方图像处理库，但是由于其强大的功能与众多的使用人数，几乎已经被认为是python官方图像处理库了
- Image类是PIL中的核心类
```
from PIL import Image             #调用库，包含图像类
    im = Image.open("3d.jpg")     #文件存在的路径，如果没有路径就是当前目录下文件
    im.show()
```
## Image.open()和cv2.imread()的区别
- Image.open(）得到的img数据类型呢是Image对象，不是普通的数组。cv2.imread()得到的img数据类型是np.array()类型。
- 具体可以看https://blog.csdn.net/weixin_42213622/article/details/109110140


# 机器学习完整例子示范
## 步骤
- step 1. 获取及加载数据集
- step 2. 数据预处理
- step 3. 划分数据集
- step 4. 定义模型
- step 5. 模型训练
- step 6. 模型预测
- step 7. 模型评估：测试集算准确率或交叉验证

## 解释说明
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

# pytorch

## 介绍
- **基于python**的科学计算包，主要服务于以下场景：1. 使用**GPU**的强大计算能力 2. 提供最大的灵活性和高速的深度学习研究平台
- Torch & Numpy：Torch是一个与Numpy类似的Tensor（张量）操作库，与numpy不同的是**Torch对GPU支持的很好**
- Tensors & ndarray: Tensors和Numpy中的ndarray类似，但是在pytorch中tensors可以使用GPU进行计算

## 历史发展
- lua语言：一门比python还简单的语言，简单高效，但过于小众
- torch：使用了lua作为接口
- pytorch：不是简单的封装Lua Torch提供python接口，而是对tensor之上的所有模块进行了重构

## Tensorflow & Pytorch & keras的比较
- tensorflow(google) & pytorch(facebook) & keras的比较：tensorflow比较复杂、市场占有率也比较低，keras太简单，封装得太好，未来如果要去深入的建一些模型不太方便。pytorch的市场占有率目前在50%以上

## 查看pytorch版本
```
import torch as t
print(t.__version__)
```

## 官方文档/官方手册
- https://pytorch.org/docs/master/torch.html

## Tensor 张量

### Tensor(张量)创建
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

### 类型转换
- torch.int()将该tensor投射为int类型
```
tensor = xxx
newtensor = tensor.int()
```
- list转tensor
- 
### tensor 拼接
- torch.cat(inputs, dimension=0) 在给定维度上对输入的张量序列 seq 进行连接操作
```
tensor([[-0.1997, -0.6900,  0.7039],
        [ 0.0268, -1.0140, -2.9764]])

>>> torch.cat((x, x, x), 0)	# 在 0 维(纵向)进行拼接

# 输出
tensor([[-0.1997, -0.6900,  0.7039],
        [ 0.0268, -1.0140, -2.9764],
        [-0.1997, -0.6900,  0.7039],
        [ 0.0268, -1.0140, -2.9764],
        [-0.1997, -0.6900,  0.7039],
        [ 0.0268, -1.0140, -2.9764]])
```
- 谁连在谁后面？torch.cat(A,B)。B在A后面，根据后面的Dim决定是按照哪个维度拼，dim=0代表竖着拼，dim=1代表横着拼

### 获取tensor的size
- 注意tensor的size和ndarray的size不一样
- t.size()的返回值是tuple，所以它支持tuple类型的所有操作
- tensor_x.size(0) 与  tensor_x.size()[0]  一样?
```
import torch as t
import numpy as np

x = t.tensor([[1, 2, 3], [2, 3, 4]])
y = np.array([[1, 2, 3], [2, 3, 4]])

print(x.size())                          # size()是函数，返回tensor的大小
print(y.size)                            # size是属性，返回ndarray中的元素数量
```

### 运算：加减乘除、矩阵乘法、矩阵求逆
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


### 改变tensor的维度和大小
- torch.view(), 返回什么？
- torch.view与numpy.reshape类似
- flatten：numpy.ndarray.flatten，返回一个一维数组。flatten只能适用于numpy对象，即array或者mat，普通的list列表不适用。a.flatten()：a是个多维数组，a.flatten()就是把a降到一维，默认是按行的方向降


### 获得数值
- 只有一个元素的张量，可以使用.item()获得python数据类型的数值
```
import torch as t

x = t.randn(1)
print(x)

y = x.item()
print(y)
```

### ndarray和tensor的互相转换
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


### CUDA 张量
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
- to(device) 和 .cuda的区别？两种写法都写一下比较一下。
- to(device) 可以指定CPU 或者GPU, 而.cuda() 只能指定GPU

```
# .cuda()版本

```
```
# to(device)版本
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- tensor.to(device):哪些参数要调到to(device)? 注意x, y, model都要to device
```
for i, (input,label) in enumerate(val_dataloader):
    input = input.to(device)
    label = label.to(device)

model = model.to(device)
```

## Autograd 自动求导机制
- 深度学习算法本质上是通过反向传播求导数，Pytorch的Autograd模块实现了此功能。在Tensor上的所有操作，Autograd都能为它们自动提供微分，避免手动计算导数的复杂过程
- Tensor和Function连成计算图，它表示和存储了完整的计算历史
- Pytorch中所有神经网络的核心是autograd包，autograd包为tensor上的所有操作提供了自动求导
- torch.Tensor中若设置require_grad为True，那么将会追踪所有对于该张量的操作
- 当完成计算后通过调用.backward()，将自动计算所有的梯度，这个tensor的所有梯度将会自动积累到.grad属性
- 如果tensor是一个标量，则不需要为backward()指定任何参数，否则，需要制定一个gradient参数来匹配张量的形状
- 要阻止张量跟踪历史记录，可以调用：1. .detach()方法 2. 将代码块包装在with torch.no_grad():中，在评估模型时特别有用，因为模型可能具有requires_grad = True的可训练参数，但是我们不需要梯度计算
```
报错 Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead
```


### autograd.Variable
- pytorch0.4更新后Tensor和Variable合并:torch.Tensor 和torch.autograd.Variable现在是同一个类。torch.Tensor 能够像之前的Variable一样追踪历史和反向传播。Variable仍能够正常工作，但是返回的是Tensor。所以在0.4的代码中，不需要使用Variable了。
- autograd.Variable是Autograd中的核心类，它简单封装了tensor，并支持几乎所有tensor的操作。Tensor在被封装为Variable之后，可以调动它的.backward实现反向传播，自动计算所有梯度
- forward函数的输入和输出都是Variable，只有Variable才具有自动求导功能，Tensor是没有的，所以在输入时，需要把Tensor封装成Variable
- Variable的数据结构，autograd.Variable中包含了data、grad、grad_fn。
- grad也是个Variable，而不是tensor，它和data的形状一样
- Variable和tensor具有近乎一致的接口，在实际使用中可以无缝切换


### Variable创建
```
import torch as t

x = t.autograd.Variable(t.ones(3, 5), requires_grad=True)

print(x)
```

### requires_grad = True
- torch.Tensor中若设置require_grad为True，那么将会追踪所有对于该张量的操作

```
import torch as t

a = t.ones(5, 3, requires_grad=True)

print(a)
```

### 梯度 反向传播
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


### 另一个autograd的例子
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


### with torch.no_grad()
- 如果requires_grad=True但是你又不希望进行autograd的计算，那么可以将变量包裹在with torch.no_grad()中
- 在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用 with torch.no_grad():，强制之后的内容不进行计算图构建。
- 
```
import torch as t

x = t.randn(3, requires_grad=True)

print(x.requires_grad)
print((x**2).requires_grad)

with t.no_grad():
    print(x.requires_grad)
    print((x**2).requires_grad)
```


## 数据加载与预处理
- Pytorch 读取数据虽然特别灵活，但是还是具有特定的流程的，它的操作顺序为：
  - 创建一个 Dataset 对象，该对象如果现有的 Dataset 不能够满足需求，我们也可以自定义 Dataset`
  - 创建一个 DataLoader 对象
  - 不停的 循环 这个 DataLoader 对象
- 一般情况下，处理图像、文本、音频和视频数据时，可以使用标准的python包来加载数据到一个numpy数组中，然后把这个数组转换成torch.*Tensor
- 图像可以使用Pillow, Opencv
- 音频可以使用scipy，librosa
- 文本可以使用原始python和Cython来加载，或是使用NLTK和SpaCy处理
- 特别地，对于图像任务,可使用torchvision，它包含了处理一些基本图像数据集地方法。这些数据集包括Imagenet,CIFAR10,MNIST等。除了数据加载意以外，torchvision还包含了数据转换器。torchvision.datasets 和 torch.utils.data.DataLoader


### Dataset
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


### torchvision
- torchvision主要包含以下三部分：
  - models： 提供深度学习中各种经典网络的网络结构和预训练好的模型，包括Alex-Net、VGG系列、ResNet系列，Inception系列等
  - datasets：提供常用的数据集加载，设计上都是继承torch.utils.data.Dataset，主要包括MNIST、CIFAR10/100、ImageNet、COCO等
  - transforms：提供常用的数据预处理操作，主要包括对Tensor以及PIL Image对象的操作
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
  T.Resize(224),                                       # 保持长宽比不变，最短边为224像素
  T.CenterCrop(224),                                   # 从图片中间切出224*224的图片
  T.ToTensor(),                                        # 将图片从图片（Image）转成Tensor，并且归一化至[0,1]
  T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])     # 标准化至[-1,1]
])
```



### torch.utils.data.DataLoader
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


## torch.nn 神经网络
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

### torch.nn.init
- 均匀分布 U(a, b)
```
tensor = t.empty(2, 3)
torch.nn.init.uniform_(tensor, a=0, b=1)
```
- 正态分布 N(mean,std)
```
torch.nn.init.normal_(tensor, mean=0, std=1)
```

### nn.functional & nn.Module
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


### 定义网络模型
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


### 网络中的可学习参数
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


## 损失函数
- Pytorch将损失函数实现为nn.Module的子类
- 一个损失函数接收一对(output,target)作为输入，计算一个值来估计网络输出和目标值相差多少
- nn包中有很多不同的损失函数。nn.MSEloss (mean squared error)是一个比较简单的损失函数，它计算输出和目标之间的均方误差。分类问题做最小二乘法的收敛性会很差，所以一般分类问题很少用最小二乘法来做
```
import torch.nn as nn

criterion = nn.MSEloss()
loss = criterion(output, target)
```
- 损失函数：在深度学习中要用到各种各样的损失函数，这些损失函数可以看作是一种特殊的layer，Pytorch也将这些损失函数定义为nn.Module的子类。然而在实际使用中，通常将这些损失函数专门提取出来，作为独立的一部分。
- 注意，criterion(predict_y, label)，注意参数的顺序，不可以反过来



## torch.optim 优化器：更新权重 
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


## 训练网络
- 多进程需要在main函数中运行，因此当num_workers设定大于1时，需要在训练时加上`if __name__=='__main__'`:
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

## GPU加速：cuda
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

## 验证
- 验证相对来说比较简单，但要注意需将模型置于验证模式model.eval(), 验证完成后还需要将其置灰训练模式model.train()。这两句代码会影响BatchNorm和Dropout等层的运行模式。


# 范例

## 神经网络的典型训练过程

### 老师版
- Step1: 定义网络模型：定义包含一些可学习的参数（或者叫权重）神经网络模型
- Step2: 加载数据集和数据预处理：访问数据集，数据集一般都是一个batch一个batch去访问的
- Step3: 输入数据到神经网络
- Step4: 定义损失函数计算损失（输出结果和正确值的差值大小），比如分类用CrossEntropy
- Step5: 反向传播获得梯度
- Step6: 用获得的梯度更新网络的参数：weight = weight - learning_rate * gradient

### 自己版
- Step1：定义网络模型
- Step2：加载数据集和数据预处理
- Step3：定义损失函数和优化器
- Step4：训练模型
- Step5：评估模型

## 示例：训练一个图像分类器：CIFAR-10
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

# math 模块
- math.floor(x) 返回数字x的下舍整数，即向下取整
- math.round(x) 四舍五入
- math.ceil(x) 向上取整

# python语法
- with open() as f:有一些任务，可能事先需要设置，事后做清理工作。如果不用with语句,一是可能忘记关闭文件句柄；二是文件读取数据发生异常，没有进行任何处理。这时候就是with一展身手的时候了。除了有更优雅的语法，with还可以很好的处理上下文环境产生的异常。
- 默认文件访问模式为只读(r)
```
with open(path, 'r') as f:
    pass
```
- a.strip():Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。返回移除字符串头尾指定的字符生成的新字符串。
```
a.strip('\n')
```
- a.split(): Python split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串。返回分割后的字符串列表。
- next(f): 跳过一行，返回值也是跳过一行的f，f本身也是跳过一行了
- f.readline()：readline() 方法用于从文件读取整行，包括 "\n" 字符。如果指定了一个非负数的参数，则返回指定大小的字节数，包括 "\n" 字符。
- f.readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，该列表可以由 Python 的 for... in ... 结构进行处理。如果碰到结束符 EOF 则返回空字符串。
```
lines = f.readlines()
for line in lines:
    pass
```
- format用法:相对基本格式化输出采用‘%’的方法，format()功能更强大，该函数把字符串当成一个模板，通过传入的参数进行格式化，并且使用大括号‘{}’作为特殊字符代替‘%’.1、基本用法:（1）不带编号，即“{}”,（2）带数字编号，可调换顺序，即“{1}”、“{2}”,（3）带关键字，即“{a}”、“{tom}”
```

```
- list.append(element), 列表中增加一个元素，例如 fruit.append('apple')
- zip():zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
- 迭代器？迭代器有两个基本的方法：iter() 和 next()。
```
>>> list=[1,2,3,4]
>>> it = iter(list)    # 创建迭代器对象
>>> print (next(it))   # 输出迭代器的下一个元素
1
>>> print (next(it))
2
>>>
```
迭代器对象可以使用常规for语句进行遍历
```
list=[1,2,3,4]
it = iter(list)    # 创建迭代器对象
for x in it:
    print (x, end=" ")
```
- 把一个类作为一个迭代器使用需要在类中实现两个方法 __iter__() 与 __next__()。如果你已经了解的面向对象编程，就知道类都有一个构造函数，Python 的构造函数为 __init__(), 它会在对象初始化的时候执行。
  - `__iter__()` 方法返回一个特殊的迭代器对象， 这个迭代器对象实现了 `__next__()` 方法并通过 StopIteration 异常标识迭代的完成。
  - `__next__()` 方法（Python 2 里是 next()）会返回下一个迭代器对象。

- string.isdigit(): Python isdigit() 方法检测字符串是否只由数字组成。如果字符串只包含数字则返回 True 否则返回 False。
- bytes 函数返回一个新的 bytes 对象，该对象是一个 0 <= x < 256 区间内的整数不可变序列。它是 bytearray 的不可变版本。
```
class bytes([source[], encoding[], errors])
```
    - 如果 source 为整数，则返回一个长度为 source 的初始化数组；
    - 如果 source 为字符串，则按照指定的 encoding 将字符串转换为字节序列；
    - 如果 source 为可迭代类型，则元素必须为[0 ,255] 中的整数；
    - 如果 source 为与 buffer 接口一致的对象，则此对象也可以被用于初始化 bytearray。
    - 如果没有输入任何参数，默认就是初始化数组为0个元素。

- assert:Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况。
```
# 官方
# if not expression, raise AssertionError(arguments)
assert expression [, arguments]

# 示例
# 如果 encoder.n_layers == decoder.n_layers， 继续往下执行
# 如果 encoder.n_layers == decoder.n_layers不成立， 报错，报错为参数内容"Encoder and decoder must have equal number of layers!"
assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"
```

-  `__call__()`?类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
- filter(): filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换。该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
```
filter(function, iterable)

# 示例
filter (None, a)       # 函数进行数据过滤空值None,默认会把0、false这样具体的值过滤掉
```



# Q&A
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

# python 工程目录组织/项目完整开发流程
## 网上找的
- https://zhuanlan.zhihu.com/p/36221226
```
Foo/
|-- bin/
|   |-- foo
|
|-- foo/
|   |-- tests/
|   |   |-- __init__.py
|   |   |-- test_main.py
|   |
|   |-- __init__.py
|   |-- main.py
|
|-- docs/
|   |-- conf.py
|   |-- abc.rst
|
|-- setup.py
|-- requirements.txt
|-- README
```
- bin/: 存放项目的一些可执行文件，当然你可以起名script/之类的也行。
- foo/: 存放项目的所有源代码。(1) 源代码中的所有模块、包都应该放在此目录。不要置于顶层目录。(2) 其子目录tests/存放单元测试代码； (3) 程序的入口最好命名为main.py。
- docs/: 存放一些文档。
- setup.py: 安装、部署、打包的脚本。
- requirements.txt: 存放软件依赖的外部Python包列表。
- README: 项目说明文件。

## pytorch书上的猫狗实战
- 程序文件的组织架构
```
checkpoints/
data/
    __init__.py
    dataset.py
    get_data.sh
models/
    __init__.py
    AlexNet.py
    BasicModule.py
    ResNet34.py
utils/
    __init__.py
    visualize.py
config.py
main.py
requirements.txt
README.md
```
- 可以看到，几乎每个文件夹下都有`__init__.py`，一个目录如果包含了__init__.py,那么它就变成了一个包。`__init__.py`可以为空，也可以定义为包的属性和方法，但其必须存在，其他程序才能从这个目录中导入相应的模块和函数
```
# main.py
from data.dataset import DogCat
```
- 如果是导入上级目录中的模块，需要加上`sys.path.append('模块所在目录eg:os.path.abspath()')`，sys.path 的作用是：当使用import语句导入模块时，解释器会搜索当前模块所在目录以及sys.path指定的路径去找需要import的模块
```
import sys
sys.path.append('../')
from fatherdirname import xxx
```
- 如果在`__init__.py`中写入`from .dataset import DogCat`,则在main.py中就可以直接写为：
```
# main.py
from data import DogCat
```
- checkpoints/： 用于保存训练好的模型，可使程序在异常推出后仍能重新载入模型，恢复训练
- data/: 数据相关操作，包括数据预处理、dataset实现等
- models/: 模型定义，可以有多个模型，一个模型对应一个文件
- utils/: 可能用到的工具函数
- config.py: 配置文件，所有可配置的变量都集中在此，并提供默认值
- main.py: 主文件，训练和测试程序的入口，可通过不同的命令来指定不同的操作和参数
- requirements.txt: 程序依赖的第三方库
- README.md: 提供程序的必要说明 

## 补充说明
- config.py中一些具体参数的配置, 将原来写的修改一下，使得config.py可以直接配置重要参数
- 可配置的参数主要包括：数据集参数（文件路径、batch_size等），训练参数（学习率，训练epoch等），模型参数
- config.py中包括默认的参数值和更新参数值的方法
```
class DefaultConfig():
    env = 'default'                                 # visdom环境
    model = 'AlexNet'                               # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = './data/train'                
    test_data_root = './data/test1'
    load_model_path = 'checkpoints/model.pth'       # 加载预训练模型的路径，为None代表不加载

    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 20                                 # print info every N batch

    debug_file = 'tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4

# 根据字典kwargs更新config参数
def parse(self, kwargs):
    # 更新参数
    for k, v in kwargs.items():
        if not hasattr(self,k):
            warning.warn("Warning:opt has not attribute %s" %k)
        setattr(self, k, v)

    # 打印配置信息
    print("user config:")
    for k, v in self.__class__.__dict__.item():
        if not k.startswith('__'):
            print(k, getattr(self, k))
```
- 我们在实际使用中不需要每次都修改config.py，只需要通过命令行传入所需参数，覆盖默认配置即可
```
opt = DefaultConfig()
new_config = {'lr':0.1, 'use_gpu':False}
opt.parse(new_config)
opt.lr = 0.1
```
- getattr(): getattr() 函数用于返回一个对象属性值
```
>>>class A(object):
...     bar = 1
... 
>>> a = A()
>>> getattr(a, 'bar')        # 获取属性 bar 值
1
```
- hasattr(): hasattr() 函数用于判断对象是否包含对应的属性。如果对象有该属性返回 True，否则返回 False。
```
class Coordinate:
    x = 10
    y = -5
    z = 0
 
point1 = Coordinate() 
print(hasattr(point1, 'x'))

# 输出
True
```



# 环境变量/工作目录/当前路径
- %systemroot%: 文件夹路径栏中输入%systemroot%即可打开windows的系统目录。
- systemroot是系统的环境变量之一
- 环境变量：指明操作系统的重要目录在哪里
- 环境变量分为系统变量和用户变量。用户变量只针对当前系统登陆的用户。用户变量中也有PATH，作用和系统变量中的PATH差不多。如果打开我的用户变量， 发现里面有几条是和python有关的，它的作用是可以让我们随时随地的运行python
- 为什么在运行对话框中输入cmd可以打开命令行，输入自己写的却不行？因为自己写的没有加入环境变量中，而cmd就在环境变量中。在运行对话框中输入cmd后，系统会尝试在PATH指明的目录中查找cmd这个程序
- cwd：Current Working Directory
- sys.path: sys.path是python的搜索模块的路径集，是一个list。可以在python 环境下使用sys.path.append(path)添加相关的路径，但在退出python环境后自己添加的路径就会自动消失
```
os.argv[0]                  # sys.argv[0]表示当前py文件本身路径
os.getcwd()                 # os.getcwd() 返回当前工作目录,与当前py文件所在的位置无关
sys.path                    # sys.path是python搜索模块时的路径集，是一个list
os.environ['eg:PATH']       # os.environ 是一个字典，是环境变量的字典
os.path.abspath('./')       # 当前工作目录，非当前py文件的路径
os.path.abspath('../')       # 上级工作目录
```
- Python import 模块的搜索路径
  - 在当前目录下搜索该模块
  - 在环境变量 PYTHONPATH 中指定的路径列表中依次搜索
  - 在 Python 安装路径的 lib 库中搜索
- PYTHONPATH是Python搜索路径，默认我们import的模块都会从PYTHONPATH里面寻找
- os.environ['PYTHONPATH'] = sys.path 。PYTHONPATH好像要自己设置，我没有设置，所以环境变量里没有PYTHONPATH这一项


# *args和**kwargs
- 在Python中的代码中经常会见到这两个词 args 和 kwargs，前面通常还会加上一个或者两个星号。其实这只是编程人员约定的变量名字，args 是 arguments 的缩写，表示位置参数；kwargs 是 keyword arguments 的缩写，表示关键字参数。这其实就是 Python 中可变参数的两种形式，并且 *args 必须放在 **kwargs 的前面，因为位置参数在关键字参数的前面。
## *args
- *args就是就是传递一个可变参数列表给函数实参，这个参数列表的数目未知，甚至长度可以为0。下面这段代码演示了如何使用args
## **kwargs
- 而`**kwargs`则是将一个可变的关键字参数的字典传给函数实参，同样参数列表长度可以为0或为其他值。`**kwargs`允许你将不定长度的键值对，作为参数传递给一个函数。



## Q&A
1. 工作目录跟环境变量的关系是什么？
2. 

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


# Homework3: CNN

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

# Homework4: RNN

## 模型

### gensim.models.word2vec
- 官方文档https://radimrehurek.com/gensim/models/word2vec.html
- 一款开源的python第三方工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。
- 主要用于主题建模和文档相似性处理，它支持包括TF-IDF，LSA，LDA，和word2vec在内的多种主题模型算法。
- Gensim在诸如获取单词的词向量等任务中非常有用。
- word2vec.LineSentence: Iterate over a file that contains sentences: one line = one sentence. Words must be already preprocessed and separated by whitespace.
```
from gensim.models import word2vec

word2vec.LineSentence(source, max_sentence_length=10000, limit=None)
```
```
sentences = LineSentence(source = 'filepath')
for sentence in sentences:
    pass
```
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
```
# 模型保存
model.save('filepath.model')
# 模型加载
model.load('filemath.model')
```
- If you save the model you can continue training it later
```
model = Word2Vec.load("word2vec.model")
model.train([["hello", "world"]], total_examples=1, epochs=1)
```
- The trained word vectors are stored in a KeyedVectors instance, as model.wv
```
# get numpy vector of a word
vector = model.wv['he']     
```
- The reason for separating the trained vectors into KeyedVectors is that if you don’t need the full model state any more (don’t need to continue training), its state can discarded, keeping just the vectors and their keys proper.This results in a much smaller and faster object that can be mmapped for lightning fast loading and sharing the vectors in RAM between processes:


### nn.Embedding():
- 一个简单的查找表（lookup table），存储固定字典和大小的词嵌入。此模块通常用于存储单词嵌入并使用索引检索它们(类似数组)。模块的输入是一个索引列表，输出是相应的词嵌入。
- num_embeddings: 词嵌入字典大小，即一个字典里要有多少个词。
- embedding_dim: 每个词嵌入向量的大小
- pytorch的nn.Embedding()是可以自动学习每个词向量对应的w权重的

```
embedding = nn.Embedding(10, 3)    # 字典里10个词，每个词是3维
```

### nn.LSTM():
- input_size ：输入的维度,就是你输入x的向量大小。输入数据的特征维数，通常就是embedding_dim(词向量的维度)
- hidden_size：h的维度。即隐藏层节点的个数
- num_layers：堆叠LSTM的层数，默认值为1。
- bias：偏置 ，默认值：True
- batch_first： 如果是True，则input为(batch, seq, input_size)。默认值为：False（seq_len, batch, input_size）
- bidirectional ：是否双向传播，默认值为False
```
# 输入的每个词的维度有100，hidden_layer的维度16
lstm = nn.LSTM(100,16,num_layers=2)
```
- 输入数据：input,(h_0,c_0)。 h_0,c_0如果不提供，那么默认是０。
- 输出数据包括output,(h_n,c_n)。
- output[-1]与h_n是相等的，因为output[-1]包含的正是batch_size个句子中每一个句子的最后一个单词的隐藏状态
- 注意LSTM中的隐藏状态其实就是输出，cell state细胞状态才是LSTM中一直隐藏的，记录着信息


## 流程及注意事项

1. 加载数据
   - 加载测试数据时的方法: 先用逗号分隔，然后取出[1:]（去掉开头的id)，再将后面的用''.join()连起来，然后用空格将每个单词分开
   ```
   sentences = [''.join(line.strip('\n').split(',')[1:]) for line in lines]
   ```
   - 注意test数据中要跳过表头，可以用next(f)。跳过一行，返回值也是跳过一行的f，f本身也是跳过一行了
   - 报错：UnicodeDecodeError: ‘gbk’ codec can’t decode byte 0xac in position 8: illegal multibyte sequence。解决办法：指定encoding = 'utf-8'。gbk和utf-8的区别: GBK包含全部中文字符，UTF-8则包含全世界所有国家需要用到的字符。
2. self.embedding.vector_size: 应该是可以得到词向量的维度，但具体我在网上没找到？
```
def get_w2v_model(self):
        # 把之前訓練好的word to vec 模型讀進來
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
```
3.  fix_embedding
```
self.embedding.weight = torch.nn.Parameter(embedding)
# 是否將 embedding fix住，如果fix_embedding為False，在訓練過程中，embedding也會跟著被訓練
self.embedding.weight.requires_grad = False if fix_embedding else True
```
4. nn.BCELoss()和nn.CrossEntropyLoss()的区别？BCELoss是Binary CrossEntropyLoss的缩写，nn.BCELoss ()为二元交叉熵损失函数，只能解决二分类问题
5. 验证集是在每个epoch下都做一次，而不是在所有epoch都做完，直接做验证集计算val的acc。这样在每个epoch下找到最优的model
6. 报错 Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.解决办法之一：将验证集的代码放在with torch.no_grad():下面


## Q&A

1. word2vec的model定义了对象之后并没有进行计算，是怎么训练的？定义对象后就是已经训练好的？
4. RNN训练的是哪些参数？
5. 为什么要转换成longtensor?PyTorch中的常用的tensor类型包括：32位浮点型torch.FloatTensor，64位浮点型torch.DoubleTensor，16位整型torch.ShortTensor，32位整型torch.IntTensor，64位整型torch.LongTensor。原因没找到？
10. main.py中写哪些？哪些写在方法里，哪些写在main.py里？
1. acc画图？
2. 每个.py文件中的 `if __name__=='__main'：` 中都是要写啥？
3. 预测结果保存在哪里?
4. 二分类的output是怎么处理的？我在作业中用的是多分类的办法，可参考猫狗实战书上的看看。作业范例上的也是和我做的方法不一样，可以看看。


# Homework5:

## task 1

- torch.load: 前面有写。用来加载torch.save() 保存的模型文件。
```
checkpoint = torch.load('./checkpoint.pth', map_location=torch.device('cpu'))
```

- model.load_state_dict: 
  - state_dict	保存 parameters 和 persistent buffers 的字典
  - strict	可选，bool型。state_dict 中的 key 是否和 model.state_dict() 返回的 key 一致
```
model.load_state_dict(checkpoint, strict=False)
# 因为checkpoint的key有可能和模型的key不是一一重合，那么strict=Fasle才可以
```


- torch.stack
- permute(置换)是什麼，為什麼這邊要用?在 pytorch 的世界，image tensor 各 dimension 的意義通常為 (channels, height, width)但在 matplolib 的世界，想要把一個 tensor 畫出來，形狀必須為 (height, width, channels)因此 permute 是一個 pytorch 很方便的工具來做 dimension 間的轉換這邊 img.permute(1, 2, 0)，代表轉換後的 tensor，其第 0 個 dimension 為原本 img 的第 1 個 dimension，也就是 height 第 1 個 dimension 為原本 img 的第 2 個 dimension，也就是 width第 2 個 dimension 為原本 img 的第 0 個 dimension，也就是 channels
- permute 返回值？
- permute只能对tensor操作吗？可以对numpy的ndarray操作吗？
- 

## task 2
- hook: 
- torch.nn.Module.register_forward_hook: 为某个模块注册一个前向传播的hook，每次该模块进行前向传播后，hook获得该模块前向传播的值。
- torch.nn.Module.register_forward_hook 返回一个handle。
- hook不应该修改 input和output的值
```
# 函数
hook(module, input, output) -> None

# 范例
hook_handle = model.cnn[cnnid].register_forward_hook(hook)
# 這一行是在告訴 pytorch，當 forward 「過了」第 cnnid 層 cnn 後，要先呼叫 hook 這個我們定義的 function 後才可以繼續 forward 下一層 cnn
```

- 可以通过handle移除注册的hook，语法为handle.remove()
```
hook_handle.remove()
# 很重要：一旦對 model register hook，該 hook 就一直存在。如果之後繼續 register 更多 hook
# 那 model 一次 forward 要做的事情就越來越多，甚至其行為模式會超出你預期 (因為你忘記哪邊用不到的 hook 了)
# 因此事情做完了之後，就把這個 hook 拿掉，下次想要再做事時再 register 就好了。
```

- global：
- Python中定义函数时，若想在函数内部对函数外的变量进行操作，就需要在函数内部声明其为global。加了global，则可以在函数内部对函数外的对象进行操作了，也可以改变它的值了
```
x = 1
def func():
    global x
    x = 2
func()
print(x)
输出：2 
```

- optim.Adam([x], lr=1), 注意x要用[], 否则会出现下面的报错
- 因为是要找出图片（即x)使得某一层的输出最大，所以optim所要更新的参数是x
- 报错：params argument given to the optimizer should be an iterable of Tensors or dicts, but got torch.FloatTensor


- plt.show()和plt.imshow()的区别: plt.imshow()函数负责对图像进行处理，并显示其格式，而plt.show()则是将plt.imshow()处理后的函数显示出来
```
plt.imshow(image)  #image表示待处理的图像
plt.show()
```


## task 3
- Lime 的部分因為有現成的套件可以使用，因此下方直接 demo 如何使用該套件。其實非常的簡單，只需要 implement 兩個 function: classifier_fn 和 segmentation_fn。基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
  - classifier_fn： 定義圖片如何經過 model 得到 prediction
  - segmentation_fn： 定義如何把圖片做 segmentation
- lime 這個套件要吃 numpy array

- lime_image.LimeImageExplainer() 
- explainer.explain_instance(image, classifier_fn, labels=(1, ), hide_color=None, top_labels=5, num_features=100000, num_samples=1000, batch_size=10, segmentation_fn=None, distance_metric='cosine', model_regressor=None, random_seed=None)
  - image:待解释图像
  - classifier_fn:分类器
  - labels:可解析标签
  - hide_color:隐藏颜色
  - top_labels:预测概率最高的K个标签生成解释
  - num_features:说明中出现的最大功能数
  - num_samples:学习线性模型的邻域大小
  - batch_size:批处理大小
  - distance_metric:距离度量
  - model_regressor:模型回归器，默认为岭回归
  - segmentation_fn:分段，将图像分为多个大小
  - random_seed:随机整数，用作分割算法的随机种子
```
explainer = lime_image.LimeImageExplainer()                                
explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
```

- explaination.get_image_and_mask: return (image, mask), where image is a 3d numpy array and mask is a 2d numpy array that can be used with skimage.segmentation.mark_boundaries
  - label: label to explain
  - positive_only: if True, only take superpixels that positively contribute to the prediction of the label.
  - negative_only: if True, only take superpixels that negatively contribute to the prediction of the label. If false, and so is positive_only, then both negativey and positively contributions will be taken. Both can’t be True at the same time
  - hide_rest: if True, make the non-explanation part of the return image gray
  - num_features: number of superpixels to include in explanation
  - min_weight: minimum weight of the superpixels to include in explanation
```
get_image_and_mask(label, positive_only=True, negative_only=False, hide_rest=False, num_features=5, min_weight=0.0)
```

- skimage.segmentation.slic: 在Color-（x，y，z）空间中使用k-means聚类来分割图像。
- 返回什么？标签：2D或3D数组整数掩码指示段标签。
- 官方手册：https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
  - n_segments=100,       # 分割输出的标签数 
  - compactness=10.0,     # 平衡颜色优先性和空间优先性. 值越大，空间优先性权重越大
  - max_iter=10,          # Kmeans 最大迭代数
  - sigma=0,              # 图像每一维预处理的高斯核宽度
```
skimage.segmentation.slic(image, n_segments=100, compactness=10.0, max_iter=10, sigma=0, spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False)
```


## Q&A
1. detach()到底是干啥用的？当我们再训练网络的时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；或者值训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播。tensor.detach()返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播。注意：使用detach返回的tensor和原始的tensor共同一个内存，即一个修改另一个也会跟着改变。
2. x.requires_grad_()   & x.requires_grad = True: requires_grad_()会修改Tensor的requires_grad属性。 x.requires_grad_()和x.requires_grad = True都会将x的requires_grad属性改为True。requires_grad_()函数会改变Tensor的requires_grad属性并返回Tensor，修改requires_grad的操作是原位操作(in place)。其默认参数为requires_grad=True。
```
x.requires_grad = True            # x的requires_gead属性改为True
x.requires_grad_()                # x的requires_gead属性改为True，与上面效果一样
new_x = x.requires_grad_()        # 返回的new_x其实就是x，如果下面修改x，那么new_x也会变化。所以如果需要将requires_grad改为True，可以直接用x.requires_grad_() 
```
3. squeeze(): 去除所有size为1的维度，包括行和列。当维度大于等于2时，squeeze()无作用。其中squeeze(0)代表若第一维度值为1则去除第一维度，squeeze(1)代表若第二维度值为1则去除第二维度。
```
filter_visualization2 = x.detach().cpu().squeeze(0)      # 若第0维度为1，则去掉该维度。否则不变
```


## 注意事项
1. 在 pytorch 的世界，image tensor 各 dimension 的意義通常為 (channels, height, width)。但在 matplolib 的世界，想要把一個 tensor 畫出來，形狀必須為 (height, width, channels)
2. 将tensor作为参数传入函数中，函数中对tensor的处理也会影响
3. 叶子结点是什么？pytorch书上autograd那一节有细讲，忘记了可以去看。
4. 多次反向传播时，梯度是累加的。反向传播的中间缓存会被清空，为进行多次反向传播，需指定retain_graph=True来保存这些缓存。例如，计算w的梯度需要用到x的数值，这些数值在前过程中会保存成buffer，在计算完梯度之后会自动清空。为了能够多次反向传播，需要指定retain_graph来保留这些buffer
5. 非叶子结点的梯度计算完之后即被清空，可以使用hook技术或auto.grad来获取非叶子节点梯度的值。autograd.grad和hook方法都是很强大的工具，更详细的用法参考官方api文档，这里只举例说明其基础的使用方法。推荐使用hook方法，但在实际使用中应尽量避免修改grad的值


# Homework6: Attack and Defense
- sign_data_grad = data_grad.sign()：sign()是Python的Numpy中的取数字符号（数字前的正负号）的函数
- torch.max():
  - 输入：
    - 1、input 是输入的tensor。
    - 2、dim 是索引的维度，dim=0寻找每一列的最大值，dim=1寻找每一行的最大值。
    - 3、keepdim 表示是否需要保持输出的维度与输入一样，keepdim=True表示输出和输入的维度一样，keepdim=False表示输出的维度被压缩了，也就是输出会比输入低一个维度。
  - 输出：
    - 1、max 表示取最大值后的结果。
    - 2、max_indices 表示最大值的索引。

```
# 函数
(max, max_indices) = torch.max(input, dim, keepdim=False)

# 示例
init_pred = output.max(1, keepdim=True)[1]：
```
- transforms.Normalize()?inpalce参数意思？这些值是什么意思，怎么算的？为什么有时候是这个值，有时候是[0.5, 0.5, 0.5]?
- 前面的(0.485,0.456,0.406)表示均值，分别对应的是RGB三个通道；后面的(0.229,0.224,0.225)则表示的是标准差。这个均值和标准差的值是ImageNet数据集计算出来的，所以很多人都使用它们。你也可以计算自己的数据集的均值和标准差。
- 参数inplace=True的意思是进行原地操作，例如：x=x+5是对x的原地操作。而y=x+5,x=y不是对x的原地操作。所以，如果指定inplace=True，则对于上层网络传递下来的tensor直接进行修改，可以少存储变量y，节省运算内存。
- 为什么有时候是[0.485, 0.456, 0.406]，[0.229, 0.224, 0.225] ，有时候是[0.5, 0.5, 0.5]， [0.5, 0.5, 0.5]?都取0.5只是一个近似的操作，实际上其均值和方差并不是这么多。如果你用的是自己创建的数据集，从头训练，那最好还是要自己统计自己数据集的这两个量。如果你加载的的是pytorch上的预训练模型，自己只是微调模型；或者你用了常见的数据集比如VOC或者COCO之类的，但是用的是自己的网络结构，即pytorch上没有可选的预训练模型那么可以使用一个pytorch上给的通用的统计值：mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
```
self.mean = [0.485, 0.456, 0.406]
self.std = [0.229, 0.224, 0.225]
self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)   # inplace默认是False
```
- categories文件没用到？只有用来在展示的图片上标注出label的name而已。
- data_grad = data.grad.data？是不是和data.grad是一样的？是的，是一样的。参考链接：https://www.cnblogs.com/king-lps/p/8570021.html
- 数据集里面出现了个.DS_Store文件 ：.DS_Store是Mac OS保存文件夹的自定义属性的隐藏文件。


## Q&A
1. F.nll_loss：NLLLoss 的全称是Negative Log Likelihood Loss,中文名称是最大似然或者log似然代价函数。F.nll_loss(torch.log(F.softmax(inputs, dim=1)，target)的函数功能与F.cross_entropy相同。F.nll_loss中实现了对于target的one-hot encoding，将其编码成与input shape相同的tensor，然后与前面那一项（即F.nll_loss输入的第一项）进行 element-wise production。用途：用于处理多分类问题
2. nll_loss 和crossentropy 的区别？什么时候用nll_loss,什么时候用crossentropy？查的资料有点看不懂。
3. epsilon设置的越高，攻击成功率越高吗？
4. 不需要算梯度的地方有没有标注？


# homework7: Network Compression

## Architecture Design
- 在這個notebook中我們會介紹MobileNet v1的Architecture Design
- 第一層我們通常不會拆解Convolution Layer
- ReLU和ReLU6的区别？ReLU6 限制最大只會到6。 MobileNet系列都是使用ReLU6。使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。ReLU的目的主要是为了在移动端float16的低精度的时候，也能有很好的数值分辨率，如果对ReLu的输出值不加限制，那么输出范围就是0到正无穷，而低精度的float16无法精确描述其数值，带来精度损失
- 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。

```
nn.Sequential(
    nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),  # Depthwise Convolution
    nn.BatchNorm2d(bandwidth[0]),                                         # Batch Normalization
    nn.ReLU6(),
    nn.Conv2d(bandwidth[0], bandwidth[1], 1),                             # Pointwise Convolution
    nn.MaxPool2d(2, 2, 0),
```
- nn.AdaptiveAvgPool2d？這邊我們採用Global Average Pooling。如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。nn.AdaptiveAvgPool2d((1,1))，首先这句话的含义是使得池化后的每个通道上的大小是一个1x1的，也就是每个通道上只有一个像素点。（1，1）表示的outputsize。
```
nn.AdaptiveAvgPool2d((1, 1))
```

### Q&A
2. 为什么用bandwidth？为什么需要8层？为什么pointerwise后面不需要激活和batchnorm？
3. 为什么最后每个通道的大小设置为是(1，1)

## Knowledge Distillation
- nn.KLDivLoss：相对熵(relative entropy)又称为KL散度（Kullback-Leibler divergence），KL距离，是两个随机分布间距离的度量。记为DKL(p||q)。它度量当真实分布为p时，假设分布q的无效性。
- reduction (string, optional) – Specifies the reduction to apply to the output: 'none' | 'batchmean' | 'sum' | 'mean'. 'none': no reduction will be applied. 'batchmean': the sum of the output will be divided by batchsize. 'sum': the output will be summed. 'mean': the output will be divided by the number of elements in the output. Default: 'mean'
- 参考链接：https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
```
def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss
```
- PyTorch的KL散度损失(KLDivLos)要求输入是概率分布和对数概率分布，这就是为什么后面我们在老师/学生输出(原始分数)上使用softmax和log-softmax。
- F.log_softmax 和 F.softmax？F.log_softmax在数学上等价于log(softmax(x))。参考链接：https://blog.csdn.net/hao5335156/article/details/80607732
- teacher_net = models.resnet18(pretrained=False, num_classes=11): models这个模块提供了深度学习中各种经典网络的网络结构以及预训练好的模型。这里的pretrained参数就是那个与训练好的模型，可以直接加载。

## Network Pruning
- weight和neuron pruning差別在於prune掉一個neuron就等於是把一個matrix的整個column全部砍掉。但如此一來速度就會比較快。因為neuron pruning後matrix整體變小，但weight pruning大小不變，只是有很多空洞。
- 在這裡我們介紹一個很簡單可以衡量Neuron重要性的方法 - 就是看batchnorm layer的γ因子來決定neuron的重要性。 (by paper - Network Slimming)



## Q&A
4. PyTorch 中，nn 与 nn.functional 有什么区别？https://www.zhihu.com/question/66782101
5. 在PyTorch的很多函数中都会包含 forward() 函数，但是也没见哪里调用过forward() 函数，不免让人产生疑惑，想要了解 forward() 函数的作用，首先要了解 Python 中的 __ call __ 函数，__ call __ 函数的作用是能够让python中的类能够像方法一样被调用。因为 PyTorch 中的大部分方法都继承自 torch.nn.Module，而 torch.nn.Module 的__call__(self)函数中会返回 forward()函数 的结果，因此PyTroch中的 forward()函数等于是被嵌套在了__call__(self)函数中；因此forward()函数可以直接通过类名被调用，而不用实例化对象。参考链接：https://blog.csdn.net/lch551218/article/details/116305995
6. 验证集不需要shuffle吧？对的，只有训练集需要shuffle。
7. f字符串
8. items():Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组。
9. startswith(): Python startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False。如果参数 beg 和 end 指定值，则在指定范围内检查
10. endswith():
11. state_dict():
12. int(): 向下取整
13. torch.argsort()

 
 ## Weight Quantization
 - Python中的 pickle 模块实现了基本的数据序列与反序列化。序列化对象可以在磁盘上保存对象，并在需要的时候读取出来。任何对象都可以执行序列化操作。
 - pickle.dump():pickle.dump(obj, file, protocol=None,)必填参数obj表示将要封装的对象。必填参数file表示obj要写入的文件对象，file必须以二进制可写模式打开，即“wb”。可选参数protocol表示告知pickler使用的协议，支持的协议有0,1,2,3，默认的协议是添加在Python 3中的协议3，其他的协议详情见参考文档
 - pickle.load()


# homwork8:seq2seq

## json 模块
- json: JavaScript Object Notation
- json.load()
```
with open(os.path.join(self.root, f'word2int_{language}.json'), "r") as f:
    word2int = json.load(f)
```

## re模块
- re.split()
- 解释一下 r"[。 |！]" : 正则表达式和 \ 会有冲突，'r'是为了保证python在解析"[。 |！]"的时候，把它当做一个字符串来处理，不转义。当定义多个分隔符的时候，要将分隔符放在[]中或者()中
```
string_list = filter_data(re.split(r"[。|！]", my_string))
```

## seq2seq模型
- nn.Embedding():
  - 创建一个词嵌入模型，numembeddings代表一共有多少个词, embedding_dim代表你想要为每个词创建一个多少维的向量来表示它
```
import torch
from torch import nn

embedding = nn.Embedding(5, 4) # 假定字典中只有5个词，词向量维度为4
word = [[1, 2, 3],
        [2, 3, 4]] # 每个数字代表一个词，例如 {'!':0,'how':1, 'are':2, 'you':3,  'ok':4}
         		   #而且这些数字的范围只能在0～4之间，因为上面定义了只有5个词
embed = embedding(torch.LongTensor(word))
print(embed) 
print(embed.size())

# 输出结果
tensor([[[-0.4093, -1.0110,  0.6731,  0.0790],
         [-0.6557, -0.9846, -0.1647,  2.2633],
         [-0.5706, -1.1936, -0.2704,  0.0708]],

        [[-0.6557, -0.9846, -0.1647,  2.2633],
         [-0.5706, -1.1936, -0.2704,  0.0708],
         [ 0.2242, -0.5989,  0.4237,  2.2405]]], grad_fn=<EmbeddingBackward>)
torch.Size([2, 3, 4])
```
  - 一个简单的查找表（lookup table），存储固定字典和大小的词嵌入。此模块通常用于存储单词嵌入并使用索引检索它们(类似数组)。模块的输入是一个索引列表，输出是相应的词嵌入。
  - num_embeddings: 词嵌入字典大小，即一个字典里要有多少个词。
  - embedding_dim: 每个词嵌入向量的大小
  - pytorch的nn.Embedding()是可以自动学习每个词向量对应的w权重的

  ```
  embedding = nn.Embedding(10, 3)    # 字典里10个词，每个词是3维
  ```
- nn.LSTM():
  - input_size ：输入的维度,就是你输入x的向量大小。输入数据的特征维数，通常就是embedding_dim(词向量的维度)
  - hidden_size：h的维度。即隐藏层节点的个数
  - num_layers：堆叠LSTM的层数，默认值为1。
  - bias：偏置 ，默认值：True
  - batch_first： 如果是True，则input为(batch, seq, input_size)。默认值为：False（seq_len, batch, input_size）
  - bidirectional ：是否双向传播，默认值为False
  ```
  # 输入的每个词的维度有100，hidden_layer的维度16
  lstm = nn.LSTM(100,16,num_layers=2)
  ```
  - 输入数据：input,(h_0,c_0)。 h_0,c_0如果不提供，那么默认是０。
  - 输出数据包括output,(h_n,c_n)。
  - output[-1]与h_n是相等的，因为output[-1]包含的正是batch_size个句子中每一个句子的最后一个单词的隐藏状态
  - 注意LSTM中的隐藏状态其实就是输出，cell state细胞状态才是LSTM中一直隐藏的，记录着信息

- nn.GRU()
  - input_size：输入数据X的特征值的数目。
  - hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
  - num_layers：循环神经网络的层数，默认值是 1。
  - bias：默认为 True，如果为 false 则表示神经元不使用 bias 偏移参数。
  - batch_first：如果设置为 True，则输入数据的维度中第一个维度就 是 batch 值，默认为 False。默认情况下第一个维度是序列的长度， 第二个维度才是 - - batch，第三个维度是特征数目。
  - dropout：如果不为空，则表示最后跟一个 dropout 层抛弃部分数据，抛弃数据的比例由该参数指定。默认为0。
  - bidirectional : 如果为True, 则是双向的网络，分为前向和后向。默认为false

## BLEU score 
- BLEU score：bilingual evaluation understudy，即：双语互译质量评估辅助工具。参考链接：https://cloud.tencent.com/developer/article/1042161
- Python自然语言工具包库（NLTK）提供了BLEU评分的实现，你可以使用它来评估生成的文本，通过与参考文本对比。
- NLTK提供了sentence_bleu（）函数，用于根据一个或多个参考语句来评估候选语句。参考语句必须作为语句列表来提供，其中每个语句是一个记号列表。候选语句作为一个记号列表被提供。
- 单独的N-Gram分数: 单独的N-gram分数是对特定顺序的匹配n元组的评分，例如单个单词（称为1-gram）或单词对（称为2-gram或bigram）。权重被指定为一个数组，其中每个索引对应相应次序的n元组。仅要计算1-gram匹配的BLEU分数，你可以指定1-gram权重为1，对于2元,3元和4元指定权重为0，也就是权重为（1,0,0,0）。
```
import nltk
from nltk.translate.bleu_score import sentence_bleu

score = sentence_bleu(reference, candidate)
score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
```
## RNN输入维度和输出维度
- RNN中：batchsize的位置是position 1。CNN中和RNN中batchSize的默认位置是不同的。CNN中：batchsize的位置是position 0.
- 输入的维度[seq_len, batch_size, input_dim]，input_dim是输入的维度，seq_len是一个句子的最大长度
- output的输出维度[seq_len, batch_size, output_dim]
- hidden_state的维度[num_layers * num_directions, batch, hidden_size]
- 参考链接：https://www.jianshu.com/p/b942e65cb0a3
- batch_first? 为什么RNN输入默认batch first=False？用过PyTorch的朋友大概都知道，对于不同的网络层，输入的维度虽然不同，但是通常输入的第一个维度都是batch_size，比如torch.nn.Linear的输入(batch_size,in_features)，torch.nn.Conv2d的输入（batch_size, C, H, W）。而RNN的输入却是(seq_len, batch_size, input_size)，batch_size位于第二维度！虽然你可以将batch_size和序列长度seq_len对换位置，此时只需要令batch_first=True。原因：这是为了便于并行计算。因为cuDNN中RNN的API就是batch_size在第二维度！进一步，为啥cuDNN要这么做呢？因为batch first意味着模型的输入（一个Tensor）在内存中存储时，先存储第一个sequence，再存储第二个... 而如果是seq_len first，模型的输入在内存中，先存储所有序列的第一个单元，然后是第二个单元... seq_len first意味着不同序列中同一个时刻对应的输入单元在内存中是毗邻的，这样才能做到真正的batch计算。参考链接：https://www.jianshu.com/p/41c15d301542
- 注意，batch_first只会让input和output的batch放在第一维度，hidden的维度并不会改变


## 要点
- RNN的forward函数怎么不用传hiddenstate进去，而是只要return hidden state？encoder没有传入hidden的初始值，但是decoder需要传入hidden的初始值，为什么？hidden的初始值如果没有传进去，那就是默认为0
- 切片会自动降维？是的。
- 如何写Configuration类? 注意要继承object
```
class Config(object):
    def __init__(self):
        self.emb_dim = 6                 
        self.n_layers = 1
        self.dropout = 0.5
        self.hid_dim = 7
        self.lr = 0.1
        self.data_path = './data/minidata'
```
- RNN模型训练，shuffle需要=True吗？老师的作业范例中，训练数据是有shuffle的。

# Q&A

3. nn.Dropout(): dropout应该放在哪一层？老师的作业范例中，都是在embedding层后使用dropout。之后学到可以再注意看看，看看别的代码是用在哪一层。
```
self.dropout = nn.Dropout(dropout)
```
6. hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)  为什么用-2，-1，而不直接用0和1？我自己的代码中，使用0，1和-2，-1得到的东西是一样的。但是我在老师的范例中，使用0，1和-2，-1，想要尝试是否一样，结果0和1报错了。也许这就是老师使用-2，-1，而不使用0，1的原因？但是为什么0，1就不行，-2，-1就可以呢？？不太明白，网上也暂时没找到答案，可以再研究研究。
```
# 报错
IndexError: too many indices for tensor of dimension 3
```
8. encoder和decoder的n_layers要一样吗?老师的作业范例中是要求encoder和decoder的n_layers一样。但不知道是否一定要一样，我感觉应该也可以不一样。一样的话主要是可以减少参数数量，防止过拟合？不太确定，可以再研究研究
9.  为什么encoder可以一次得到，而decoder却要循环得到？具体的模型可以看作业要求的那个PPT，里面有模型的图形。
11. output维度是[batch, seq_len, vocab_size], 但是target的维度是[batch, seq_len]，他们是怎么算crossentropy的？？？？为什么结果还能跑通？？？回想了下，crossentropy的用法确实是这样的，但具体里面做了什么不太了解，可以再看看
12. .ipynb文件转换为.py文件: .ipynb文件夹下点开命令行
```
jupyter nbconvert --to script xxxxx.ipynb
```

# Homework9:Unsupervised Learning

## 要点
- nn.ConvTranspose2d():该函数是用来进行转置卷积的，它主要做了这几件事：首先，对输入的feature map进行padding操作，得到新的feature map；然后，随机初始化一定尺寸的卷积核；最后，用随机初始化的一定尺寸的卷积核在新的feature map上进行卷积操作。
```
nn.ConvTranspose2d(256, 128, 5, stride=1)
```
- p.numel():返回数组中元素的个数
```
if only_trainable:
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
else:
  return sum(p.numel() for p in model.parameters())
```
- torch.manual_seed(seed):设置CPU生成随机数的种子，方便下次复现实验结果
```
torch.manual_seed(0)
# 这样下面生成的随机数就固定是第0组随机数种子
```
- torch.cuda.manual_seed(seed)：为当前GPU设置随机种子
- torch.cuda.manual_seed_all(seed)：如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子
- np.random.seed(seed)
- random.seed()
- torch.backends.cudnn.benchmark：设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。注意：适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
- torch.backends.cudnn.deterministic：Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异。如果想要避免这种结果波动，设置：torch.backends.cudnn.deterministic = True。置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的，
```
def same_seeds(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
  np.random.seed(seed)  # Numpy module.
  random.seed(seed)  # Python random module.
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
```
- nn.ReLU(inplace=True):
- PCA:PCA适用于数据的线性降维。而核主成分分析(Kernel PCA, KPCA)可实现数据的非线性降维，用于处理线性不可分的数据集。KPCA的大致思路是：对于输入空间(Input space)中的矩阵X，我们先用一个非线性映射把X中的所有样本映射到一个高维甚至是无穷维的空间(称为特征空间，Feature space)，(使其线性可分)，然后在这个高维空间进行PCA降维。
  - kernel='rbf'：（高斯）径向基函数核（Radial basis function kernel），或称为RBF核，是一种常用的核函数。
  - PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
  - n_jobs:指定计算所用的进程数。等于-1的时候，表示cpu里的所有core进行工作。
```
from sklearn.decomposition import KernelPCA

# First Dimension Reduction
transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
kpca = transformer.fit_transform(latents)
print('First Reduction Shape:', kpca.shape)
```
- t-SNE:
```
from sklearn.manifold import TSNE

# Second Dimesnion Reduction
X_embedded = TSNE(n_components=2).fit_transform(kpca)
print('Second Reduction Shape:', X_embedded.shape)
```
- cluster
```
from sklearn.cluster import MiniBatchKMeans

# Clustering
pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
pred = [int(i) for i in pred.labels_]
pred = np.array(pred)
return pred, X_embedded
```



## Q&A
- 为什么要將圖片的數值介於 0~255 的 int 線性轉為 0～1 的 float？在python图像处理过程中，遇到的RGB图像的值是处于0-255之间的，为了更好的处理图像，通常会将图像值转变到0-1之间。这个处理的过程就是图像的uint8类型转变为float类型过程。
- 归一化和标准化的区别？归一化是将数据的值压缩到0到1之间，标准化是将数据所防伪均值是0，方差为1的状态。
- 为什么decoder的最后一层用tanh函数而前面都用ReLU？最后使用的激活函数是Tanh，这个激活函数能够将最后的输出转换到-1 ～1之间，这是因为我们输入的图片已经变换到了-1~1之间了，这里的输出必须和其对应。
- 为什么decoder只有反卷积，却没有反池化？
- np.transpose 和 permute？transpose是对numpy的ndarray，permute是对tensor
- 如何选择损失函数？autodencoder的损失函数为什么用nn.MSELoss()而不是用nn.CrossEntropyLoss()
- 如何选择激活函数？一般的神经网络都使用ReLU作为激活函数，但是在RNN中，使用ReLU的performance不太好，所以会使用tanh激活函数
- fit 和 transform？
- 图片一定要是0-1之间或是0-255之间才能显示？那为什么要转成-1~1之间呢？



# Homework10：Anomaly Detection

## KNN（K Nearest Neighbour)
- K-Nearest-Neighbor(KNN): 假設training data的label種類不多（e.g. < 20），然而因其為未知，可以猜測其為n，亦即假設training data有n群。先用K-means計算training data中的n個centroid，再用這n個centroid對testing data分群。應該可以觀察到，inlier data與所分到群的centroid的距離應較outlier的此距離來得小。
- 
### 要点
- MiniBatchKMeans():MiniBatchKmeans算法是K-Means算法的一种优化变种，采用小规模的数据子集减少计算时间，即每次训练使用的数据集是在训练算法的时候随机抽取的数据子集，同时试图优化目标函数。MiniBatchKmeans算法可以减少K-Means算法的收敛时间，而且产生的结果效果只是略差于K-Means算法。算法步骤如下：
  - 首先抽取部分数据集，使用K-Means算法构建出K个聚簇点的模型
  - 继续抽取训练数据集中的部分数据集样本数据，将其添加到模型中，分配给距离最近的聚簇中心点
  - 更新聚簇的中心点值（每次更新都只用抽取出来的部分数据集）
  - 循环迭代第二步和第三步的操作，直到中心点稳定或者达到迭代次数，停止计算操作
```

```
- MiniBatchKMeans类主要参数
  - n_clusters: 即我们的k值，和KMeans类的n_clusters意义一样。
  - max_iter：最大的迭代次数， 和KMeans类的max_iter意义一样。
  - n_init：用不同的初始化质心运行算法的次数。这里和KMeans类意义稍有不同，KMeans类里的n_init是用同样的训练集数据来跑不同的初始化质心从而运行算法。而MiniBatchKMeans类的n_init则是每次用不一样的采样数据集来跑不同的初始化质心运行算法。
  - batch_size：即用来跑Mini Batch KMeans算法的采样集的大小，默认是100.如果发现数据集的类别较多或者噪音点较多，需要增加这个值以达到较好的聚类效果。
  - init： 即初始值选择的方式，和KMeans类的init意义一样。
  - init_size: 用来做质心初始值候选的样本个数，默认是batch_size的3倍，一般用默认值就可以了。
  - reassignment_ratio: 某个类别质心被重新赋值的最大次数比例，这个和max_iter一样是为了控制算法运行时间的。这个比例是占样本总数的比例，乘以样本总数就得到了每个类别质心可以重新赋值的次数。如果取值较高的话算法收敛时间可能会增加，尤其是那些暂时拥有样本数较少的质心。默认是0.01。如果数据量不是超大的话，比如1w以下，建议使用默认值。如果数据量超过1w，类别又比较多，可能需要适当减少这个比例值。具体要根据训练集来决定。
  - max_no_improvement：即连续多少个Mini Batch没有改善聚类效果的话，就停止算法， 和reassignment_ratio， max_iter一样是为了控制算法运行时间的。默认是10.一般用默认值就足够了。
- MiniBatchKMeans类attr
  - cluster_centers_: ndarray of shape (n_clusters, n_features)聚类中心的坐标
  - labels_: int 每个点的标签
  - inertia_: float 与所选分区关联的惯性标准的值（如果 compute_labels 设置为 True）。惯性定义为样本到其最近邻居的平方距离之和。
  - n_iter_: int 处理的批次数。
  - counts_: ndarray of shape (n_clusters,) 每个簇的权重和。
  - init_size_: int 用于初始化的有效样本数。

- f1_score: 理论可见数据挖掘笔记上模型评估与选择那一页
```
score = f1_score(y_label, y_pred, average='micro')
```
- roc_auc_score:理论见笔记本
```
score = roc_auc_score(y_label, y_pred, average='micro')
```


## PCA
- PCA:首先計算training data的principle component，將testing data投影在這些component上，再將這些投影重建回原先space的向量。對重建的圖片和原圖計算MSE，inlier data的數值應該較outlier的數值為小。
```
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(x)
y_projected = pca.transform(y)
y_reconstructed = pca.inverse_transform(y_projected)

dist = np.sqrt(np.sum(np.square(y_reconstructed - y).reshape(len(y), -1), axis=1))
```


## Auto-Encoder
- VAE：Variational Autoencoder，变分自动编码器。VAE 模型是一种有趣的生成模型，与GAN相比，VAE 有更加完备的数学理论（引入了隐变量），理论推导更加显性，训练相对来说更加容易。VAE 可以从神经网络的角度或者概率图模型的角度来解释。VAE 全名叫 变分自编码器，是从之前的 auto-encoder 演变过来的，auto-encoder 也就是自编码器，自编码器，顾名思义，就是可以自己对自己进行编码，重构。所以 AE 模型一般都由两部分的网络构成，一部分称为 encoder, 从一个高维的输入映射到一个低维的隐变量上，另外一部分称为 decoder, 从低维的隐变量再映射回高维的输入。
- 



## Q&A
- nn.ReLU(True): nn.ReLU()函数默认inplace 默认是False。inplace = False 时,不会修改输入对象的值,而是返回一个新创建的对象,所以打印出对象存储地址不同,类似于C语言的值传递。inplace = True 时,会修改输入对象的值,所以打印出对象存储地址相同,类似于C语言的址传递。inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好。
- np.inf 表示正无穷大
- torch.optim.AdamW:
- roc_auc_score:见笔记本