"""
书上例题
2021-3-19感知机V1
"""
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import randint, sample

data = np.array([[3, 3, 1], [4, 3, 1], [1, 1, -1]])


class Model:
    def __init__(self):
        self.w = np.zeros(data.shape[1]-1)
        self.b = 0
        self.a = 0.1
        # self.data = data

    def fit(self, data):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i in range(data.shape[0]):
                yi = data[i][-1]
                xi = data[i][:-1]
                judge = yi * (np.dot(self.w, xi.T) + self.b)
                if judge <= 0:  # 判断是否都分对
                    wrong_count += 1
                    # index = sample(range(1, len(data)), 1)  #如果有分错的，从中随机取一个
                    self.w = self.w + self.a * yi * xi  #迭代更新公式
                    self.b = self.b + self.a * yi   #迭代更新公式
                    print("self.w:", self.w, "b", self.b)


            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!',self.w,self.b


perceptron = Model()
perceptron.fit(data)


#绘图
x_points = np.linspace(0, 8, 6)
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]  #构建感知机超平面
plt.plot(x_points, y_)

plt.plot(data[:2, 0], data[:2, 1], 'bo', color='blue', label='0')
plt.plot(data[2, 0], data[2, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()


##书上例2.1对偶形式
class Model_duiou:
    def __init__(self):
        self.a = np.zeros(len(data))
        self.rate =1
        self.b = 0

    def sum_a(self):
        sigma_a = np.zeros(len(data) - 1)
        for j in range(len(data)):
            sigma_a += self.a[j] * data[j][-1] * data[j][:-1]
        return sigma_a


    def fit(self,data):
        is_wrong =False

        while not is_wrong:
            wrong_count=0
            for i in range(len(data)):
                yi = data[i][-1]
                xi = data[i][:-1]
                sigma_a = self.sum_a()
                judge = yi*(np.dot(sigma_a,xi.T)+self.b)
                if judge<=0:
                    wrong_count +=1
                    self.a[i]=self.a[i]+1
                    self.b = self.b+yi
                    print("self.a:",self.a, "b", self.b)
            if wrong_count ==0:
                is_wrong =True
        return 'Perceptron Model!'

perceptron_duiou = Model_duiou()
perceptron_duiou.fit(data)



#Iris数据集测试

# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
label = pd.DataFrame(iris.target)
data_all= df.join(label)[:100]#对于二分类问题（前50为负，后50为正）
data = data_all.iloc[:100, [0, 1, -1]]
for i in range(len(data)):
   if data[i][-1] ==0:
      data[i][-1]=-1

perceptron = Model()
perceptron.fit(data)


#import sklearn
from sklearn.linear_model import Perceptron


X = data_all.iloc[:, [0, 1]]
y = data_all.iloc[:,-1]

clf = Perceptron(fit_intercept=True,
                 max_iter=1000,
                 shuffle=True)

clf.fit(X, y)

# Weights assigned to the features.
print(clf.coef_)

# 截距 Constants in decision function.
print(clf.intercept_)
# 画布大小
plt.figure(figsize=(10,10))

# 中文标题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('鸢尾花线性数据示例')

plt.scatter(data[:50, 0], data[:50, 1], c='b', label='Iris-setosa',)
plt.scatter(data[50:100, 0], data[50:100, 1], c='orange', label='Iris-versicolor')

# 画感知机的线
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

# 其他部分
plt.legend()  # 显示图例
plt.grid(False)  # 不显示网格
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()



import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
x = np.array([2.4,2.4,2.4,2.2,2.1,1.5,2.3,2.3,2.5,1.9,1.7,2.2,1.8,3.2,3.2,2.7,
              2.2,2.2,1.9,1.9,1.8,2.7,3.0,2.3,2.0,2.0,2.9,2.9,2.7,2.7,2.3,2.6,2.4,
              1.8,1.7,1.5,1.4,2.1,3.3,3.5,3.5,3.1,2.6,2.1,2.4,3.4,3.0,2.9])
data = pd.DataFrame(x)
arima = ARIMA(data, (1,0,0)).fit()
arima.bic
