import pandas as pd
import numpy as np

data = pd.read_csv("D:/Data/mardia.dat", header=None)

f = open("D:/Data/mardia.dat")
L = []
for lines in f.readlines():
    a = lines.split()
    L.append(a)
data = pd.DataFrame(L)

for i in range(0, 5):
    data[i] = pd.to_numeric(data[i])
"""
设置子函数


"""


def random_sample(n):
    rd = list(n * np.random.rand(n))
    index = []
    for a in rd:
        index.append(int(a))
        xb = data.iloc[index]
    return xb


n = data.shape[0]  # 样本量
cov_data = data.cov()  # 协方差
data_val, data_xvec = np.linalg.eig(np.array(cov_data))  # 特征值,特征向量
theta = data_val.max() / data_val.sum()  # 计算最大特征值/总特征值和

Nr = 200
dataval_Nr = []
theta_Nr = []
for i in range(Nr):
    xb = random_sample(n)
    cov_xb = xb.cov()
    xb_val, xb_vec = np.linalg.eig(np.array(cov_xb))

    xb_theta = xb_val.max() / xb_val.sum()

    dataval_Nr.append(list(xb_val))
    theta_Nr.append(xb_theta)

mean_theta = sum(theta_Nr) / Nr
var_theta = sum(np.array(theta_Nr) ** 2) / Nr - mean_theta ** 2  # 计算方差
SE_theta = np.sqrt(var_theta)

import pandas as pd
import numpy as np

h = [576, 635, 558, 578, 666, 580, 555, 661, 651, 605, 653, 575, 545, 572, 593]
w = [3.39, 3.30, 2.81, 3.03, 3.44, 3.07, 3.00, 3.43, 3.36, 3.13, 3.12, 2.74, 2.76, 2.88, 2.96]


def random_sample(h,w, n):
    rd = list(n * np.random.rand(n))
    index = []
    xb = []
    for a in rd:
        index.append(int(a))
        xh = np.array(h)[index]
        xw =np.array(w)[index]
    return xh,xw


Nr = 1600
corr_Nr = []
n = len(h)
for i in range(Nr):
    xh,xw = random_sample(h,w,n)
    corr_Nr_i = np.corrcoef(xh, xw)[1][0]
    corr_Nr.append(corr_Nr_i)
seb = np.std(corr_Nr)

from matplotlib import pyplot as plt
plt.hist(corr_Nr)