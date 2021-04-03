#数据导入和可视化
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
iris = sns.load_dataset("iris")

iris.plot(kind="scatter", x="sepal_length", y="sepal_width")
sns.pairplot(iris,hue='species')
sns.plt.show()

'''
2-nd logistic regression using sklearn
'''
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

# log-regression lib model
y = iris.species
features = ['sepal_length','sepal_width','petal_length','petal_width']
X = iris[features]
log_model = LogisticRegression()
m = np.shape(X)[0]

# 10-folds CV
y_pred = cross_val_predict(log_model, X, y, cv=10)
print(metrics.accuracy_score(y, y_pred))

# LOOCV
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
accuracy = 0;
for train, test in loo.split(X):
    log_model.fit(X[train], y[train])  # fitting
    y_p = log_model.predict(X[test])
    if y_p == y[test] : accuracy += 1
print(accuracy / np.shape(X)[0])
