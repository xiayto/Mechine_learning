# 集成学习的比较

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,GradientBoostingRegressor

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
## 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)

def notEmpty(s):
    return s != ''

## 加载数据
names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
path = "datas/boston_housing.data"
## 由于数据文件格式不统一，所以读取的时候，先按照一行一个字段属性读取数据，然后再安装每行数据进行处理
fd = pd.read_csv(path,header=None)
# print (fd.shape)
data = np.empty((len(fd), 14))
for i, d in enumerate(fd.values):#enumerate生成一列索 引i,d为其元素

    d = map(float, filter(notEmpty, d[0].split(' ')))#filter一个函数，一个list
    
    #根据函数结果是否为真，来过滤list中的项。
    data[i] = list(d)
    
## 分割数据
x, y = np.split(data, (13,), axis=1)
y = y.ravel() # 转换格式 拉直操作
# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28)

model_names = ["线性回归","Bagging","Adaboost","GBDT"]

pipes = [
    Pipeline([
        ('lr', Ridge())
    ])
]

parameters = [
    {
        "lr__alpha":[0.001, 0.003, 0.01, 0.03, 0.1, 0.5]
    }
]

gscv = GridSearchCV(pipes[0], param_grid=parameters[0], cv=10)
gscv.fit(x_train, y_train)
print ("score值:",gscv.best_score_,"最优参数列表:", gscv.best_params_)
# score值: 0.700917061126 最优参数列表: {'lr__alpha': 0.01}

pipes = [
    Pipeline([
        ('bagging', BaggingRegressor(Ridge(alpha=0.01), random_state=28))
    ]),
    Pipeline([
        ('Ada', AdaBoostRegressor(Ridge(alpha=0.01), random_state=28))
    ]),
    Pipeline([
        ('GBDT', GradientBoostingRegressor(random_state=28))
    ])
]

parameters = [
    {
        "bagging__n_estimators":[10, 20, 50, 100],
        "bagging__max_samples":[0.5, 0.7, 0.9],
        "bagging__max_features":[0.5, 0.7, 0.9]
    },
    {
        "Ada__n_estimators":[10, 20, 50, 100],
        "Ada__learning_rate":[0.01, 0.03, 0.001, 0.003]
    },
    {
        "GBDT__n_estimators":[10, 20, 50, 100],
        "GBDT__max_features":[0.5, 0.7, 0.9],
        "GBDT__learning_rate":[0.01, 0.03, 0.001, 0.003]
    }
]

for t in range(3):
    pipe = pipes[t]
    gscv = GridSearchCV(pipe, param_grid=parameters[t], cv=3)
    gscv.fit(x_train, y_train)
    print(model_names[t+1],': score值:',gscv.best_score_,"最优参数列表:", gscv.best_params_)
# 线性回归 : score值: 0.700917061126 最优参数列表: {'lr__alpha': 0.01}
# Bagging : score值: 0.713536483116 最优参数列表: {'bagging__max_features': 0.9, 'bagging__max_samples': 0.7, 'bagging__n_estimators': 100}
# Adaboost : score值: 0.71727230805 最优参数列表: {'Ada__learning_rate': 0.003, 'Ada__n_estimators': 100}
# GBDT : score值: 0.843742602375 最优参数列表: {'GBDT__learning_rate': 0.03, 'GBDT__max_features': 0.5, 'GBDT__n_estimators': 100}
