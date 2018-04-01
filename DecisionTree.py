import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest #特征选择
from sklearn.feature_selection import chi2 #卡方统计量

from sklearn.preprocessing import MinMaxScaler  #数据归一化
from sklearn.decomposition import PCA #主成分分析
from sklearn.model_selection import GridSearchCV #网格搜索交叉验证，用于选择最优的参数?

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=FutureWarning)

#读取数据
path = './datas/iris.data'  
data = pd.read_csv(path, header=None)
x=data[list(range(4))]#获取X变量
y=pd.Categorical(data[4]).codes #把Y转换成分类型的0,1,2

#数据进行分割（训练数据和测试数据）
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=14)

# 利用管道进行系列操作
# 利用网格搜索进行参数选择

pipe = Pipeline([
            ('mms', MinMaxScaler()),
            ('skb', SelectKBest(chi2)),
            ('pca', PCA()),
            ('decision', DecisionTreeClassifier(random_state=0))
        ])
parameters = {
    "skb__k": [1,2,3,4],
    "pca__n_components": [0.5,0.99],#设置为浮点数代表主成分方差所占最小比例的阈值，这里不建议设置为数值，思考一下？
    "decision__criterion": ["gini", "entropy"],
    "decision__max_depth": [1,2,3,4,5,6,7,8,9,10]
}

x_train1, x_test1, y_train1, y_test1 = x_train, x_test, y_train, y_test
#模型构建：通过网格交叉验证，寻找最优参数列表， param_grid可选参数列表，cv：进行几折交叉验证
gscv = GridSearchCV(pipe, param_grid=parameters,cv=3)
#模型训练
gscv.fit(x_train1, y_train1)
#算法的最优解
print("最优参数列表:", gscv.best_params_)
print("score值：",gscv.best_score_)
print("最优模型:", end='')
print(gscv.best_estimator_)
#预测值
y_test_hat1 = gscv.predict(x_test1)

#模型结果的评估
y_test1 = y_test.reshape(-1)
result = (y_test1 == y_test_hat1)
print ("准确率:%.2f%%" % (np.mean(result) * 100))
