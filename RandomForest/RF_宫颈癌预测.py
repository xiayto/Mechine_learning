import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import label_binarize
from sklearn import metrics

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

names = [u'Age', u'Number of sexual partners', u'First sexual intercourse',
       u'Num of pregnancies', u'Smokes', u'Smokes (years)',
       u'Smokes (packs/year)', u'Hormonal Contraceptives',
       u'Hormonal Contraceptives (years)', u'IUD', u'IUD (years)', u'STDs',
       u'STDs (number)', u'STDs:condylomatosis',
       u'STDs:cervical condylomatosis', u'STDs:vaginal condylomatosis',
       u'STDs:vulvo-perineal condylomatosis', u'STDs:syphilis',
       u'STDs:pelvic inflammatory disease', u'STDs:genital herpes',
       u'STDs:molluscum contagiosum', u'STDs:AIDS', u'STDs:HIV',
       u'STDs:Hepatitis B', u'STDs:HPV', u'STDs: Number of diagnosis',
       u'STDs: Time since first diagnosis', u'STDs: Time since last diagnosis',
       u'Dx:Cancer', u'Dx:CIN', u'Dx:HPV', u'Dx', u'Hinselmann', u'Schiller',
       u'Citology', u'Biopsy']#df.columns
path = "datas/risk_factors_cervical_cancer.csv"  # 数据文件路径
data = pd.read_csv(path)

## 模型存在多个需要预测的y值，如果是这种情况下，简单来讲可以直接模型构建，在模型内部会单独的处理每个需要预测的y值，相当于对每个y创建一个模型
X = data[names[0:-4]]
Y = data[names[-4:]]
X.head(1)#随机森林可以处理多个目标变量的情况

#空值的处理
X = X.replace("?", np.NAN)
# 使用Imputer给定缺省值，默认的是以mean
# 对于缺省值，进行数据填充；默认是以列/特征的均值填充
imputer = Imputer(missing_values="NaN")
X = imputer.fit_transform(X, Y)

#数据分割
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print ("训练样本数量:%d,特征属性数目:%d,目标属性数目:%d" % (x_train.shape[0],x_train.shape[1],y_train.shape[1]))
print ("测试样本数量:%d" % x_test.shape[0])

pipes =[
    Pipeline([
        ('mms', MinMaxScaler()),
        ('rf' , RandomForestClassifier(random_state=0, criterion='gini'))
    ]),
    Pipeline([
        ('rf' , RandomForestClassifier(random_state=0, criterion='gini'))
    ])
]

parameters = [
    {
        "rf__n_estimators":[10, 50, 100],
        "rf__max_depth": [1, 2, 3, 5, 10]
    },
    {
        "rf__n_estimators":[10, 50, 100],
        "rf__max_depth": [1, 2, 3, 5, 10]
    }
]

for t in range(2):
    pipe = pipes[t]
    gscv = GridSearchCV(pipe, param_grid=parameters[t], cv=3)
    gscv.fit(x_train, y_train)
    print (t,"score值:",gscv.best_score_,"最优参数列表:", gscv.best_params_)

# 用最优的参数构造模型
ss = MinMaxScaler()
x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)

forest = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=1, random_state=0)
forest.fit(x_train, y_train)

score = forest.score(x_test, y_test)
print ("准确率:%.2f%%" % (score * 100))
