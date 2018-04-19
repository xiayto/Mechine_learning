import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

# 自定义评价函数：
from sklearn.metrics import mean_squared_log_error
def evalerror(preds, dtrain):       # written by myself
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', mean_squared_log_error(preds,labels)

# 找到合适的训练轮数
def modelfit(clf, x_train, y_train, cv_folds, early_stopping_rounds, feval):
    dtrain = xgb.DMatrix(x_train, y_train)
    xgb_params = clf.get_xgb_params()
    cvresult = xgb.cv(xgb_params, dtrain, nfold=cv_folds, num_boost_round=2000,
                      early_stopping_rounds=early_stopping_rounds)

    clf_xgb = xgb.train(xgb_params, dtrain, num_boost_round=cvresult.shape[0])
    fscore = clf_xgb.get_fscore()

    return cvresult.shape[0], fscore

def find_params(para_dict, estimator, x_train, y_train):
    gsearch = GridSearchCV(estimator, param_grid=para_dict, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    gsearch.fit(x_train, y_train)
    return gsearch.best_params_, gsearch.best_score_


def run_find(x_train, y_train, i, x_predict):

    # 找到合适的参数调优的估计器数目

    clf = XGBRegressor(
        objective='reg:linear',
        learning_rate=0.1,  # [默认是0.3]学习率类似，调小能减轻过拟合，经典值是0.01-0.2
        gamma=0,  # 在节点分裂时，只有在分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。这个参数值越大，算法越保守。
        subsample=0.8,  # 随机采样比例，0.5-1 小欠拟合，大过拟合
        colsample_bytree=0.8,  # 训练每棵树时用来训练的特征的比例
        reg_alpha=1,  # [默认是1] 权重的L1正则化项
        reg_lambda=1,  # [默认是1] 权重的L2正则化项
        max_depth=10,  # [默认是6] 树的最大深度，这个值也是用来避免过拟合的3-10
        min_child_weight=1,  # [默认是1]决定最小叶子节点样本权重和。当它的值较大时，可以避免模型学习到局部的特殊样本。但如果这个值过高，会导致欠拟合。
    )
    nums, fscore= modelfit(clf, x_train, y_train, cv_folds=5, early_stopping_rounds=50, feval=evalerror)
    print('test_estimators:', nums)
    clf.set_params(n_estimators=nums)

    # 1 先对 max_depth和min_child_weight 这两个比较重要的参数进行调优
    ## 粗调：
    param_test1 = {
        'max_depth': [i for i in range(3, 12, 2)],
        'min_child_weight': [i for i in range(1, 10, 2)]
    }
    best_params, best_score= find_params(param_test1, clf, x_train, y_train)
    print('model',i,':')
    print(best_params, ':best_score:', best_score)

    ## 精调：
    max_d = best_params['max_depth']
    min_cw = best_params['min_child_weight']
    param_test2 = {
        'max_depth': [max_d-1, max_d, max_d+1],
        'min_child_weight': [min_cw-1, min_cw, min_cw+1]
    }
    best_params, best_score= find_params(param_test2, clf, x_train, y_train)
    clf.set_params(max_depth=best_params['max_depth'], min_child_weight=best_params['min_child_weight'])
    print('model', i, ':')
    print(best_params, ':best_score:', best_score)

    # 2 对 gamma 进行调参：
    ## 粗调：
    param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 10, 2)]
    }
    best_params, best_score= find_params(param_test3, clf, x_train, y_train)
    print('model', i, ':')
    print(best_params, ':best_score:', best_score)

    ## 精调:
    b_gamma = best_params['gamma']
    param_test4 = {
        'gamma': [b_gamma, b_gamma+0.1, b_gamma+0.2]
    }
    best_params, best_score = find_params(param_test4, clf, x_train, y_train)
    clf.set_params(gamma = best_params['gamma'])
    print('model', i, ':')
    print(best_params, ':best_score:', best_score)

    # 3 对subsample和colsample_bytree进行调参
    ## 粗调
    param_test5 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    best_params, best_score = find_params(param_test5, clf, x_train, y_train)
    print('model', i, ':')
    print(best_params, ':best_score:', best_score)

    ## 精调
    b_subsample = best_params['subsample']
    b_colsample_bytree = best_params['colsample_bytree']
    param_test6 = {
        'subsample': [b_subsample-0.05, b_subsample, b_subsample+0.05],
        'colsample_bytree': [b_colsample_bytree-0.05, b_colsample_bytree, b_colsample_bytree+0.05]
    }
    best_params, best_score = find_params(param_test6, clf, x_train, y_train)
    clf.set_params(subsample=best_params['subsample'], colsample_bytree=best_params['colsample_bytree'])
    print('model', i, ':')
    print(best_params, ':best_score:', best_score)

    # 4 对 reg_alpha和lambda 进行调节
    ## 粗调
    param_test7 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 2],
        'reg_lambda': [1e-5, 1e-2, 0.1, 1, 2]
    }
    best_params, best_score = find_params(param_test7, clf, x_train, y_train)
    print('model', i, ':')
    print(best_params, ':best_score:', best_score)

    ## 精调
    b_alp = best_params['reg_alpha']
    b_lam = best_params['reg_lambda']
    param_test8 = {
        'reg_alpha': [b_alp, 2*b_alp, 3*b_alp],
        'reg_lambda': [b_lam, 2*b_lam, 3*b_lam]
    }
    best_params, best_score = find_params(param_test7, clf, x_train, y_train)
    clf.set_params(reg_alpha=best_params['reg_alpha'], reg_lambda=best_params['reg_lambda'])
    print('model', i, ':')
    print(best_params, ':best_score:', best_score)

    # 5 调小learning_rate, 提高迭代次数
    clf.set_params(learning_rate=0.01)
    nums, fscore= modelfit(clf, x_train, y_train, cv_folds=5, early_stopping_rounds=50, feval=evalerror)
    clf.set_params(n_estimators=nums)

    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_predict)

    return y_predict, fscore




if __name__ == '__main__':
    path_train = './x_train.csv'
    path_predict = './predict.csv'

    df_train  = pd.read_csv(path_train, sep=',', encoding='gbk', header=None)
    df_predict = pd.read_csv(path_predict, sep=',', encoding='gbk', header=None)

    x_train = df_train.values[:,1:-5]
    y_train_all = df_train.values[:, -5:]

    x_predict = df_predict.values[:,1:]

    y_predict = []
    for i in range(5):
        y_predict_i, fscore= run_find(x_train, y_train_all[:, i], i, x_predict)
        y_predict.append(y_predict_i.reshape((-1,1)))

        fscore= dict(sorted(fscore.items(), key=lambda item: item[1]))

        f = open('feature_importance.txt', 'a')
        f.write('model'+str(i)+':')
        f.write(str(fscore))
        f.write('\n')
        f.close()


    y_predict = np.concatenate(y_predict, axis=1)
    y_predict = np.concatenate([df_predict.iloc[:,0].values.reshape((-1,1)), y_predict], axis=1)
    predict = pd.DataFrame(y_predict)
    predict.to_csv('./submit.csv', index=None, header=None)








