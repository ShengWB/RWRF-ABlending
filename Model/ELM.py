import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

import warnings
warnings.filterwarnings(action='ignore')  # 忽略告警
# 读取数据
df = pd.read_excel('wheat_C_riskpre.xlsx')
# 构建特征和标签
X = df.drop(columns=['cvar'])  # 构建特征
y = df['cvar']  # 构建标签
# 数据集拆分：划分80%训练集和20%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
py_test=pd.DataFrame(y_test)

class ELMRegressor():
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape
        self.W = np.random.randn(n_features + 1, self.n_hidden)
        H = np.dot(np.concatenate((X, np.ones((n_samples, 1))), axis=1), self.W)
        H = 1 / (1 + np.exp(-H))  # 加上激活函数
        self.beta = np.dot(np.linalg.pinv(H), y)

    def predict(self, X):
        n_samples = X.shape[0]
        H = np.dot(np.concatenate((X, np.ones((n_samples, 1))), axis=1), self.W)
        H = 1 / (1 + np.exp(-H))  # 加上激活函数
        y_pred = np.dot(H, self.beta)
        return y_pred


def evaluation(y_test, y_predict):
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    r2 = r2_score(y_test, y_predict)
    return mse, mae, rmse, r2

clf=ELMRegressor(8)
clf.fit(X_train,y_train)
elm_pre=clf.predict(X_test)
elm_mse, elm_mae, elm_rmse, elm_r2=evaluation(elm_pre,y_test)
print(elm_mse,elm_mae,elm_rmse,elm_r2)

